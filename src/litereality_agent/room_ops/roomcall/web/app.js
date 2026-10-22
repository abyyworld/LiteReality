/* Room call client: one shared MuJoCo room in the browser, for the people on a video call.

   The server owns the physics. This page only draws what it is told (the "state" stream), says
   where you are ("pose") and what you are holding ("grab" / "move" / "release"), and relays
   WebRTC signalling so webcams can ride on the avatars peer to peer. Everything on the wire is in
   MuJoCo WORLD coordinates (Z-up, metres); the conversion to three.js (Y-up) happens in exactly
   two places — the `world` group's rotation for the room itself, and mjToThree / threeToMj for
   the camera, avatars and pointer rays.

   With the camera on, each person is sent as a hologram: cut out of their background on their own
   device and stood life-size in the room (see "hologram", sender, and `holoMaterial`, receiver).

   CSP: the page's only inline script is the import map; there are no inline handlers or style
   attributes, so every listener is attached here and styling goes through classes or CSSOM. */

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { acceleratedRaycast, computeBoundsTree, disposeBoundsTree } from 'three-mesh-bvh';

// Picking walks a BVH instead of every triangle: a scanned room's merged static shell is up to a
// million triangles and its bounds contain the camera, so a plain raycast tests all of them on
// every click (~90 ms). Meshes without a bounds tree (avatars) fall back to the stock raycast.
THREE.BufferGeometry.prototype.computeBoundsTree = computeBoundsTree;
THREE.BufferGeometry.prototype.disposeBoundsTree = disposeBoundsTree;
THREE.Mesh.prototype.raycast = acceleratedRaycast;

/* ------------------------------------------------------------------ tuning */
const ROOM_RE = /^[A-Za-z0-9_-]{1,64}$/;
const EYE = 1.6;                                   // eye height over floor_z, metres
const WALK = 1.4, HURRY = 2.8, TURN = 1.9;         // m/s, m/s, rad/s
const LOOK_MOUSE = 0.0042, LOOK_TOUCH = 0.0055;    // rad per pixel
const PITCH_LIMIT = 1.35;
const POSE_HZ = 15, MOVE_HZ = 30;
const HOLD_MIN = 0.3, HOLD_MAX = 6;
const BODY_SMOOTH = 20, AVATAR_SMOOTH = 12;        // 1/s for the exponential smoothing
const HOLD_MS = 350, TAP_SLOP = 10;                // touch press-and-hold to grab
const FACE_H = 0.4, LABEL_Y = 0.36;                // webcam plane height at <= 1.5 m, name tag over the eye
// The hologram. MediaPipe's runtime comes from jsDelivr at an exact version (like three.js), its
// model from our own server. 0.10.35, not 1.x: from 1.0 the runtime also posts usage metrics to
// Google (odml.pa.googleapis.com; see its README's privacy notice), which this page neither wants
// nor asks people's consent for, and which the CSP would block anyway. Every frame sent is
// colour | mask, HOLO_W x HOLO_H per half: 8:3, which no webcam is (they are 9:16 to 16:9), so
// anything HOLO_PACKED wide or wider is a cut-out.
const MP_BASE = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35';
const SEG_MODEL = 'selfie_segmenter.tflite';
const HOLO_W = 320, HOLO_H = 240, HOLO_PACKED = 2.3;
const HOLO_KBPS = 800;                             // per peer; half the frame is flat grey
const HOLO_STALE_MS = 140;                         // a mask older than ~2 camera frames no longer fits
const HOLO_IDLE_MS = 400;                          // how often to look for someone while nobody is in frame
const HEAD_M = 0.23, EYES_IN_HEAD = 0.45;          // crown to chin, and the eyes' depth into that
const HAND_SIDE = 0.2, HAND_DROP = 0.45;           // where a grab line starts: their right hand
// A laptop webcam's usual framing (fractions of the image), until the sender measures its own:
// head top 15 % down, head 35 % of the height, centred. Life-size, that puts the image top
// ~0.2 m above the eyes.
const HOLO_FRAMING = { top: 0.15, head: 0.35, cx: 0.5 };
const STATE_DEAD_MS = 5000, MAX_ICE_RESTARTS = 3;
const STUN = [{ urls: 'stun:stun.l.google.com:19302' }];
const ZOOM_SDK = 'https://appssdk.zoom.us/sdk.min.js';
const COLORS = ['#2f6fde', '#e0573e', '#1f9d5c', '#c2410c', '#8b5cf6', '#db2777', '#0891b2', '#ca8a04'];
const VIDEO = { width: { ideal: 320 }, height: { ideal: 240 }, frameRate: { ideal: 15, max: 24 }, facingMode: 'user' };
const AUDIO = { echoCancellation: true, noiseSuppression: true, autoGainControl: true };
const IS_TOUCH = matchMedia('(pointer: coarse)').matches || 'ontouchstart' in window;

/* ------------------------------------------------------------------ small helpers */
const $ = (id) => document.getElementById(id);
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const r3 = (v) => Math.round(v * 1000) / 1000;
const finite = (a) => Array.isArray(a) && a.every(Number.isFinite);
const smooth = (rate, dt) => 1 - Math.exp(-dt * rate);
const wrapAngle = (a) => Math.atan2(Math.sin(a), Math.cos(a));

/** MuJoCo (x, y, z) -> three (x, z, -y): what the world group's -PI/2 X rotation does. */
function mjToThree(x, y, z, out = new THREE.Vector3()) { return out.set(x, z, -y); }
/** three (X, Y, Z) -> MuJoCo [X, -Z, Y], rounded for the wire. */
function threeToMj(v) { return [r3(v.x), r3(-v.z), r3(v.y)]; }

const store = {
  get(k, d) { try { const v = localStorage.getItem('roomcall.' + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem('roomcall.' + k, JSON.stringify(v)); } catch { /* private mode */ } },
};

function withTimeout(p, ms) {
  return Promise.race([p, new Promise((_, rej) => setTimeout(() => rej(new Error('timeout')), ms))]);
}

function loadScript(src, ms = 8000) {
  return withTimeout(new Promise((res, rej) => {
    const s = document.createElement('script');
    s.src = src; s.async = true;
    s.addEventListener('load', res); s.addEventListener('error', () => rej(new Error('load ' + src)));
    document.head.appendChild(s);
  }), ms);
}

/** First 16 hex digits of SHA-256, so a meeting UUID (which may hold '+' and '/') becomes a
 *  room id the server accepts — and everyone in the meeting computes the same one. */
async function shortHash(text) {
  const bytes = new TextEncoder().encode(text);
  if (crypto?.subtle) {
    const d = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
    return Array.from(d.slice(0, 8), (b) => b.toString(16).padStart(2, '0')).join('');
  }
  let h1 = 0x811c9dc5, h2 = 0x01000193;           // not a secure context: two FNV-1a lanes
  for (const b of bytes) { h1 = Math.imul(h1 ^ b, 16777619); h2 = Math.imul(h2 ^ b, 2246822519); }
  return ((h1 >>> 0).toString(16).padStart(8, '0') + (h2 >>> 0).toString(16).padStart(8, '0'));
}

function toast(msg, kind = '') {
  const el = document.createElement('div');
  el.className = 'toast ' + kind;
  el.textContent = msg;
  $('toasts').appendChild(el);
  while ($('toasts').children.length > 3) $('toasts').firstChild.remove();
  setTimeout(() => { el.classList.add('out'); setTimeout(() => el.remove(), 400); }, kind === 'error' ? 6000 : 3200);
}

/* ------------------------------------------------------------------ page state */
const params = new URLSearchParams(location.search);
const qRoom = params.get('room');
let roomId = qRoom && ROOM_RE.test(qRoom) ? qRoom : 'lobby';
let zoom = null;                                   // {context, room, name} inside the Zoom client

const me = {
  id: null,
  name: String(store.get('name', '')).slice(0, 40),
  color: COLORS.includes(store.get('color', '')) ? store.get('color', '') : COLORS[Math.floor(Math.random() * COLORS.length)],
  media: { camera: false, mic: false },
};
const peers = new Map();                           // id -> peer record (see addPeer)
const canMedia = !!(window.isSecureContext && navigator.mediaDevices?.getUserMedia);

/* ------------------------------------------------------------------ renderer and scene */
const canvas = $('view');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: devicePixelRatio < 2, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight, false);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.NeutralToneMapping;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(70, innerWidth / innerHeight, 0.05, 200);
camera.rotation.order = 'YXZ';
scene.add(camera);

const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
scene.add(new THREE.HemisphereLight(0xffffff, 0x8d8a84, 0.9));
const sun = new THREE.DirectionalLight(0xffffff, 1.1);
sun.position.set(3, 8, 2);
scene.add(sun);

/** The whole room, in MuJoCo coordinates. Rotating this group -PI/2 about X maps MuJoCo Z-up onto
 *  three's Y-up, so body poses from the server are applied to its children untouched. */
const world = new THREE.Group();
world.rotation.x = -Math.PI / 2;
scene.add(world);

function applyTheme() {
  const bg = getComputedStyle(document.documentElement).getPropertyValue('--scene-bg').trim() || '#dfe5e8';
  scene.background = new THREE.Color(bg);
}
applyTheme();
matchMedia('(prefers-color-scheme: dark)').addEventListener?.('change', applyTheme);

addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight, false);
});

/* ------------------------------------------------------------------ the room, one version at a time */
// The server publishes a room in VERSIONS — a quick box room seconds after a scan, better ones as
// they are ready — and swaps each room's physics to a new one without anyone leaving. Body ids
// only mean something within one version, so every "state" frame says which version it is for
// ("v"), and the page follows in two steps:
//   loadVersion(job)       fetch scene.json?v= and its glb, parse and prepare it (BVH, per-body
//                          materials, textures uploaded, shaders compiled) off to the side,
//                          touching nothing on screen;
//   installVersion(next)   the swap itself, synchronous, between two frames.
// A prepared version is installed at the first state frame the server sends for it, so the room
// changes for everyone at the moment its physics does; a frame for a version not on screen yet
// is kept (the latest one) and applied, snapping, when that version goes in. Only one load runs
// at a time, for where the server said the room is going ("scene") or else where it is (the
// frames): a newer target abandons an older load, so a burst of versions costs one download.
let sceneRoot = null, sceneReady = false, loadFrac = 0;
let bodies = new Map();                            // id -> {id, name, node, tp, tq, has, mats}, on screen
let shown = null;                                  // the version on screen: {v, manifest, root, bodies, byName}
const bounds = { minX: -5, maxX: 5, minZ: -5, maxZ: 5 };   // three-space walk limits
let eyeY = EYE;

const ver = {
  room: undefined,      // the version the room's physics is on: welcome.v, then each frame's v
  next: undefined,      // where a "scene" message said it is going, until it gets there
  nextAt: 0,
  job: null,            // the one load in flight: {v, ctrl, stage, progress, fetched, prep}
  frame: null,          // the latest state frame for a version not on screen
  gone: new Set(),      // versions the server answered 404 for (until it names them again)
  fails: 0, failedV: undefined, retryAt: 0,
};
// A version the server announced but the room has not reached by then failed there (the server
// keeps the old one): stop waiting for it and follow the frames.
const NEXT_TIMEOUT_MS = 20000;
const MAX_LOAD_FAILS = 3;
// Phones: hold the new room's bytes only, and free the old room's GPU memory before parsing it.
const LOW_MEMORY = matchMedia('(pointer: coarse)').matches || (navigator.deviceMemory || 8) <= 4;
const STAGE_TAGS = { quick: 'quick preview', objects: 'preview' };
const versionOf = (m) => (typeof m?.v === 'string' && m.v ? m.v : undefined);

function setLoad(frac, text) {
  loadFrac = clamp(frac, 0, 1);
  const pct = Math.round(loadFrac * 100);
  $('loadBar').style.width = pct + '%';
  $('loadProgress').setAttribute('aria-valuenow', String(pct));
  if (text) $('loadText').textContent = text;
}

/** A task boundary, so a long preparation lets frames through. A hidden tab renders nothing and
 *  has its timers throttled to ~1 Hz, so there it just keeps going. */
const breathe = () => (document.hidden ? Promise.resolve() : new Promise((r) => setTimeout(r, 0)));

function httpError(what, status) {
  const err = new Error(`${what}: HTTP ${status}`);
  err.gone = status === 404;
  return err;
}

/** The whole body, with progress; aborted by `signal`. */
async function fetchBytes(url, signal, onProgress) {
  const res = await fetch(url, { signal });        // ?h= glbs are immutable: a repeat comes from cache
  if (!res.ok) throw httpError('scene.glb', res.status);
  const total = Number(res.headers.get('Content-Length')) || 0;
  if (!res.body?.getReader) return res.arrayBuffer();
  const reader = res.body.getReader(), parts = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value); got += value.length;
    onProgress(got, total);
  }
  const out = new Uint8Array(got);
  let at = 0;
  for (const p of parts) { out.set(p, at); at += p.length; }
  return out.buffer;
}

/** Everything a load must check after each await: is it still the load we want? */
const current = (job) => ver.job === job;

function jobProgress(job, frac, text) {
  job.progress = clamp(frac, 0, 1);
  if (!shown) setLoad(frac, text);                // the first room: the join card shows it
}

/** Step 1 of a load: the manifest and the glb bytes. `job.v` null means "whatever is live" —
 *  the page's first load, before any welcome says which version the room is on. */
async function fetchVersion(job) {
  jobProgress(job, 0.02, 'Fetching the room…');
  const url = job.v == null ? 'scene.json' : 'scene.json?v=' + encodeURIComponent(job.v);
  const res = await fetch(url, { cache: 'no-cache', signal: job.ctrl.signal });
  if (!res.ok) throw httpError('scene.json', res.status);
  const man = await res.json();
  if (!man || typeof man !== 'object') throw new Error('scene.json: not a manifest');
  if (job.v === null && versionOf(man)) {
    job.v = versionOf(man);                       // now it is a load of that version, and is
    reconsider();                                 // abandoned here if the room wants another
    if (!current(job)) return null;
  }
  const buf = await fetchBytes(man.glb || 'scene.glb', job.ctrl.signal, (got, total) => {
    const mb = `Loading the room… ${(got / 1e6).toFixed(1)} MB`;
    jobProgress(job, total ? 0.05 + 0.85 * got / total : Math.min(0.85, job.progress + 0.01), mb);
  });
  return { v: versionOf(man) ?? job.v ?? undefined, manifest: man, buf };
}

/** Step 2: parse and prepare, all off screen. Returns null (having freed what it built) if the
 *  load was abandoned meanwhile. */
async function loadVersion(job, f) {
  jobProgress(job, 0.9, 'Preparing the room…');
  const gltf = await new GLTFLoader().parseAsync(f.buf, '');
  const root = gltf.scene;
  if (!current(job)) { disposeTree(root); return null; }
  const meshes = [];
  root.traverse((o) => { if (o.isMesh) meshes.push(o); });
  for (const o of meshes) {
    for (const m of [].concat(o.material)) {
      // MuJoCo has no PBR metal workflow worth trusting; a stray metalness of 1 renders black.
      if (m && 'metalness' in m && !m.metalnessMap) m.metalness = Math.min(m.metalness, 0.2);
    }
  }
  const recs = new Map(), byName = new Map();
  for (const b of f.manifest.bodies || []) {
    const node = root.getObjectByName(b.node || `body_${b.id}`);
    if (!node || !Number.isInteger(b.id)) continue;
    const rec = { id: b.id, name: String(b.name ?? ''), node, tp: node.position.clone(), tq: node.quaternion.clone(), has: false, mats: [] };
    node.traverse((o) => {
      if (!o.isMesh) return;
      // Per-body material copies so a hover highlight never bleeds onto another object.
      o.material = Array.isArray(o.material) ? o.material.map((m) => m.clone()) : o.material.clone();
      for (const m of [].concat(o.material)) if (m.emissive) { m.userData.emissive0 = m.emissive.clone(); rec.mats.push(m); }
      o.userData.bodyId = b.id;
    });
    recs.set(b.id, rec);
    if (rec.name) byName.set(rec.name, b.id);
  }
  // Bounds trees, a few meshes per task: a scanned room's static shell alone can be a million
  // triangles. Built in each mesh's own frame, so they stay valid while bodies move; a geometry
  // the BVH cannot take keeps the plain (slow but correct) raycast.
  let t0 = performance.now();
  for (const o of meshes) {
    try { if (!o.geometry.boundsTree) o.geometry.computeBoundsTree(); } catch (err) { console.warn('bvh', o.name, err); }
    if (performance.now() - t0 > 12) {
      await breathe();
      if (!current(job)) { disposeTree(root); return null; }
      t0 = performance.now();
    }
  }
  // Textures uploaded and shaders compiled now, so the swap's first frame has nothing to do.
  for (const t of texturesOf(root)) {
    try { renderer.initTexture(t); } catch (err) { console.warn('texture upload', err); }
    await breathe();
    if (!current(job)) { disposeTree(root); return null; }
  }
  try {
    // Without the parallel-compile extension compileAsync gains nothing (and its probe warns).
    if (renderer.extensions.has('KHR_parallel_shader_compile')) await renderer.compileAsync(root, camera, scene);
    else renderer.compile(root, camera, scene);
  } catch (err) { console.warn('compile', err); }
  if (!current(job)) { disposeTree(root); return null; }
  return { v: f.v, manifest: f.manifest, root, bodies: recs, byName };
}

function texturesOf(root) {
  const out = new Set();
  root.traverse((o) => {
    for (const m of [].concat(o.material || [])) for (const v of Object.values(m)) if (v?.isTexture) out.add(v);
  });
  return out;
}

/** Free a room's GPU and CPU copies: geometry (and its bounds tree), materials, textures. */
function disposeTree(root) {
  const textures = texturesOf(root), materials = new Set();
  root.traverse((o) => {
    if (o.geometry) { o.geometry.disposeBoundsTree?.(); o.geometry.dispose(); }
    for (const m of [].concat(o.material || [])) materials.add(m);
  });
  for (const m of materials) m.dispose();
  for (const t of textures) { t.dispose(); t.source?.data?.close?.(); }   // ImageBitmaps hold memory until closed
}

/** Where the loads should be heading, and start / abandon one to match. Cheap and idempotent:
 *  called on every welcome, scene message and state frame, and when a load ends. */
function reconsider() {
  const now = performance.now();
  if (ver.next !== undefined && (ver.next === ver.room || ver.gone.has(ver.next) || now - ver.nextAt > NEXT_TIMEOUT_MS)) ver.next = undefined;
  let target = ver.next !== undefined ? ver.next : ver.room;
  if (target !== undefined && ver.gone.has(target)) target = undefined;
  if (!shown?.root && target === undefined) target = null;   // nothing on screen: take what is live
  if (shown?.root && target === shown.v) target = undefined;
  if (target === undefined) {                    // what we show is where the room is (or is going)
    if (ver.job && ver.job.stage !== 'swapping') abandonJob();
    return;
  }
  // On its way. A first load (v null: "whatever is live") is most likely the very version the
  // welcome then names, so it counts as on its way to any target until its manifest says which
  // version it is (`fetchVersion`) — a welcome mid-download must not restart the download; and
  // with nothing on screen and nothing known yet (target null), any load will do.
  if (ver.job && (ver.job.v === target || ver.job.v === null || target === null)) return;
  abandonJob();
  if (target === ver.failedV && now < ver.retryAt) return;   // backing off; a timer comes back
  startJob(target);
}

function abandonJob() {
  const job = ver.job;
  if (!job) return;
  ver.job = null;
  job.ctrl.abort();
  if (job.prep) disposeTree(job.prep.root);
}

function startJob(v) {
  const job = { v, ctrl: new AbortController(), stage: 'fetch', progress: 0, fetched: null, prep: null };
  ver.job = job;
  runJob(job);
}

async function runJob(job) {
  try {
    job.fetched = await fetchVersion(job);
    if (!current(job)) return;
    if (!(LOW_MEMORY && shown?.root)) {
      job.prep = await loadVersion(job, job.fetched);
      if (!job.prep || !current(job)) return;
      job.fetched = null;
    }
    job.stage = 'ready';
    ver.fails = 0; ver.failedV = undefined;
    maybeInstall();
  } catch (err) {
    if (!current(job)) return;                     // abandoned: the abort is expected
    ver.job = null;
    if (job.prep) disposeTree(job.prep.root);
    loadFailed(job, err);
  }
}

function loadFailed(job, err) {
  if (err?.gone && job.v != null) {
    // The server does not have that version (pruned, or it restarted with another scene): wait
    // for it to name the next one rather than retrying this one.
    console.warn('room version not on the server:', job.v);
    ver.gone.add(job.v);
    reconsider();
    return;
  }
  console.warn('room load failed', err);
  ver.fails++; ver.failedV = job.v;
  const wait = Math.min(30000, 1000 * 2 ** (ver.fails - 1));
  ver.retryAt = performance.now() + wait;
  if (!shown?.root) {
    setLoad(0, 'Could not load the room: ' + (err?.message || err));
    if (ver.fails === 1) toast('Could not load the room. Is the server still running?', 'error');
  }
  // In the room and still failing: a fresh page is the last resort, never a loop — a page that
  // cannot load at all stops at its join screen, not in a room.
  if (ver.fails >= MAX_LOAD_FAILS && welcomed) { location.reload(); return; }
  setTimeout(reconsider, wait + 20);
}

/** A prepared (or, on a phone, fetched) version goes in when nothing is on screen yet, or at the
 *  first frame the room sends for it. */
function maybeInstall() {
  const job = ver.job;
  if (!job || job.stage !== 'ready') return;
  const v = (job.prep || job.fetched).v;
  if (shown?.root && v !== ver.room) return;
  if (job.prep) { ver.job = null; installVersion(job.prep); reconsider(); return; }
  swapLowMemory(job);
}

/** Phones: free the old room first, then parse the new one. The room is empty for the parse
 *  (people and grab lines stay); the old version's names stay for remapping the grab. */
async function swapLowMemory(job) {
  job.stage = 'swapping';
  setHover(null);
  if (shown?.root) { world.remove(shown.root); disposeTree(shown.root); shown.root = null; }
  sceneRoot = null; sceneReady = false;
  try {
    const prep = await loadVersion(job, job.fetched);
    if (!prep || !current(job)) return;
    ver.job = null;
    installVersion(prep);
    reconsider();
  } catch (err) {
    if (!current(job)) return;
    ver.job = null;
    loadFailed(job, err);
  }
}

/** The swap: synchronous, so no frame ever sees half of it. The person stays where they stand
 *  (only clamped into the new walls); what they hold stays held if the new version has a movable
 *  body of that name; the newest frame for this version is applied at once, snapping. */
function installVersion(next) {
  const old = shown, first = !old;
  setHover(null);
  if (grab) {
    const name = old?.bodies.get(grab.body)?.name;
    const id = name ? next.byName.get(name) : undefined;
    if (id === undefined) endGrab(true);          // gone, or no longer movable: let go
    else grab.body = id;                          // the server carries the grab by name too
  }
  if (old?.root) { world.remove(old.root); disposeTree(old.root); }
  world.add(next.root);
  shown = next; bodies = next.bodies; sceneRoot = next.root;
  sceneReady = true;
  applyManifest(next.manifest, first);
  if (grab) setHighlight(grab.body, true);
  const f = ver.frame;
  ver.frame = null;
  if (f && versionOf(f) === next.v) applyBodies(f);
  setLoad(1, `${bodies.size} object${bodies.size === 1 ? '' : 's'} you can move`);
  $('loadProgress').classList.add('done');
  if (!first) toast('The room was updated');
}

/** Label, stage, walk limits and eye height from a version's manifest. The spawn point only for
 *  the first room: later versions of the same room keep everyone where they are. */
function applyManifest(man, first) {
  const label = String(man.label || 'Room');
  $('sceneLabel').textContent = label;
  $('hudLabel').textContent = label;
  document.title = label + ' · Room call';
  const tag = STAGE_TAGS[man.stage] || '';
  $('hudStage').textContent = tag;
  $('hudStage').hidden = !tag;
  const bmin = man.bounds?.min, bmax = man.bounds?.max;
  if (finite(bmin) && finite(bmax)) {
    const m = 0.25;
    const cx = (bmin[0] + bmax[0]) / 2, cy = (bmin[1] + bmax[1]) / 2;
    const hx = Math.max(0, (bmax[0] - bmin[0]) / 2 - m), hy = Math.max(0, (bmax[1] - bmin[1]) / 2 - m);
    bounds.minX = cx - hx; bounds.maxX = cx + hx;
    bounds.minZ = -(cy + hy); bounds.maxZ = -(cy - hy);   // three z = -mj y
  }
  eyeY = (Number.isFinite(man.floor_z) ? man.floor_z : 0) + EYE;
  if (first) placeSpawn(man); else clampWalker();
}

/** The spawn point from the manifest (MuJoCo frame) into three space. */
function placeSpawn(man) {
  const bmin = man.bounds?.min, bmax = man.bounds?.max;
  const sp = finite(man.spawn) ? man.spawn : [0, 0, eyeY];
  // A small random offset so people who join together do not stand inside one another.
  const a = Math.random() * Math.PI * 2, r = Math.random() * 0.45;
  mjToThree(sp[0] + Math.cos(a) * r, sp[1] + Math.sin(a) * r, eyeY, walker.pos);
  clampWalker();
  // Face the middle of the room if the spawn is off-centre; otherwise look along +X.
  if (finite(bmin) && finite(bmax)) {
    const dx = (bmin[0] + bmax[0]) / 2 - sp[0], dy = (bmin[1] + bmax[1]) / 2 - sp[1];
    if (Math.hypot(dx, dy) > 0.5) walker.yaw = Math.atan2(dy, dx);
  }
}

function setHighlight(id, on) {
  const b = bodies.get(id);
  if (!b) return;
  for (const m of b.mats) {
    if (on) m.emissive.setRGB(0.16, 0.24, 0.38); else m.emissive.copy(m.userData.emissive0);
  }
}

/* ------------------------------------------------------------------ walking (first person) */
// yaw is about MuJoCo +Z with 0 looking along +X (the wire convention); pitch > 0 looks up.
const walker = { pos: new THREE.Vector3(0, EYE, 0), vel: new THREE.Vector3(), yaw: 0, pitch: 0 };
const keys = new Set();
const joy = { x: 0, y: 0 };                        // on-screen stick, -1..1 (y up = forward)
const twoFinger = { x: 0, y: 0 };

function clampWalker() {
  walker.pos.x = clamp(walker.pos.x, bounds.minX, bounds.maxX);
  walker.pos.z = clamp(walker.pos.z, bounds.minZ, bounds.maxZ);
  walker.pos.y = eyeY;
}

function stepWalker(dt) {
  const k = (c) => keys.has(c);
  let fwd = (k('KeyW') || k('ArrowUp') ? 1 : 0) - (k('KeyS') || k('ArrowDown') ? 1 : 0);
  let side = (k('KeyD') || k('ArrowRight') ? 1 : 0) - (k('KeyA') || k('ArrowLeft') ? 1 : 0);
  const turn = (k('KeyQ') ? 1 : 0) - (k('KeyE') ? 1 : 0);
  fwd += joy.y + twoFinger.y; side += joy.x + twoFinger.x;
  const len = Math.hypot(fwd, side);
  if (len > 1) { fwd /= len; side /= len; }
  walker.yaw = wrapAngle(walker.yaw + turn * TURN * dt);
  const speed = k('ShiftLeft') || k('ShiftRight') ? HURRY : WALK;
  // MuJoCo forward (cos, sin, 0) -> three (cos, 0, -sin); right (sin, -cos, 0) -> three (sin, 0, cos).
  const c = Math.cos(walker.yaw), s = Math.sin(walker.yaw);
  const want = new THREE.Vector3(c * fwd + s * side, 0, -s * fwd + c * side).multiplyScalar(speed);
  walker.vel.lerp(want, smooth(10, dt));
  walker.pos.addScaledVector(walker.vel, dt);
  clampWalker();
  camera.position.copy(walker.pos);
  camera.rotation.set(walker.pitch, walker.yaw - Math.PI / 2, 0);
}

addEventListener('keydown', (e) => {
  if (e.target instanceof HTMLInputElement || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.code === 'Escape') { closePanels(); return; }
  if (!joined) return;
  keys.add(e.code);
  if (e.code.startsWith('Arrow')) e.preventDefault();
});
addEventListener('keyup', (e) => keys.delete(e.code));
// Alt-Tab or a tab switch swallows the pointerup, so let go here rather than leave it floating.
addEventListener('blur', () => { keys.clear(); endGrab(); });
document.addEventListener('visibilitychange', () => { if (document.hidden) { keys.clear(); endGrab(); } });

/* ------------------------------------------------------------------ pointer: look and grab */
const raycaster = new THREE.Raycaster();
raycaster.far = 40;
raycaster.firstHitOnly = true;                     // three-mesh-bvh: nearest hit per mesh only
const ndc = new THREE.Vector2();
const pointers = new Map();                        // pointerId -> {x, y, sx, sy, type}
let gesture = 'none';                              // none | look | pending | grab | walk2
let holdTimer = 0;
let grab = null;                                   // {body, dist, pointerId, x, y}
let hoverBody = null, hoverAt = null, hoverPickedAt = 0;
let lastMoveSent = 0, lastMoveTarget = null;

function setNdc(x, y) {
  const r = canvas.getBoundingClientRect();
  ndc.set(((x - r.left) / r.width) * 2 - 1, -((y - r.top) / r.height) * 2 + 1);
  raycaster.setFromCamera(ndc, camera);
}

/** Whether the frames coming in are for the version on screen. While they are not (the room has
 *  moved on and its new version is still loading here), body ids on screen mean nothing to the
 *  server, so nothing new can be grabbed. */
const inStep = () => ver.room === shown?.v;

/** The movable body under the screen point, or null. Static geometry in front blocks it, so you
 *  cannot grab (or see the grab cursor on) something through a wall. Cheap enough for hover too
 *  thanks to the BVH. */
function pick(x, y) {
  if (!sceneReady || !inStep()) return null;
  setNdc(x, y);
  const h = raycaster.intersectObject(sceneRoot, true).find((i) => i.object.isMesh && i.object.visible);
  if (!h || h.object.userData.bodyId === undefined) return null;
  return { body: h.object.userData.bodyId, point: h.point.clone(), distance: h.distance };
}

function setHover(id) {
  if (id === hoverBody) return;
  if (hoverBody !== null && hoverBody !== grab?.body) setHighlight(hoverBody, false);
  hoverBody = id;
  if (id !== null) setHighlight(id, true);
  canvas.classList.toggle('can-grab', id !== null && !grab);
}

function startGrab(hit, pointerId, x, y) {
  grab = { body: hit.body, dist: clamp(hit.distance, HOLD_MIN, HOLD_MAX), pointerId, x, y, since: performance.now() };
  setHighlight(hit.body, true);
  send({ t: 'grab', v: shown?.v, body: hit.body, point: threeToMj(hit.point) });   // the server ignores another version's ids
  lastMoveTarget = null;
  canvas.classList.add('grabbing');
  canvas.classList.remove('can-grab');
  if (navigator.vibrate && IS_TOUCH) navigator.vibrate(12);
}

/** Let go. The release is sent unless the socket is already gone (the server drops a leaving
 *  peer's grabs itself); it follows one last move to where the pointer is now, so a throw keeps
 *  the motion of its final frames instead of the last throttled target. */
function endGrab(sendRelease = true) {
  if (!grab) return;
  if (sendRelease) { send({ t: 'move', target: grabTarget() }); send({ t: 'release' }); }
  if (grab.body !== hoverBody) setHighlight(grab.body, false);
  grab = null;
  canvas.classList.remove('grabbing');
}

/** Where the held point should be: on the pointer ray, at the held distance (MuJoCo frame). */
function grabTarget() {
  setNdc(grab.x, grab.y);
  return threeToMj(camera.position.clone().addScaledVector(raycaster.ray.direction, grab.dist));
}

/** Each frame while holding: the target rides the pointer ray at the held distance, so walking
 *  while holding carries the object along. */
function updateGrabTarget(now) {
  if (!grab) return;
  const target = grabTarget();
  if (now - lastMoveSent < 1000 / MOVE_HZ) return;
  if (lastMoveTarget && Math.hypot(target[0] - lastMoveTarget[0], target[1] - lastMoveTarget[1], target[2] - lastMoveTarget[2]) < 0.001
      && now - lastMoveSent < 250) return;
  send({ t: 'move', target });
  lastMoveSent = now; lastMoveTarget = target;
}

function look(dx, dy, type) {
  if (type === 'mouse') { walker.yaw -= dx * LOOK_MOUSE; walker.pitch -= dy * LOOK_MOUSE; }
  else { walker.yaw += dx * LOOK_TOUCH; walker.pitch += dy * LOOK_TOUCH; }   // touch drags the world
  walker.yaw = wrapAngle(walker.yaw);
  walker.pitch = clamp(walker.pitch, -PITCH_LIMIT, PITCH_LIMIT);
}

function centroid() {
  let x = 0, y = 0;
  for (const p of pointers.values()) { x += p.x; y += p.y; }
  return { x: x / pointers.size, y: y / pointers.size };
}

canvas.addEventListener('pointerdown', (e) => {
  if (!joined) return;
  closePanels();
  canvas.focus({ preventScroll: true });
  try { canvas.setPointerCapture(e.pointerId); } catch { /* already released */ }
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY, sx: e.clientX, sy: e.clientY, type: e.pointerType });

  if (e.pointerType === 'mouse') {
    if (e.button !== 0) { gesture = 'look'; return; }
    const hit = pick(e.clientX, e.clientY);
    if (hit) { gesture = 'grab'; startGrab(hit, e.pointerId, e.clientX, e.clientY); }
    else { gesture = 'look'; canvas.classList.add('looking'); }
    return;
  }
  // touch / pen
  if (pointers.size === 1) {
    gesture = 'pending';
    clearTimeout(holdTimer);
    const { clientX: x, clientY: y, pointerId: id } = e;
    holdTimer = setTimeout(() => {
      if (gesture !== 'pending') return;
      const hit = pick(x, y);
      if (hit) { gesture = 'grab'; startGrab(hit, id, x, y); } else gesture = 'look';
    }, HOLD_MS);
  } else if (pointers.size === 2 && gesture !== 'grab') {
    clearTimeout(holdTimer);
    gesture = 'walk2';
    const c = centroid();
    twoFinger.cx = c.x; twoFinger.cy = c.y;
  }
});

canvas.addEventListener('pointermove', (e) => {
  const p = pointers.get(e.pointerId);
  if (!p) {
    if (e.pointerType === 'mouse' && joined && !grab) { hoverAt = { x: e.clientX, y: e.clientY }; }
    return;
  }
  const dx = e.clientX - p.x, dy = e.clientY - p.y;
  p.x = e.clientX; p.y = e.clientY;
  if (gesture === 'pending' && Math.hypot(p.x - p.sx, p.y - p.sy) > TAP_SLOP) { clearTimeout(holdTimer); gesture = 'look'; }
  if (gesture === 'look') look(dx, dy, p.type);
  else if (gesture === 'grab') {
    if (grab && e.pointerId === grab.pointerId) { grab.x = e.clientX; grab.y = e.clientY; }
    else if (grab) grab.dist = clamp(grab.dist * Math.exp(-dy * 0.01), HOLD_MIN, HOLD_MAX);  // second finger: up pushes away
  } else if (gesture === 'walk2' && pointers.size >= 2) {
    const c = centroid();
    twoFinger.x = clamp((c.x - twoFinger.cx) / 70, -1, 1);
    twoFinger.y = clamp(-(c.y - twoFinger.cy) / 70, -1, 1);
  }
});

function pointerEnd(e) {
  if (!pointers.delete(e.pointerId)) return;
  if (grab && e.pointerId === grab.pointerId) endGrab();
  if (gesture === 'walk2') { twoFinger.x = 0; twoFinger.y = 0; }
  if (pointers.size === 0) { clearTimeout(holdTimer); gesture = 'none'; canvas.classList.remove('looking'); }
  else if (gesture === 'walk2' || gesture === 'grab') gesture = 'none';   // wait for all fingers up
}
canvas.addEventListener('pointerup', pointerEnd);
canvas.addEventListener('pointercancel', pointerEnd);
canvas.addEventListener('lostpointercapture', pointerEnd);
canvas.addEventListener('pointerleave', () => { if (!grab) setHover(null); });
canvas.addEventListener('contextmenu', (e) => e.preventDefault());

canvas.addEventListener('wheel', (e) => {
  if (!grab) return;
  e.preventDefault();
  const dy = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY;
  grab.dist = clamp(grab.dist * Math.exp(-dy * 0.0015), HOLD_MIN, HOLD_MAX);   // wheel up pushes away
}, { passive: false });

/* ------------------------------------------------------------------ on-screen joystick (touch) */
(() => {
  const pad = $('joystick'), knob = $('joyKnob');
  let id = null;
  const setFrom = (e) => {
    const r = pad.getBoundingClientRect(), R = r.width / 2;
    let x = (e.clientX - r.left - R) / R, y = (e.clientY - r.top - R) / R;
    const l = Math.hypot(x, y);
    if (l > 1) { x /= l; y /= l; }
    joy.x = x; joy.y = -y;
    knob.style.transform = `translate(${x * R * 0.6}px, ${y * R * 0.6}px)`;
  };
  const stop = () => { id = null; joy.x = 0; joy.y = 0; knob.style.transform = ''; };
  pad.addEventListener('pointerdown', (e) => { id = e.pointerId; pad.setPointerCapture(id); setFrom(e); e.stopPropagation(); });
  pad.addEventListener('pointermove', (e) => { if (e.pointerId === id) setFrom(e); });
  pad.addEventListener('pointerup', stop);
  pad.addEventListener('pointercancel', stop);
})();

/* ------------------------------------------------------------------ grabs drawn as lines */
// Each grab is a line from the grabber's right hand (another person's; not your own) to where
// they are pulling, then on to the point they hold, with a dot there. The hand end follows their
// avatar every frame (`stepGrabLines`); the rest comes with each state frame.
const grabLines = new Map();                       // peer id -> {line, dot}
const dotGeo = new THREE.SphereGeometry(0.025, 12, 8);
const handAt = new THREE.Vector3();

function drawGrabs(rows) {
  const seen = new Set();
  for (const g of rows || []) {
    const [pid, , ax, ay, az, tx, ty, tz] = g;
    if (!finite([ax, ay, az, tx, ty, tz])) continue;
    seen.add(pid);
    let rec = grabLines.get(pid);
    const color = pid === me.id ? me.color : (peers.get(pid)?.color || '#888888');
    if (!rec) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(9), 3));
      const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color, depthTest: false, transparent: true, opacity: 0.9 }));
      const dot = new THREE.Mesh(dotGeo, new THREE.MeshBasicMaterial({ color, depthTest: false }));
      line.renderOrder = dot.renderOrder = 10;
      line.frustumCulled = false;
      world.add(line, dot);                       // inside `world`: MuJoCo coordinates as-is
      rec = { line, dot };
      grabLines.set(pid, rec);
    }
    const pos = rec.line.geometry.attributes.position;
    pos.setXYZ(1, tx, ty, tz); pos.setXYZ(2, ax, ay, az);
    rec.dot.position.set(ax, ay, az);
    placeHand(pid, pos);
  }
  for (const [pid, rec] of grabLines) {
    if (seen.has(pid)) continue;
    world.remove(rec.line, rec.dot);
    rec.line.geometry.dispose(); rec.line.material.dispose(); rec.dot.material.dispose();
    grabLines.delete(pid);
  }
}

/** The line's first point: that person's right hand, or (yours, or no avatar yet) the target. */
function placeHand(pid, pos) {
  const h = pid !== me.id && handOf(pid, handAt);
  if (h) pos.setXYZ(0, h.x, -h.z, h.y);            // three -> MuJoCo, as the `world` group expects
  else pos.setXYZ(0, pos.getX(1), pos.getY(1), pos.getZ(1));
  pos.needsUpdate = true;
}

function stepGrabLines() {
  for (const [pid, rec] of grabLines) placeHand(pid, rec.line.geometry.attributes.position);
}

/** Roughly where someone's right hand is, in three space. A cut-out always faces you, so their
 *  right is your left of it; a capsule's right follows its own yaw. */
function handOf(pid, out) {
  const av = peers.get(pid)?.avatar;
  if (!av?.has || !av.root.visible) return null;
  const p = av.root.position;
  let rx, rz;
  if (av.cut.visible) {
    const dx = camera.position.x - p.x, dz = camera.position.z - p.z, l = Math.hypot(dx, dz) || 1;
    rx = -dz / l; rz = dx / l;
  } else { rx = Math.sin(av.yaw); rz = Math.cos(av.yaw); }   // MuJoCo right (sin, -cos, 0) in three
  return out.set(p.x + rx * HAND_SIDE, p.y - HAND_DROP, p.z + rz * HAND_SIDE);
}

/* ------------------------------------------------------------------ avatars */
function labelSprite(name, color) {
  const c = document.createElement('canvas');
  const ctx = c.getContext('2d');
  const font = '600 34px system-ui, -apple-system, "Segoe UI", sans-serif';
  ctx.font = font;
  const w = Math.min(512, Math.ceil(ctx.measureText(name).width) + 72), h = 56;
  c.width = w; c.height = h;
  ctx.font = font;
  ctx.fillStyle = 'rgba(22,25,29,0.78)';
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(0, 0, w, h, h / 2); else ctx.rect(0, 0, w, h);
  ctx.fill();
  ctx.fillStyle = color; ctx.beginPath(); ctx.arc(28, h / 2, 10, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = '#ffffff'; ctx.textBaseline = 'middle';
  ctx.fillText(name, 48, h / 2 + 1, w - 60);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, toneMapped: false }));
  s.scale.set(0.11 * w / h, 0.11, 1);
  return s;
}

function makeAvatar(peer) {
  const color = new THREE.Color(peer.color);
  const skin = new THREE.MeshStandardMaterial({ color, roughness: 0.55, metalness: 0 });
  const root = new THREE.Group();
  const torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.19, 0.9, 6, 18), skin);
  torso.position.y = -0.9;                        // from ~1.55 m below the eye up to the neck
  const head = new THREE.Group();
  head.add(new THREE.Mesh(new THREE.SphereGeometry(0.165, 28, 18), skin));
  const visor = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.05, 0.04),
    new THREE.MeshStandardMaterial({ color: 0x1d232a, roughness: 0.3 }));
  visor.position.set(0, 0.02, -0.155);            // shows which way they face (the head turns, the video does not)
  head.add(visor);
  // The webcam plane hangs on its own pivot that turns about the vertical to face the viewer, so
  // a face is readable from any side; it sits just outside the head and torso radii.
  const face = new THREE.Group();
  const screen = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ color: 0xffffff, toneMapped: false }));
  screen.position.set(0, 0, 0.2);                 // +Z: the side lookAt turns toward the viewer
  screen.visible = false;
  face.add(screen);
  // Their hologram: a life-size cut-out on a pivot of its own that, like the face, turns about the
  // vertical toward whoever looks. Shown instead of the capsule while they send one.
  const cut = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), holoMaterial());
  cut.visible = false;
  const stand = new THREE.Group();
  stand.add(cut);
  const label = labelSprite(peer.name, peer.color);
  label.position.y = LABEL_Y;
  root.add(torso, head, face, stand, label);
  root.visible = false;                           // until their first pose arrives
  scene.add(root);
  return { root, torso, head, face, screen, cut, stand, label, skin, sw: FACE_H * 4 / 3, sh: FACE_H,
           aspect: 4 / 3, fr: { ...HOLO_FRAMING },
           tp: new THREE.Vector3(), yaw: 0, pitch: 0, tyaw: 0, tpitch: 0, has: false };
}

// The receiving end of the hologram. One video carries both halves of each frame: colour on the
// left, the person mask as grey on the right (see "hologram"). The mask went through a video
// codec, so it is read as luma (what codecs keep best) and passed through a ramp that drops the
// blocky noise near 0 and 1 while keeping a soft edge. Both lookups stay a texel and a half clear
// of the seam, so filtering never mixes the halves. The bottom of the frame cuts the body, so it
// fades out, and a hair of fade at the other edges hides a hard frame line. The texture is
// sampled raw (video textures are uploaded without sRGB decoding), so colour comes out exactly as
// the face plane shows it. Output is premultiplied; near-transparent fragments are discarded so
// the depth written is the person's outline and later transparent things sort against it.
const HOLO_VERT = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;
const HOLO_FRAG = /* glsl */ `
uniform sampler2D map;
uniform float inset;
uniform float fade;
varying vec2 vUv;
void main() {
  float x = clamp(vUv.x, inset, 1.0 - inset) * 0.5;
  vec3 color = texture2D(map, vec2(x, vUv.y)).rgb;
  float m = dot(texture2D(map, vec2(x + 0.5, vUv.y)).rgb, vec3(0.299, 0.587, 0.114));
  float a = smoothstep(0.3, 0.7, m);
  a *= smoothstep(0.0, fade, vUv.y) * smoothstep(0.0, 0.02, 1.0 - vUv.y)
     * smoothstep(0.0, 0.03, vUv.x) * smoothstep(0.0, 0.03, 1.0 - vUv.x);
  if (a < 0.05) discard;
  gl_FragColor = vec4(color * a, a);
}`;

function holoMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: { map: { value: null }, inset: { value: 3 / (2 * HOLO_W) }, fade: { value: 0.22 } },
    vertexShader: HOLO_VERT, fragmentShader: HOLO_FRAG,
    transparent: true, premultipliedAlpha: true, depthWrite: true,
  });
}

function disposeAvatar(av) {
  scene.remove(av.root);
  av.root.traverse((o) => {
    o.geometry?.dispose?.();
    for (const m of [].concat(o.material || [])) { m.map?.dispose?.(); m.dispose?.(); }
  });
}

/** What their video is (a webcam for the face plane, a packed cut-out for the hologram, or
 *  nothing yet) decides how they are drawn. Runs on every change of their video's size, which is
 *  also when a sender switches between the two. */
function updateAvatarScreen(peer) {
  const av = peer.avatar, v = peer.video;
  const show = !!(peer.media.camera && v && v.readyState >= 2 && v.videoWidth > 0);
  const packed = show && v.videoWidth / v.videoHeight >= HOLO_PACKED;
  if (show && !peer.tex) {
    peer.tex = new THREE.VideoTexture(v);
    peer.tex.colorSpace = THREE.SRGBColorSpace;
    av.screen.material.map = peer.tex;
    av.screen.material.needsUpdate = true;
    av.cut.material.uniforms.map.value = peer.tex;
  }
  if (packed) {
    av.aspect = v.videoWidth / 2 / v.videoHeight;
    av.cut.material.uniforms.inset.value = 3 / v.videoWidth;   // 1.5 texels, in one half's uv
  } else if (show) {
    const aspect = clamp(v.videoWidth / v.videoHeight, 0.5, 2);
    av.sw = Math.min(FACE_H * 4 / 3, FACE_H * aspect);   // FACE_H tall, no wider than 4:3
    av.sh = av.sw / aspect;
  }
  const was = av.cut.visible;
  av.screen.visible = show && !packed;
  av.cut.visible = packed;
  av.torso.visible = av.head.visible = !packed;
  if (was !== packed) renderPeople();
}

/** Life-size: the head HEAD_M tall (from where the sender says the head is in its frame), the
 *  eyes at the avatar's eye height, the head over the avatar's position, facing you. */
function placeCut(peer, dt) {
  const av = peer.avatar, f = peer.framing || HOLO_FRAMING, fr = av.fr, a = smooth(3, dt);
  fr.top += (f.top - fr.top) * a; fr.head += (f.head - fr.head) * a; fr.cx += (f.cx - fr.cx) * a;
  const h = clamp(HEAD_M / fr.head, 0.45, 1.7), w = h * av.aspect;
  const eyes = fr.top + EYES_IN_HEAD * fr.head;   // the eyes' row, as a fraction from the image top
  av.cut.scale.set(w, h, 1);
  av.cut.position.set(clamp((0.5 - fr.cx) * w, -0.25 * w, 0.25 * w), h * (eyes - 0.5), 0);
  av.label.position.y = av.cut.position.y + h / 2 + 0.08;
  av.stand.lookAt(camera.position.x, av.root.position.y, camera.position.z);
}

function stepAvatars(dt) {
  const a = smooth(AVATAR_SMOOTH, dt);
  for (const peer of peers.values()) {
    const av = peer.avatar;
    if (!av.has) continue;
    av.root.position.lerp(av.tp, a);
    av.yaw += wrapAngle(av.tyaw - av.yaw) * a;
    av.pitch += (av.tpitch - av.pitch) * a;
    av.root.rotation.y = av.yaw - Math.PI / 2;
    av.head.rotation.x = clamp(av.pitch, -PITCH_LIMIT, PITCH_LIMIT) * 0.7;
    if (av.cut.visible) { placeCut(peer, dt); continue; }
    if (!av.screen.visible) { av.label.position.y = LABEL_Y; continue; }
    // Grow with distance (constant on-screen size from 1.5 m to ~4 m) so a face stays a face
    // across the room, and keep the name tag above it.
    const s = clamp(camera.position.distanceTo(av.root.position) / 1.5, 1, 2.5);
    av.screen.scale.set(av.sw * s, av.sh * s, 1);
    av.label.position.y = Math.max(LABEL_Y, av.sh * s / 2 + 0.1);
    av.face.lookAt(camera.position.x, av.root.position.y, camera.position.z);
  }
}

/* ------------------------------------------------------------------ state stream */
let lastStateAt = 0;

/** People and grab lines are in world coordinates and always drawn; body poses only from frames
 *  for the version on screen (anything else is kept for when that version goes in). */
function onState(m) {
  lastStateAt = performance.now();
  const fv = versionOf(m);
  if (fv !== ver.room) { ver.room = fv; if (fv === ver.next) ver.next = undefined; }
  if (shown?.root && fv === shown.v) { ver.frame = null; applyBodies(m); }
  else { ver.frame = m; reconsider(); maybeInstall(); }
  if (ver.next !== undefined) reconsider();       // times out a version the room never reaches
  drawGrabs(m.grabs);
  for (const [pid, v] of Object.entries(m.avatars || {})) {
    if (pid === me.id || !finite(v) || v.length < 5) continue;
    const peer = peers.get(pid);
    if (!peer) continue;
    const av = peer.avatar;
    mjToThree(v[0], v[1], v[2], av.tp);
    av.tyaw = v[3]; av.tpitch = v[4];
    if (!av.has) { av.root.position.copy(av.tp); av.yaw = av.tyaw; av.pitch = av.tpitch; av.has = true; av.root.visible = true; }
  }
}

function applyBodies(m) {
  for (const row of m.bodies || []) {
    if (!finite(row) || row.length < 8) continue;
    const b = bodies.get(row[0]);
    if (!b) continue;
    b.tp.set(row[1], row[2], row[3]);
    b.tq.set(row[5], row[6], row[7], row[4]).normalize();   // MuJoCo w,x,y,z -> three x,y,z,w
    if (!b.has) { b.node.position.copy(b.tp); b.node.quaternion.copy(b.tq); b.has = true; }
  }
  if (grab && Array.isArray(m.grabs) && !m.grabs.some((g) => g[0] === me.id) && performance.now() - grab.since > 2000) {
    // The server dropped it (reset, bad body, a swap to a version where it is not movable…) or
    // never got it (a grab for the version it had just left). Still send the release: if the
    // grab is only late, it must not be applied after we stopped steering it and float forever.
    endGrab(true);
  }
}

function stepBodies(dt) {
  const a = smooth(BODY_SMOOTH, dt);
  for (const b of bodies.values()) {
    if (!b.has) continue;
    b.node.position.lerp(b.tp, a);
    b.node.quaternion.slerp(b.tq, a);
  }
}

/* ------------------------------------------------------------------ peers and the people list */
function addPeer(p) {
  if (!p || typeof p.id !== 'string' || p.id === me.id) return null;
  let peer = peers.get(p.id);
  if (peer) return peer;
  const color = /^#[0-9a-fA-F]{6}$/.test(p.color) ? p.color : '#888888';
  peer = {
    id: p.id, name: String(p.name || 'Guest').slice(0, 40), color,
    media: { camera: !!p.media?.camera, mic: !!p.media?.mic },
    rtc: null, video: null, audio: null, tex: null,
    framing: null,                                 // where their head is in their cut-out (data channel)
  };
  peer.avatar = makeAvatar(peer);
  peers.set(p.id, peer);
  renderPeople();
  return peer;
}

function removePeer(id) {
  const peer = peers.get(id);
  if (!peer) return;
  closeRtc(peer);
  disposeAvatar(peer.avatar);
  peer.tex?.dispose();
  peer.video?.remove(); peer.audio?.remove();
  peers.delete(id);
  renderPeople();
}

function clearPeers() { for (const id of [...peers.keys()]) removePeer(id); drawGrabs([]); }

function renderPeople() {
  const list = $('peopleList');
  list.replaceChildren();
  const row = (name, color, media, you, rtcFailed = false, cutOut = false) => {
    const li = document.createElement('li');
    const dot = document.createElement('span'); dot.className = 'dot'; dot.style.background = color;
    const who = document.createElement('span'); who.className = 'who'; who.textContent = name + (you ? ' (you)' : '');
    const tags = document.createElement('span'); tags.className = 'tags';
    if (rtcFailed) {
      // Their cam/mic flags mean nothing here when no direct link could be made.
      const t = document.createElement('span'); t.className = 'tag bad'; t.textContent = "video couldn't connect";
      t.title = 'No direct link to this person (a strict firewall or NAT). You still share the room; only their camera and mic do not reach you.';
      tags.appendChild(t);
    } else for (const [k, label] of [['camera', 'cam'], ['mic', 'mic']]) {
      const t = document.createElement('span'); t.className = 'tag' + (media[k] ? ' on' : '');
      t.textContent = k === 'camera' && media[k] && cutOut ? 'hologram' : label + (media[k] ? ' on' : ' off');
      tags.appendChild(t);
    }
    li.append(dot, who, tags);
    list.appendChild(li);
  };
  row(me.name || 'You', me.color, me.media, true, false, holo.state === 'live');
  for (const p of peers.values()) row(p.name, p.color, p.media, false, !!p.rtc?.failed, p.avatar.cut.visible);
  $('peopleCount').textContent = String(peers.size + 1);
}

/* ------------------------------------------------------------------ local media */
const local = { stream: new MediaStream(), video: null, audio: null,
                want: { video: false, audio: false },  // the latest on/off asked for, per slot
                asking: { video: null, audio: null } }; // the getUserMedia call in flight, per slot

function mediaError(kind, err) {
  const what = kind === 'video' ? 'camera' : 'microphone';
  const name = err?.name || '';
  if (name === 'NotAllowedError' || name === 'SecurityError') return `The browser blocked the ${what}. You can still join without it.`;
  if (name === 'NotFoundError' || name === 'OverconstrainedError') return `No ${what} found.`;
  if (name === 'NotReadableError') return `The ${what} is busy — another app (maybe the call) is using it.`;
  return `Could not start the ${what}${err?.message ? ': ' + err.message : '.'}`;
}

/** Turn one local track on or off and push the change to every peer connection: a new track goes
 *  out with replaceTrack on an existing sender (no renegotiation) or addTrack on a fresh one.
 *  At most one getUserMedia per slot is in flight: clicks while the permission prompt is up only
 *  update `want`, and the call already waiting applies the latest wish when it returns — a track
 *  nobody wants any more is stopped at once, so the camera light goes out and the call app can
 *  still have the camera. */
async function setLocal(kind, on) {
  const slot = kind === 'video' ? 'video' : 'audio';
  if (on && !canMedia) { toast('Camera and microphone need https or localhost.', 'error'); return false; }
  local.want[slot] = on;
  if (local.asking[slot]) { syncMediaButtons(); await local.asking[slot].catch(() => {}); return !!local[slot]; }
  if (on && !local[slot]) {
    let track = null;
    try {
      local.asking[slot] = navigator.mediaDevices.getUserMedia(kind === 'video' ? { video: VIDEO } : { audio: AUDIO });
      syncMediaButtons();
      const s = await local.asking[slot];
      track = s.getTracks()[0] || null;
    } catch (err) {
      toast(mediaError(kind, err), 'error');
      local.want[slot] = false;
    } finally { local.asking[slot] = null; }
    if (track && (!local.want[slot] || local[slot])) { track.stop(); track = null; }   // changed our mind meanwhile
    if (track) {
      local[slot] = track;
      local.stream.addTrack(track);
      track.addEventListener('ended', () => { if (local[slot] === track) setLocal(kind, false); });
      // The webcam itself goes out first; the hologram replaces it once it sees someone.
      for (const peer of peers.values()) attachTrack(peer, track);
    }
  } else if (!on && local[slot]) {
    const track = local[slot];
    local[slot] = null;
    local.stream.removeTrack(track);
    track.stop();
    for (const peer of peers.values()) peer.rtc?.senders[slot]?.replaceTrack(null).catch(() => {});
  }
  if (kind === 'video') me.media.camera = !!local.video; else me.media.mic = !!local.audio;
  if (kind === 'video') syncHologram();
  showSelfView();
  syncMediaButtons();
  send({ t: 'media', camera: me.media.camera, mic: me.media.mic });
  renderPeople();
  return !!local[slot];
}

/** What a media button shows: the wish while the browser is still asking, else the real state. */
const shownOn = (slot) => (local.asking[slot] ? local.want[slot] : !!local[slot]);

/** Your own corner view: your cut-out while the others get one, else your webcam. */
function showSelfView() {
  const v = $('selfView'), cut = $('selfHolo');
  const live = !!local.video && holo.state === 'live';
  cut.hidden = !live;
  if (local.video && !live) { if (v.srcObject !== local.stream) v.srcObject = local.stream; v.hidden = false; v.play().catch(() => {}); }
  else { v.hidden = true; }
}

function syncMediaButtons() {
  $('camBtn').setAttribute('aria-pressed', String(shownOn('video')));
  $('micBtn').setAttribute('aria-pressed', String(shownOn('audio')));
  $('camBtn').disabled = !canMedia;
  $('micBtn').disabled = !canMedia || !!zoom;
  $('micBtn').title = zoom ? 'Zoom carries your voice' : 'Send your microphone';
  syncHoloButton();
}

/* ------------------------------------------------------------------ hologram: you, cut out of your webcam */
// With the camera on, each webcam frame is segmented here, on this device, by MediaPipe's selfie
// segmenter, and the person mask travels INSIDE the video: every frame sent is the colour image
// on the left and the soft mask as grey on the right, drawn into one canvas whose captured track
// replaces the webcam on every peer connection (replaceTrack, no renegotiation). WebRTC codecs
// carry no alpha channel, but grey pixels survive them fine, so each receiver can cut the person
// out (`holoMaterial`) and stand them life-size in the room. The packed frame's 8:3 shape is how a
// receiver tells it from a webcam, with nothing added to the protocol; the data channel only adds
// where the head is in the frame (`sendFraming`), so the cut-out can be sized life-size.
//
// Whenever the cut-out cannot be trusted, the plain webcam goes out instead, and everyone shows
// it on the avatar's face as before: while the runtime loads, if it cannot load (CDN, CSP, no
// WebAssembly or WebGL), on a device too slow for it, while nobody is in frame, while the tab is
// in the background (it gets no video frame callbacks there), and when the Hologram button is off.
const holo = {
  want: store.get('hologram', true) !== false,
  state: 'off',          // off | loading | ready (running; nobody in frame yet) | live (sending it) | failed
  why: '',               // why it failed, for the button's title
  mp: null,              // {vision, files}: the loaded runtime
  seg: null, segLoad: null, loadFailed: false, delegate: '', triedCpu: false,
  gen: 0,                // bumped by every start and stop, so a stale frame callback ends itself
  video: null, color: null, mask: null, alpha: null, packed: null, self: null, track: null,
  maskImg: null, alphaImg: null, sm: null, tmp: null, rowW: null, rowX: null,
  segAt: 0, ms: 0, segs: 0, lastTs: 0, lastTime: -1, slowSince: 0, maskAt: 0,
  slowMs: 150,           // a mask costing more than this for 8 s on end: too slow, give up
  seen: 0, goneAt: 0,    // masks in a row with someone in them; since when nobody has been
  framing: { ...HOLO_FRAMING }, sent: null, sentAt: 0,
};

const holoCapable = () => !!(HTMLCanvasElement.prototype.captureStream && window.WebAssembly && window.RTCRtpSender);

/** Start or stop to match the camera and the button. */
function syncHologram() {
  const want = holo.want && !!local.video && holoCapable();
  if (want && holo.state === 'off') startHologram();
  else if (!want && holo.state !== 'off' && holo.state !== 'failed') stopHologram('off');
  syncHoloButton();
}

function syncHoloButton() {
  const b = $('holoBtn');
  b.setAttribute('aria-pressed', String(holo.want && holo.state !== 'failed'));
  b.dataset.state = local.video ? holo.state : 'off';
  b.disabled = !canMedia || !holoCapable();
  b.title = !holo.want ? 'Hologram off: others see your plain webcam'
    : !local.video ? 'Hologram: once your camera is on, others see you cut out of your background'
    : { off: 'Hologram', loading: 'Hologram: getting the cut-out ready…',
        ready: 'Hologram: ready — waiting to see you in the camera',
        live: 'Hologram on: others see you cut out of your background',
        failed: `Hologram unavailable (${holo.why}): others see your plain webcam. Click to try again.` }[holo.state];
}

function holoCanvas(w, h, opts) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  return { c, ctx: c.getContext('2d', opts) };
}

function holoSetup() {
  if (holo.video) return;
  // GPU-side canvases throughout, even on a software GL: there a CPU-side colour canvas makes
  // the runtime's input cheaper (~20 ms) but every camera frame drawn into it dearer (~20 ms,
  // against 0.5), and frames are drawn more often than they are segmented.
  holo.color = holoCanvas(HOLO_W, HOLO_H, { alpha: false });
  holo.mask = holoCanvas(HOLO_W, HOLO_H, { alpha: false });
  holo.alpha = holoCanvas(HOLO_W, HOLO_H);
  holo.packed = holoCanvas(2 * HOLO_W, HOLO_H, { alpha: false });
  holo.maskImg = holo.mask.ctx.createImageData(HOLO_W, HOLO_H);
  holo.alphaImg = holo.alpha.ctx.createImageData(HOLO_W, HOLO_H);
  const n = HOLO_W * HOLO_H;
  holo.sm = new Float32Array(n); holo.tmp = new Float32Array(n);
  holo.rowW = new Uint16Array(HOLO_H); holo.rowX = new Float32Array(HOLO_H);
  const self = $('selfHolo');
  self.width = HOLO_W; self.height = HOLO_H;
  holo.self = self.getContext('2d');
  // The camera plays into a video of our own: frame callbacks need one that is in the page.
  holo.video = document.createElement('video');
  holo.video.muted = true; holo.video.playsInline = true; holo.video.autoplay = true;
  $('mediaSink').append(holo.video);
}

/** The runtime and the model, once per page. The GPU delegate first, the CPU one if that fails;
 *  the other way round on a software GL (no usable GPU), where WebAssembly SIMD is far faster. */
function loadSegmenter() {
  holo.segLoad ??= (async () => {
    const vision = await withTimeout(import(`${MP_BASE}/vision_bundle.mjs`), 30000);
    const files = await vision.FilesetResolver.forVisionTasks(`${MP_BASE}/wasm`);
    holo.mp = { vision, files };
    const order = softwareGL() ? ['CPU', 'GPU'] : ['GPU', 'CPU'];
    try { return { seg: await makeSegmenter(order[0]), delegate: order[0] }; }
    catch (err) { console.info(`hologram: no ${order[0]} segmenter here, trying ${order[1]}`, err); }
    return { seg: await makeSegmenter(order[1]), delegate: order[1] };
  })();
  return holo.segLoad;
}

/** Whether WebGL here is drawn in software (SwiftShader, llvmpipe: a VM, or a blocklisted GPU). */
let softGL;
function softwareGL() {
  softGL ??= softwareRenderer();
  return softGL;
}
function softwareRenderer() {
  try {
    const gl = renderer.getContext();
    let name = String(gl.getParameter(gl.RENDERER));
    if (/webkit/i.test(name)) {                    // masked (Chrome, Safari): ask for the real one
      const ext = gl.getExtension('WEBGL_debug_renderer_info');
      if (ext) name = String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL));
    }
    return /swiftshader|llvmpipe|softpipe|software|basic render/i.test(name);
  } catch { return false; }
}

// The runtime logs in glog's format ("W1021 16:23:58.43 ... gl_context.cc:1119] ..."), as it
// starts and on its first frames: delegate notes and GL hints, not ours to act on (real failures
// throw). While it runs, those lines go to debug; anything else passes through untouched.
const GLOG = /^[IWEF]\d{4} \d\d:\d\d:\d\d/;
function quietly(fn) {
  const saved = {};
  for (const k of ['log', 'info', 'warn', 'error']) {
    saved[k] = console[k];
    console[k] = (...a) => (typeof a[0] === 'string' && GLOG.test(a[0]) ? console.debug('[mediapipe]', ...a) : saved[k].apply(console, a));
  }
  // Its Emscripten output goes through the hook its loader reads (and clears) as well.
  const print = (t) => console.debug('[mediapipe]', t);
  window.Module = { print, printErr: print };
  const restore = () => { Object.assign(console, saved); delete window.Module; };
  let out;
  try { out = fn(); } catch (err) { restore(); throw err; }
  if (typeof out?.then === 'function') return out.finally(restore);
  restore();
  return out;
}

async function makeSegmenter(delegate) {
  const making = quietly(() => holo.mp.vision.ImageSegmenter.createFromOptions(holo.mp.files, {
    baseOptions: { modelAssetPath: SEG_MODEL, delegate },
    runningMode: 'VIDEO', outputCategoryMask: false, outputConfidenceMasks: true,
  }));
  try { return await withTimeout(making, 30000); }
  catch (err) { making.then((s) => s.close(), () => {}); throw err; }   // one that turns up late is freed
}

async function startHologram() {
  holoSetup();
  const gen = ++holo.gen;
  Object.assign(holo, { state: 'loading', seen: 0, goneAt: 0, segAt: 0, lastTime: -1, slowSince: 0, maskAt: 0 });
  holo.video.srcObject = new MediaStream([local.video]);
  holo.video.play().catch(() => {});
  holoSchedule(gen);
  syncHoloButton();
  try {
    const got = await loadSegmenter();
    if (gen !== holo.gen) return;
    if (!holo.seg) { holo.seg = got.seg; holo.delegate = got.delegate; }   // kept across restarts
    holo.state = 'ready';
    syncHoloButton();
  } catch (err) {
    if (gen !== holo.gen) return;
    holo.loadFailed = true;
    holoFail('the cut-out did not load', err);
  }
}

function stopHologram(state) {
  holo.gen++;
  holo.state = state;
  if (holo.video) holo.video.srcObject = null;
  if (holo.track) { holo.track.stop(); holo.track = null; }
  sendOut();
  showSelfView();
  syncHoloButton();
  renderPeople();
}

function holoFail(why, err) {
  // A load or runtime error is worth a warning; a device that is merely too slow is not.
  (err ? console.warn : console.info)('hologram off:', why, err || '');
  holo.why = why;
  stopHologram('failed');
  toast(`Hologram off: ${why}. Others see your plain webcam.`, 'error');
}

/** One callback per camera frame where the browser has them; else a poll that skips repeats. */
function holoSchedule(gen) {
  const v = holo.video;
  if (v.requestVideoFrameCallback) v.requestVideoFrameCallback(() => holoFrame(gen));
  else setTimeout(() => holoFrame(gen), 1000 / 30);
}

function holoFrame(gen) {
  if (gen !== holo.gen) return;
  holoSchedule(gen);
  const v = holo.video;
  if (v.readyState < 2 || !v.videoWidth || v.currentTime === holo.lastTime) return;
  holo.lastTime = v.currentTime;
  drawFit(holo.color.ctx, v);
  if (!holo.seg) return;
  // At most about a third of the main thread: a mask starts no sooner than three times what one
  // costs after the last one started. A fast device masks every camera frame; a slow one every
  // Nth, keeping the last mask. With nobody in frame, a look every HOLO_IDLE_MS is enough.
  const now = performance.now();
  const idle = holo.state === 'ready' && holo.seen === 0 ? HOLO_IDLE_MS : 0;
  const fresh = now - holo.segAt >= Math.max(3 * holo.ms, idle);
  if (fresh) { holo.segAt = now; segment(gen); }
  // A mask a frame or two old still fits the person; an older one leaves a ghost edge around
  // anyone moving. So once masks come further apart than that, only the frames that got their own
  // mask go out: a lower frame rate, but the colour and its mask always match.
  if (gen === holo.gen && holo.state === 'live' && (fresh || 3 * holo.ms < HOLO_STALE_MS)) compose();
}

/** The camera image into one 4:3 half: a wider image loses its sides, a narrower one (a phone
 *  held upright) keeps all of it between black bars that the mask leaves out. */
function drawFit(ctx, v) {
  const vw = v.videoWidth, vh = v.videoHeight, k = HOLO_W / HOLO_H;
  if (vw / vh >= k) {
    const sw = vh * k;
    ctx.drawImage(v, (vw - sw) / 2, 0, sw, vh, 0, 0, HOLO_W, HOLO_H);
  } else {
    const dw = HOLO_H * vw / vh;
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, HOLO_W, HOLO_H);
    ctx.drawImage(v, 0, 0, vw, vh, (HOLO_W - dw) / 2, 0, dw, HOLO_H);
  }
}

function segment(gen) {
  const t0 = performance.now();
  holo.lastTs = Math.max(t0, holo.lastTs + 1);    // VIDEO mode wants strictly increasing timestamps
  try {
    quietly(() => holo.seg.segmentForVideo(holo.color.c, holo.lastTs, (res) => {
      const m = res.confidenceMasks?.[0];          // the selfie segmenter's one channel: person
      if (m) takeMask(m.getAsFloat32Array(), m.width, m.height);
    }));
  } catch (err) { holoFail('the cut-out stopped working', err); return; }
  if (gen !== holo.gen) return;
  const dt = performance.now() - t0;
  holo.segs++;
  // The first one is mostly warm-up (shader compiles, first allocations): not what a mask costs.
  holo.ms = holo.segs === 1 ? 0 : holo.segs <= 3 ? dt : holo.ms * 0.9 + dt * 0.1;
  pace();
}

/** On a slow device: try the CPU delegate once if the GPU one is slow (a software GL loses to
 *  WebAssembly SIMD), and give up when a mask has cost over `slowMs` for 8 s on end (the pacing
 *  in `holoFrame` keeps the page usable meanwhile). */
function pace() {
  if (holo.segs < 5) return;
  if (holo.delegate === 'GPU' && holo.ms > 25 && !holo.triedCpu) { holo.triedCpu = true; tryCpu(); return; }
  const now = performance.now();
  if (holo.ms <= holo.slowMs) holo.slowSince = 0;
  else if (!holo.slowSince) holo.slowSince = now;
  else if (now - holo.slowSince > 8000 && (holo.triedCpu || holo.delegate === 'CPU')) holoFail('this device is too slow for it');
}

async function tryCpu() {
  let cpu = null;
  try {
    cpu = await makeSegmenter('CPU');
    if (!holo.seg || holo.state === 'off' || holo.state === 'failed') { cpu.close(); return; }
    const run = () => {
      holo.lastTs = Math.max(performance.now(), holo.lastTs + 1);
      quietly(() => cpu.segmentForVideo(holo.color.c, holo.lastTs, () => {}));
    };
    run();                                         // warm-up, not timed
    const t0 = performance.now();
    for (let i = 0; i < 3; i++) run();
    const ms = (performance.now() - t0) / 3;
    if (ms < holo.ms * 0.8) {
      console.info(`hologram: the CPU segmenter is faster here (${ms.toFixed(1)} ms vs ${holo.ms.toFixed(1)} ms)`);
      holo.seg.close();
      Object.assign(holo, { seg: cpu, delegate: 'CPU', ms, segs: 5 });
    } else cpu.close();
  } catch (err) { console.info('hologram: no CPU segmenter to compare', err); cpu?.close(); }
}

/** A new mask: smoothed over time (a little), feathered (a 3x3 box), stretched so that what the
 *  model is sure about is exactly black or white (flat areas cost the codec nothing), then
 *  written as grey for the packed frame and as alpha for the self view. Also counts, per row, how
 *  much of the row is person: for whether anyone is there, and where their head is.
 *  The smoothing has a time constant (MASK_TAU), not a fixed weight per mask: at a mask per
 *  camera frame it calms the flicker at the edges, while masks that come seldom (a slow device)
 *  are used as they are, so a moving person leaves no ghost of where they were. */
const MASK_TAU = 70;                               // ms
function takeMask(src, w, h) {
  const W = HOLO_W, H = HOLO_H, sm = holo.sm, tmp = holo.tmp, now = performance.now();
  const k = holo.maskAt ? 1 - Math.exp(-(now - holo.maskAt) / MASK_TAU) : 1;
  holo.maskAt = now;
  if (w === W && h === H) for (let i = 0; i < W * H; i++) sm[i] += (src[i] - sm[i]) * k;
  else {
    for (let y = 0; y < H; y++) {
      const sy = Math.min(h - 1, Math.floor(y * h / H)) * w;
      for (let x = 0; x < W; x++) { const i = y * W + x; sm[i] += (src[sy + Math.min(w - 1, Math.floor(x * w / W))] - sm[i]) * k; }
    }
  }
  for (let y = 0; y < H; y++) {
    const r = y * W;
    tmp[r] = (2 * sm[r] + sm[r + 1]) / 3;
    for (let x = 1; x < W - 1; x++) tmp[r + x] = (sm[r + x - 1] + sm[r + x] + sm[r + x + 1]) / 3;
    tmp[r + W - 1] = (sm[r + W - 2] + 2 * sm[r + W - 1]) / 3;
  }
  const md = holo.maskImg.data, ad = holo.alphaImg.data, rowW = holo.rowW, rowX = holo.rowX;
  let count = 0;
  for (let y = 0; y < H; y++) {
    const up = (y > 0 ? y - 1 : y) * W, mid = y * W, dn = (y < H - 1 ? y + 1 : y) * W;
    let n = 0, sx = 0;
    for (let x = 0; x < W; x++) {
      const v = (tmp[up + x] + tmp[mid + x] + tmp[dn + x]) / 3;
      const g = v <= 0.2 ? 0 : v >= 0.8 ? 255 : Math.round((v - 0.2) * 425);
      const j = (mid + x) * 4;
      md[j] = md[j + 1] = md[j + 2] = g; md[j + 3] = 255;
      ad[j + 3] = g;
      if (g > 127) { n++; sx += x; }
    }
    rowW[y] = n; rowX[y] = n ? sx / n : 0;
    count += n;
  }
  holo.mask.ctx.putImageData(holo.maskImg, 0, 0);
  holo.alpha.ctx.putImageData(holo.alphaImg, 0, 0);
  // Someone in frame: go live after three masks in a row with a person in them; back to the
  // webcam after 1.5 s with nobody (walked off, camera knocked aside).
  const cover = count / (W * H);
  if (cover >= 0.03) { holo.seen++; holo.goneAt = 0; } else if (cover < 0.01) { holo.seen = 0; holo.goneAt ||= now; }
  if (holo.state === 'ready' && holo.seen >= 3 && !document.hidden) goLive();
  else if (holo.state === 'live' && holo.goneAt && now - holo.goneAt > 1500) goRaw();
  if (holo.state === 'live' && holo.segs % 3 === 0) measureFraming();
}

/** Where the head is in the frame, from the mask's row widths: down from the top the silhouette
 *  widens over the head, narrows at the chin and neck, then jumps at the shoulders; the narrowest
 *  row before that jump ends the head. Without a clear neck (hair to the shoulders, a hood) the
 *  head is taken as 1.3 times as tall as it is wide. Smoothed, and sent when it moves. */
function measureFraming() {
  const W = HOLO_W, H = HOLO_H, rw = holo.rowW, rx = holo.rowX, minW = W * 0.03;
  let top = -1;
  for (let y = 0; y < H - 2; y++) if (rw[y] >= minW && rw[y + 1] >= minW && rw[y + 2] >= minW) { top = y; break; }
  if (top < 0) return;
  let widest = 0, shoulders = -1;
  for (let y = top; y < H; y++) {
    if (y - top > 6 && rw[y] > 1.6 * widest) { shoulders = y; break; }
    widest = Math.max(widest, rw[y]);
  }
  let head = 0, headW = 0;
  if (shoulders > 0) {
    let neck = shoulders - 1;
    for (let y = Math.round(top + 0.4 * (shoulders - top)); y < shoulders; y++) if (rw[y] < rw[neck]) neck = y;
    head = neck - top;
    for (let y = top; y < neck; y++) headW = Math.max(headW, rw[y]);
  }
  if (!(head > 0.9 * headW && head < 2 * headW)) {
    // No clear neck: 1.3 times as tall as the silhouette is wide over its top part.
    const span = shoulders > 0 ? 0.5 * (shoulders - top) : 0.25 * H;
    headW = 0;
    for (let y = top; y < Math.min(H, top + span); y++) headW = Math.max(headW, rw[y]);
    head = 1.3 * headW;
  }
  let n = 0, cx = 0;
  for (let y = top; y < Math.min(H, top + 0.8 * head); y++) if (rw[y]) { cx += rx[y]; n++; }
  const f = holo.framing, a = 0.3;
  f.top += (top / H - f.top) * a;
  f.head += (clamp(head / H, 0.12, 0.8) - f.head) * a;
  f.cx += ((n ? cx / n : W / 2) / W - f.cx) * a;
  const s = holo.sent, now = performance.now();
  if (now - holo.sentAt > 500 && (!s || Math.abs(s.top - f.top) + Math.abs(s.head - f.head) + Math.abs(s.cx - f.cx) > 0.02)) sendFraming();
}

function goLive() {
  if (!holo.track) {
    holo.track = holo.packed.c.captureStream(30).getVideoTracks()[0] || null;
    if (!holo.track) { holoFail('this browser cannot send it'); return; }
  }
  holo.state = 'live';
  compose();                                       // the track's first frame is a whole one
  sendOut();
  sendFraming();
  showSelfView();
  syncHoloButton();
  renderPeople();
}

function goRaw() {
  holo.state = 'ready';
  sendOut();
  showSelfView();
  syncHoloButton();
  renderPeople();
}

function compose() {
  const p = holo.packed.ctx;
  p.drawImage(holo.color.c, 0, 0);
  p.drawImage(holo.mask.c, HOLO_W, 0);
  if (!$('selfHolo').hidden) {
    const s = holo.self;
    s.globalCompositeOperation = 'copy';
    s.drawImage(holo.alpha.c, 0, 0);
    s.globalCompositeOperation = 'source-in';
    s.drawImage(holo.color.c, 0, 0);
    s.globalCompositeOperation = 'source-over';
  }
}

/** The video every peer gets: the packed cut-out while it is live, else the webcam itself. */
const outVideo = () => (holo.state === 'live' && holo.track ? holo.track : local.video);

function sendOut() {
  const t = outVideo();
  for (const peer of peers.values()) {
    const s = peer.rtc?.senders.video;
    if (s) s.replaceTrack(t).then(() => tuneSender(s)).catch(() => {});
    else if (t) attachTrack(peer, t);
  }
}

/** The cut-out has twice the webcam's pixels (half of them flat grey): cap what it may cost per
 *  peer, so a mesh of a few people fits a home uplink. The webcam keeps WebRTC's default. */
function tuneSender(sender) {
  try {
    const p = sender.getParameters();
    if (!p.encodings?.length) return;              // not negotiated yet: tuned again once connected
    if (sender.track && sender.track === holo.track) p.encodings[0].maxBitrate = HOLO_KBPS * 1000;
    else delete p.encodings[0].maxBitrate;
    sender.setParameters(p).catch(() => {});
  } catch { /* closing */ }
}

/** Where the head is in the cut-out, to everyone (or one peer whose channel just opened). */
function sendFraming(only) {
  const f = holo.framing;
  const msg = JSON.stringify({ t: 'holo', top: r3(f.top), head: r3(f.head), cx: r3(f.cx) });
  for (const peer of only ? [only] : peers.values()) {
    const dc = peer.rtc?.dc;
    if (dc?.readyState === 'open') { try { dc.send(msg); } catch { /* closing */ } }
  }
  if (!only) { holo.sent = { ...f }; holo.sentAt = performance.now(); }
}

function onPeerData(peer, data) {
  if (typeof data !== 'string' || data.length > 512) return;
  let m;
  try { m = JSON.parse(data); } catch { return; }
  const ok = (v, a, b) => Number.isFinite(v) && v >= a && v <= b;
  if (m?.t === 'holo' && ok(m.top, 0, 0.9) && ok(m.head, 0.05, 1) && ok(m.cx, 0, 1)) {
    peer.framing = { top: m.top, head: m.head, cx: m.cx };
  }
}

// A background tab gets no video frame callbacks, so the cut-out would freeze on its last frame:
// the plain webcam (which keeps running) goes out until the tab is back and sees someone again.
document.addEventListener('visibilitychange', () => { if (document.hidden && holo.state === 'live') goRaw(); });

/* ------------------------------------------------------------------ WebRTC mesh (perfect negotiation) */
// polite = the lower id. The polite side yields on an offer collision, the impolite side ignores
// the incoming offer; together that settles glare without either side tracking who started.
function ensureRtc(peer, initiator = false) {
  if (peer.rtc) return peer.rtc;
  const pc = new RTCPeerConnection({ iceServers: STUN });
  const rtc = { pc, polite: me.id < peer.id, makingOffer: false, ignoreOffer: false, answerPending: false,
                senders: {}, remote: new MediaStream(), restarts: 0, failed: false, dc: null };
  peer.rtc = rtc;

  pc.addEventListener('negotiationneeded', async () => {
    try {
      rtc.makingOffer = true;
      await pc.setLocalDescription();
      sendRtc(peer.id, { description: pc.localDescription });
    } catch (err) { console.warn('rtc offer', err); } finally { rtc.makingOffer = false; }
  });
  pc.addEventListener('icecandidate', ({ candidate }) => { if (candidate) sendRtc(peer.id, { candidate }); });
  // A few ICE restarts ride out a network change; behind a symmetric NAT (STUN only, no TURN)
  // they never succeed, so stop after MAX_ICE_RESTARTS and say so in the people list.
  pc.addEventListener('iceconnectionstatechange', () => {
    const st = pc.iceConnectionState;
    if (st === 'connected' || st === 'completed') {
      rtc.restarts = 0;
      if (rtc.failed) { rtc.failed = false; renderPeople(); }
    } else if (st === 'failed') {
      if (rtc.restarts < MAX_ICE_RESTARTS) { rtc.restarts++; pc.restartIce?.(); }
      else if (!rtc.failed) { rtc.failed = true; renderPeople(); }
    }
  });
  pc.addEventListener('track', ({ track, streams }) => {
    const stream = streams[0] || rtc.remote;
    if (!streams[0]) rtc.remote.addTrack(track);
    attachRemote(peer, stream, track);
  });
  // A sender's bitrate cap needs its encodings, which exist once negotiated.
  pc.addEventListener('connectionstatechange', () => {
    if (pc.connectionState === 'connected' && rtc.senders.video) tuneSender(rtc.senders.video);
  });
  pc.addEventListener('datachannel', ({ channel }) => wireChannel(peer, rtc, channel));

  const video = outVideo();
  if (video) attachTrack(peer, video);
  if (local.audio) attachTrack(peer, local.audio);
  // A newcomer with no camera and no mic still opens the connection, so the other side's tracks
  // have something to ride on. The channel carries only where a hologram's head is (sendFraming).
  if (initiator) wireChannel(peer, rtc, pc.createDataChannel('room'));
  return rtc;
}

function wireChannel(peer, rtc, ch) {
  rtc.dc = ch;
  ch.addEventListener('open', () => sendFraming(peer));
  ch.addEventListener('message', (e) => onPeerData(peer, e.data));
  if (ch.readyState === 'open') sendFraming(peer);
}

function attachTrack(peer, track) {
  const rtc = peer.rtc;
  if (!rtc) return;
  const slot = track.kind === 'video' ? 'video' : 'audio';
  if (rtc.senders[slot]) rtc.senders[slot].replaceTrack(track).catch(() => {});
  else rtc.senders[slot] = rtc.pc.addTrack(track, local.stream);
}

async function onRtc(from, data) {
  if (!data || typeof data !== 'object') return;
  const peer = peers.get(from) || addPeer({ id: from, name: 'Guest' });
  if (!peer) return;
  const rtc = ensureRtc(peer);
  const pc = rtc.pc;
  try {
    if (data.description) {
      const desc = data.description;
      const ready = !rtc.makingOffer && (pc.signalingState === 'stable' || rtc.answerPending);
      const collision = desc.type === 'offer' && !ready;
      rtc.ignoreOffer = !rtc.polite && collision;
      if (rtc.ignoreOffer) return;
      rtc.answerPending = desc.type === 'answer';
      await pc.setRemoteDescription(desc);
      rtc.answerPending = false;
      if (desc.type === 'offer') {
        await pc.setLocalDescription();
        sendRtc(peer.id, { description: pc.localDescription });
      }
    } else if (data.candidate) {
      try { await pc.addIceCandidate(data.candidate); } catch (err) { if (!rtc.ignoreOffer) console.warn('rtc ice', err); }
    }
  } catch (err) { console.warn('rtc', err); }
}

function attachRemote(peer, stream, track) {
  if (!peer.video) {
    peer.video = document.createElement('video');
    peer.video.muted = true; peer.video.playsInline = true; peer.video.autoplay = true;
    peer.video.addEventListener('loadeddata', () => updateAvatarScreen(peer));
    peer.video.addEventListener('resize', () => updateAvatarScreen(peer));
    peer.audio = document.createElement('audio');
    peer.audio.autoplay = true;
    $('mediaSink').append(peer.video, peer.audio);
  }
  if (track.kind === 'video' || peer.video.srcObject !== stream) { peer.video.srcObject = stream; peer.video.play().catch(() => {}); }
  if (track.kind === 'audio' || peer.audio.srcObject !== stream) { peer.audio.srcObject = stream; }
  track.addEventListener('unmute', () => updateAvatarScreen(peer));
  syncPeerMedia(peer);
}

/** Audio plays only while that peer says it sends a mic; the video plane only while it says camera. */
function syncPeerMedia(peer) {
  if (peer.audio) {
    peer.audio.muted = !peer.media.mic;
    if (peer.media.mic) peer.audio.play().catch(() => toast(`Tap anywhere to hear ${peer.name}.`));
  }
  updateAvatarScreen(peer);
}

function closeRtc(peer) {
  if (!peer.rtc) return;
  try { peer.rtc.pc.close(); } catch { /* already closed */ }
  peer.rtc = null;
  if (peer.video) peer.video.srcObject = null;
  if (peer.audio) peer.audio.srcObject = null;
}

function sendRtc(to, data) { send({ t: 'rtc', to, data }); }

/* ------------------------------------------------------------------ WebSocket and reconnect */
let ws = null, wantOnline = false, attempts = 0, retryTimer = 0, joined = false, welcomed = false, booted = false;
let statusShown = '';

function setStatus(kind, text) {
  if (text === statusShown) return;
  statusShown = text;
  $('statusDot').className = 'status-dot ' + (kind === 'ok' ? 'ok' : kind === 'bad' ? 'bad' : '');
  $('statusText').textContent = text;
}

/** While connected the chip says what is still missing: the room download, or a state stream
 *  that has gone quiet (the server is up but its physics loop is not answering). */
function liveStatus(now) {
  if (!me.id) return;
  const pct = Math.round((ver.job && shown ? ver.job.progress : loadFrac) * 100);
  if (!sceneReady) setStatus('wait', `connected · loading room ${pct}%`);
  else if (lastStateAt && now - lastStateAt > 3000) setStatus('wait', 'waiting for the room…');
  else if (!inStep()) setStatus('wait', 'switching to the new room…');
  else if (ver.job) setStatus('ok', ver.job.stage === 'fetch' ? `connected · new room ${pct}%` : 'connected · new room ready');
  else setStatus('ok', 'connected');
}

function send(msg) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
}

function connect() {
  clearTimeout(retryTimer);
  if (ws) { try { ws.close(); } catch { /* closing */ } }
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const url = `${proto}//${location.host}/ws?room=${encodeURIComponent(roomId)}&name=${encodeURIComponent(me.name)}`;
  setStatus('wait', attempts ? 'reconnecting…' : 'connecting…');
  const sock = new WebSocket(url);
  ws = sock;
  sock.addEventListener('open', () => {
    send({ t: 'hello', name: me.name, color: me.color });
    send({ t: 'media', camera: me.media.camera, mic: me.media.mic });
  });
  sock.addEventListener('message', (ev) => {
    if (typeof ev.data !== 'string') return;
    let m;
    try { m = JSON.parse(ev.data); } catch { return; }
    if (m && typeof m.t === 'string') handle(m);
  });
  sock.addEventListener('close', () => socketLost(sock));
}

/** The socket is gone (closed, or silent for too long): forget the room's people and dial again
 *  with backoff. Later events from that socket are ignored because `ws` no longer points at it. */
function socketLost(sock) {
  if (ws !== sock) return;
  ws = null;
  me.id = null;
  endGrab(false);
  clearPeers();
  if (!wantOnline) { setStatus('bad', 'offline'); return; }
  const delay = Math.min(10000, 500 * 2 ** attempts) * (0.75 + Math.random() * 0.5);
  attempts++;
  setStatus('bad', `offline — retrying in ${Math.ceil(delay / 1000)} s`);
  if (!welcomed) showJoinError('Cannot reach the room server yet — retrying…', false);
  retryTimer = setTimeout(connect, delay);
}

addEventListener('online', () => { if (wantOnline && !ws) { attempts = 0; connect(); } });

function handle(m) {
  switch (m.t) {
    case 'welcome': {
      me.id = String(m.id);
      // The server cleans names and colours; use its version so everyone draws the same thing.
      if (typeof m.name === 'string' && m.name) me.name = m.name.slice(0, 40);
      if (typeof m.color === 'string' && /^#[0-9a-fA-F]{6}$/.test(m.color)) me.color = m.color;
      $('hudRoom').textContent = roomId;
      attempts = 0;
      lastStateAt = performance.now();             // the dead-socket watchdog counts from here
      // The version the room is on right now. After a reconnect that may be another one (the
      // server restarted with a new scene, or swapped while we were away): it is loaded and
      // swapped in like any other, no page reload.
      const v = versionOf(m);
      if (v !== undefined) { ver.gone.delete(v); ver.room = v; ver.frame = null; reconsider(); }
      clearPeers();
      for (const p of m.peers || []) {
        const peer = addPeer(p);
        if (peer) ensureRtc(peer, true);         // the newcomer dials everyone already here
      }
      liveStatus(performance.now());
      if (!welcomed) { welcomed = true; enterRoom(); }
      renderPeople();
      break;
    }
    case 'join': {
      const peer = addPeer(m.peer);
      if (peer) toast(`${peer.name} joined`);
      break;
    }
    case 'leave': {
      const peer = peers.get(m.id);
      if (peer) { toast(`${peer.name} left`); removePeer(m.id); }
      break;
    }
    case 'state': onState(m); break;
    case 'scene': {
      // A new version of the room: fetch it now; the room switches to it a few seconds later, at
      // the first state frame that carries it (or already has, if the server did not wait). The
      // version the room is already on means the opposite: whatever was pending is withdrawn.
      const v = versionOf(m);
      if (!v) break;
      ver.gone.delete(v);
      ver.next = v === ver.room ? undefined : v;
      ver.nextAt = performance.now();
      reconsider();
      break;
    }
    case 'media': {
      const peer = peers.get(m.id);
      if (!peer) break;
      peer.media = { camera: !!m.camera, mic: !!m.mic };
      syncPeerMedia(peer);
      renderPeople();
      break;
    }
    case 'rtc': if (typeof m.from === 'string') onRtc(m.from, m.data); break;
    case 'reset': {
      if (m.by === 'server') toast('The physics hiccuped, so the room was reset');
      else toast(`${m.by === me.id ? 'You' : (peers.get(m.by)?.name || 'Someone')} reset the room`);
      endGrab(true);                                // a grab sent after the server's reset is still live there
      for (const b of bodies.values()) b.has = false;   // snap to the reset pose, no slow glide
      break;
    }
    case 'error': {
      const msg = String(m.msg || 'error');
      if (!me.id) {                                 // refused before welcome (room full, bad room…)
        wantOnline = false;
        try { ws?.close(); } catch { /* closing */ }
        showJoinError(msg, true);
      } else toast(msg, 'error');
      break;
    }
    default: break;
  }
}

/* ------------------------------------------------------------------ join screen */
function showJoinError(msg, fatal) {
  const el = $('joinError');
  el.textContent = msg; el.hidden = !msg;
  if (fatal) {
    $('join').hidden = false; $('hud').hidden = true;
    joined = false; welcomed = false;
    $('joinBtn').disabled = false; $('joinBtn').textContent = 'Try again';
  }
}

function buildSwatches() {
  const box = $('swatches');
  for (const c of COLORS) {
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'swatch'; b.style.background = c;
    b.setAttribute('role', 'radio'); b.setAttribute('aria-label', 'Colour ' + c);
    b.setAttribute('aria-checked', String(c === me.color));
    b.addEventListener('click', () => {
      me.color = c;
      for (const s of box.children) s.setAttribute('aria-checked', String(s === b));
    });
    box.appendChild(b);
  }
}

function prepareJoinScreen() {
  $('roomName').textContent = roomId;
  $('hudRoom').textContent = roomId;
  $('nameInput').value = me.name;
  $('camInput').checked = store.get('camera', true);
  $('micInput').checked = false;                  // never remembered on: the call usually carries voice
  if (!canMedia) {
    $('secureNote').hidden = false;
    for (const id of ['camInput', 'micInput']) { $(id).checked = false; $(id).disabled = true; $(id).closest('label').classList.add('disabled'); }
  }
  if (zoom) {
    $('zoomBadge').hidden = false;
    $('micInput').checked = false; $('micInput').disabled = true;
    $('micRow').classList.add('disabled');
    $('micHint').textContent = 'Inside Zoom the meeting carries your voice, so this stays off.';
  }
  if (!me.name) $('nameInput').focus();
}

$('joinForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  if (joined || !booted) return;                  // Enter before the room id is known
  me.name = $('nameInput').value.trim().replace(/\s+/g, ' ').slice(0, 40) || 'Guest';
  store.set('name', me.name); store.set('color', me.color); store.set('camera', $('camInput').checked);
  joined = true;
  $('joinError').hidden = true;
  $('joinBtn').disabled = true; $('joinBtn').textContent = 'Joining…';
  // The click is the user gesture browsers want before camera prompts and audio playback.
  if ($('camInput').checked && !local.video) await setLocal('video', true);
  if ($('micInput').checked && !zoom && !local.audio) await setLocal('audio', true);
  wantOnline = true; attempts = 0;
  connect();
});

function enterRoom() {
  $('join').hidden = true;
  $('hud').hidden = false;
  $('joystick').hidden = !IS_TOUCH;
  $('joinBtn').disabled = false; $('joinBtn').textContent = 'Join the room';
  renderPeople();
  syncMediaButtons();
  canvas.focus({ preventScroll: true });
  if (!store.get('helpSeen', false)) { togglePanel('help', true); store.set('helpSeen', true); }
}

/* ------------------------------------------------------------------ HUD buttons */
function togglePanel(id, force) {
  const el = $(id);
  const open = force ?? el.hidden;
  for (const other of ['people', 'help', 'invitePop']) if (other !== id) $(other).hidden = true;
  el.hidden = !open;
  $('peopleBtn').setAttribute('aria-expanded', String(!$('people').hidden));
  $('helpBtn').setAttribute('aria-expanded', String(!$('help').hidden));
}
function closePanels() {
  for (const id of ['people', 'help', 'invitePop']) $(id).hidden = true;
  $('peopleBtn').setAttribute('aria-expanded', 'false');
  $('helpBtn').setAttribute('aria-expanded', 'false');
}

$('peopleBtn').addEventListener('click', () => togglePanel('people'));
$('helpBtn').addEventListener('click', () => togglePanel('help'));
$('camBtn').addEventListener('click', () => setLocal('video', !shownOn('video')));
$('holoBtn').addEventListener('click', () => {
  if (holo.state === 'failed' && holo.want) {      // unavailable: a click tries again
    if (holo.loadFailed) { holo.segLoad = null; holo.loadFailed = false; }
    Object.assign(holo, { state: 'off', why: '', segs: 0, slowSince: 0 });
  } else {
    holo.want = !holo.want;
    store.set('hologram', holo.want);
    if (holo.state === 'failed') holo.state = 'off';
  }
  syncHologram();
  if (holo.want && !local.video) toast('Turn your camera on to appear as a hologram');
});
$('micBtn').addEventListener('click', () => { if (!zoom) setLocal('audio', !shownOn('audio')); });
$('inviteClose').addEventListener('click', () => togglePanel('invitePop', false));

$('inviteBtn').addEventListener('click', async () => {
  const link = location.origin + '/?room=' + encodeURIComponent(roomId);
  try {
    await navigator.clipboard.writeText(link);
    toast('Invite link copied — paste it in the Zoom, Meet or Teams chat');
  } catch {
    // No clipboard (insecure page, or an embedded browser such as Zoom's): show it to copy by hand.
    $('inviteInput').value = link;
    togglePanel('invitePop', true);
    $('inviteInput').focus(); $('inviteInput').select();
  }
});

let resetArmed = 0;
$('resetBtn').addEventListener('click', () => {
  const btn = $('resetBtn'), txt = btn.querySelector('.tool-txt');
  if (!resetArmed) {
    // Two clicks rather than confirm(): embedded browsers (Zoom) may suppress dialogs.
    btn.classList.add('armed'); txt.textContent = 'Reset for all?';
    resetArmed = setTimeout(() => { btn.classList.remove('armed'); txt.textContent = 'Reset'; resetArmed = 0; }, 3000);
    toast('Click Reset again to put every object back — for everyone in the room');
    return;
  }
  clearTimeout(resetArmed); resetArmed = 0;
  btn.classList.remove('armed'); txt.textContent = 'Reset';
  send({ t: 'reset' });
});

// A tap unlocks audio playback on browsers that blocked it when a peer's mic arrived.
addEventListener('pointerdown', () => { for (const p of peers.values()) if (p.media.mic && p.audio?.paused) p.audio.play().catch(() => {}); }, true);

/* ------------------------------------------------------------------ Zoom App mode */
/** Inside the Zoom client: everyone in the meeting gets the same room (hash of the meeting UUID)
 *  and their Zoom screen name. Detected by Zoom's embedded-browser user agent, or forced with
 *  ?zoom=1 (set that on the app's home URL to be sure). Any failure falls back to ?room / lobby. */
async function detectZoom() {
  const ua = /ZoomApps|ZoomWebKit|ZoomRooms/i.test(navigator.userAgent);
  if (!ua && params.get('zoom') !== '1') return null;
  try {
    if (!window.zoomSdk) await loadScript(ZOOM_SDK);
    const sdk = window.zoomSdk;                    // the CDN build defines this global
    if (!sdk?.config) return null;
    const cfg = await withTimeout(sdk.config({
      version: '0.16',
      capabilities: ['getRunningContext', 'getMeetingUUID', 'getUserContext'],
    }), 8000);
    const out = { context: cfg?.runningContext || null, room: null, name: null };
    try {
      const { meetingUUID } = await withTimeout(sdk.getMeetingUUID(), 5000);
      if (meetingUUID) out.room = await shortHash(String(meetingUUID));
    } catch (err) { console.warn('zoom getMeetingUUID', err); }
    try {
      const u = await withTimeout(sdk.getUserContext(), 5000);
      if (u?.screenName) out.name = String(u.screenName).slice(0, 40);
    } catch (err) { console.warn('zoom getUserContext', err); }
    return out;
  } catch (err) {
    console.warn('zoom sdk unavailable', err);
    return null;
  }
}

/* ------------------------------------------------------------------ frame loop */
const clock = new THREE.Clock();
let lastPose = 0, lastPoseMsg = '';
let lastFrameAt = 0;

function frame() {
  const dt = Math.min(clock.getDelta(), 0.2);   // a slow frame still walks, a stall does not teleport
  const now = performance.now();
  // After this page itself stood still for a while (a long task such as the segmenter starting
  // on a slow machine, or a tab the browser paused), the state frames sent meanwhile are still
  // waiting to be read: the silence was ours, so it does not count toward a dead socket.
  if (lastFrameAt && now - lastFrameAt > 1000) lastStateAt += now - lastFrameAt;
  lastFrameAt = now;
  stepWalker(dt);
  stepBodies(dt);
  stepAvatars(dt);
  stepGrabLines();
  updateGrabTarget(now);

  if (hoverAt && !grab && gesture === 'none' && now - hoverPickedAt > 70) {
    setHover(pick(hoverAt.x, hoverAt.y)?.body ?? null);   // occluded: no grab cursor through walls
    hoverAt = null; hoverPickedAt = now;
  }
  if (me.id && now - lastPose > 1000 / POSE_HZ) {
    const msg = JSON.stringify({ t: 'pose', p: threeToMj(walker.pos), yaw: r3(walker.yaw), pitch: r3(walker.pitch) });
    if (msg !== lastPoseMsg || now - lastPose > 1000) { if (ws?.readyState === WebSocket.OPEN) ws.send(msg); lastPoseMsg = msg; }
    lastPose = now;
  }
  // A socket can die without a 'close' (laptop sleep, a NAT or tunnel dropping it): the server
  // sends state at 30 Hz, so a long silence after the welcome means it is gone. Treat it as closed
  // now; the real 'close' may take minutes and is ignored when it comes.
  if (ws?.readyState === WebSocket.OPEN && me.id && now - lastStateAt > STATE_DEAD_MS) {
    const sock = ws;
    socketLost(sock);
    try { sock.close(); } catch { /* closing */ }
  }
  liveStatus(now);
  renderer.render(scene, camera);
}

/* ------------------------------------------------------------------ boot */
(async function boot() {
  buildSwatches();
  canvas.tabIndex = 0;
  // Join waits for the room id: inside Zoom it comes from the meeting, which takes a moment. The
  // room download does not wait for Zoom.
  $('joinBtn').disabled = true; $('joinBtn').textContent = 'Starting…';
  reconsider();                                    // nothing on screen: load whatever is live
  syncMediaButtons();
  stepWalker(0);
  renderer.setAnimationLoop(frame);
  zoom = await detectZoom();
  if (!welcomed) {                                 // never move someone who is already in a room
    if (zoom?.room) roomId = zoom.room;
    if (zoom?.name) me.name = zoom.name;          // inside Zoom the meeting's screen name wins
  }
  prepareJoinScreen();
  syncMediaButtons();
  booted = true;
  $('joinBtn').disabled = false; $('joinBtn').textContent = 'Join the room';
})();
