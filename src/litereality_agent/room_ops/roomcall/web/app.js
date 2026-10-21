/* Room call client: one shared MuJoCo room in the browser, for the people on a video call.

   The server owns the physics. This page only draws what it is told (the "state" stream), says
   where you are ("pose") and what you are holding ("grab" / "move" / "release"), and relays
   WebRTC signalling so webcams can ride on the avatars peer to peer. Everything on the wire is in
   MuJoCo WORLD coordinates (Z-up, metres); the conversion to three.js (Y-up) happens in exactly
   two places — the `world` group's rotation for the room itself, and mjToThree / threeToMj for
   the camera, avatars and pointer rays.

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
const grabLines = new Map();                       // peer id -> {line, dot}
const dotGeo = new THREE.SphereGeometry(0.025, 12, 8);

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
      geo.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(6), 3));
      const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color, depthTest: false, transparent: true, opacity: 0.9 }));
      const dot = new THREE.Mesh(dotGeo, new THREE.MeshBasicMaterial({ color, depthTest: false }));
      line.renderOrder = dot.renderOrder = 10;
      line.frustumCulled = false;
      world.add(line, dot);                       // inside `world`: MuJoCo coordinates as-is
      rec = { line, dot };
      grabLines.set(pid, rec);
    }
    const pos = rec.line.geometry.attributes.position;
    pos.setXYZ(0, ax, ay, az); pos.setXYZ(1, tx, ty, tz); pos.needsUpdate = true;
    rec.dot.position.set(ax, ay, az);
  }
  for (const [pid, rec] of grabLines) {
    if (seen.has(pid)) continue;
    world.remove(rec.line, rec.dot);
    rec.line.geometry.dispose(); rec.line.material.dispose(); rec.dot.material.dispose();
    grabLines.delete(pid);
  }
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
  const label = labelSprite(peer.name, peer.color);
  label.position.y = LABEL_Y;
  root.add(torso, head, face, label);
  root.visible = false;                           // until their first pose arrives
  scene.add(root);
  return { root, head, face, screen, label, skin, sw: FACE_H * 4 / 3, sh: FACE_H,
           tp: new THREE.Vector3(), yaw: 0, pitch: 0, tyaw: 0, tpitch: 0, has: false };
}

function disposeAvatar(av) {
  scene.remove(av.root);
  av.root.traverse((o) => {
    o.geometry?.dispose?.();
    for (const m of [].concat(o.material || [])) { m.map?.dispose?.(); m.dispose?.(); }
  });
}

function updateAvatarScreen(peer) {
  const av = peer.avatar, v = peer.video;
  const show = !!(peer.media.camera && v && v.readyState >= 2 && v.videoWidth > 0);
  if (show && !peer.tex) {
    peer.tex = new THREE.VideoTexture(v);
    peer.tex.colorSpace = THREE.SRGBColorSpace;
    av.screen.material.map = peer.tex;
    av.screen.material.needsUpdate = true;
  }
  if (show) {
    const aspect = clamp(v.videoWidth / v.videoHeight, 0.5, 2);
    av.sw = Math.min(FACE_H * 4 / 3, FACE_H * aspect);   // FACE_H tall, no wider than 4:3
    av.sh = av.sw / aspect;
  }
  av.screen.visible = show;
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
  const row = (name, color, media, you, rtcFailed = false) => {
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
      const t = document.createElement('span'); t.className = 'tag' + (media[k] ? ' on' : ''); t.textContent = label + (media[k] ? ' on' : ' off');
      tags.appendChild(t);
    }
    li.append(dot, who, tags);
    list.appendChild(li);
  };
  row(me.name || 'You', me.color, me.media, true);
  for (const p of peers.values()) row(p.name, p.color, p.media, false, !!p.rtc?.failed);
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
  showSelfView();
  syncMediaButtons();
  send({ t: 'media', camera: me.media.camera, mic: me.media.mic });
  renderPeople();
  return !!local[slot];
}

/** What a media button shows: the wish while the browser is still asking, else the real state. */
const shownOn = (slot) => (local.asking[slot] ? local.want[slot] : !!local[slot]);

function showSelfView() {
  const v = $('selfView');
  if (local.video) { if (v.srcObject !== local.stream) v.srcObject = local.stream; v.hidden = false; v.play().catch(() => {}); }
  else { v.hidden = true; }
}

function syncMediaButtons() {
  $('camBtn').setAttribute('aria-pressed', String(shownOn('video')));
  $('micBtn').setAttribute('aria-pressed', String(shownOn('audio')));
  $('camBtn').disabled = !canMedia;
  $('micBtn').disabled = !canMedia || !!zoom;
  $('micBtn').title = zoom ? 'Zoom carries your voice' : 'Send your microphone';
}

/* ------------------------------------------------------------------ WebRTC mesh (perfect negotiation) */
// polite = the lower id. The polite side yields on an offer collision, the impolite side ignores
// the incoming offer; together that settles glare without either side tracking who started.
function ensureRtc(peer, initiator = false) {
  if (peer.rtc) return peer.rtc;
  const pc = new RTCPeerConnection({ iceServers: STUN });
  const rtc = { pc, polite: me.id < peer.id, makingOffer: false, ignoreOffer: false, answerPending: false,
                senders: {}, remote: new MediaStream(), restarts: 0, failed: false };
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

  if (local.video) attachTrack(peer, local.video);
  if (local.audio) attachTrack(peer, local.audio);
  // A newcomer with no camera and no mic still opens the connection, so the other side's tracks
  // have something to ride on; the channel itself carries nothing.
  if (initiator) pc.createDataChannel('room');
  return rtc;
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

function frame() {
  const dt = Math.min(clock.getDelta(), 0.2);   // a slow frame still walks, a stall does not teleport
  const now = performance.now();
  stepWalker(dt);
  stepBodies(dt);
  stepAvatars(dt);
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
