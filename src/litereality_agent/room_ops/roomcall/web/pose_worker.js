/* The room call's pose tracker, off the main thread: MediaPipe's PoseLandmarker (lite) on the
   sender's own camera frames, for their full-body avatar's arms and head (app.js `poseTrack`).

   A module worker, so the page's thread (three.js, the hologram's segmenter) never waits on it.
   The runtime comes from jsDelivr at the version app.js pins (the CSP admits it and WebAssembly);
   the model from this server ('self'). Nothing leaves the device but the joint angles app.js
   computes from what this returns.

   in:  {t: 'init', base: MP_BASE, model: '/pose_landmarker_lite.task'}
        {t: 'frame', bmp: ImageBitmap, ts}             (the bitmap is transferred, and closed here)
   out: {t: 'ready'} | {t: 'error', msg}
        {t: 'pose', world: Float32Array(33*4) | null, image: Float32Array(33*2) | null, ms}
        world: metres, x y z visibility per landmark; image: normalized x y. null: nobody in view. */

let pose = null, starting = false, ts = 0;

// The runtime logs in glog's format ("W1021 16:23:58.43 ... ] ...") as it starts: delegate notes
// and GL hints, not ours to act on (real failures throw). Those go to debug; anything else passes.
const GLOG = /^[IWEF]\d{4} \d\d:\d\d:\d\d/;
for (const k of ['log', 'info', 'warn', 'error']) {
  const orig = console[k].bind(console);
  console[k] = (...a) => (typeof a[0] === 'string' && GLOG.test(a[0]) ? console.debug('[mediapipe]', ...a) : orig(...a));
}
self.Module = { print: (t) => console.debug('[mediapipe]', t), printErr: (t) => console.debug('[mediapipe]', t) };

async function init(base, model) {
  if (pose || starting) return;
  starting = true;
  try {
    const vision = await import(`${base}/vision_bundle.mjs`);
    // true: load the WebAssembly runtime as an ES module (a module worker has no importScripts).
    const files = await vision.FilesetResolver.forVisionTasks(`${base}/wasm`, true);
    pose = await vision.PoseLandmarker.createFromOptions(files, {
      baseOptions: { modelAssetPath: model, delegate: 'CPU' },   // WebAssembly SIMD: no GL contention
      runningMode: 'VIDEO', numPoses: 1,
    });
    self.postMessage({ t: 'ready' });
  } catch (err) {
    self.postMessage({ t: 'error', msg: String(err?.message || err) });
  } finally { starting = false; }
}

function frame(bmp, at) {
  if (!pose) { bmp?.close?.(); return; }
  const t0 = performance.now();
  let world = null, image = null;
  try {
    ts = Math.max(ts + 1, Math.round(at) || 0);         // VIDEO mode wants increasing timestamps
    pose.detectForVideo(bmp, ts, (r) => {
      const w = r.worldLandmarks?.[0], l = r.landmarks?.[0];
      if (!w || !l || w.length < 33 || l.length < 33) return;
      world = new Float32Array(33 * 4); image = new Float32Array(33 * 2);
      for (let i = 0; i < 33; i++) {
        world[4 * i] = w[i].x; world[4 * i + 1] = w[i].y; world[4 * i + 2] = w[i].z;
        world[4 * i + 3] = w[i].visibility ?? l[i].visibility ?? 1;
        image[2 * i] = l[i].x; image[2 * i + 1] = l[i].y;
      }
    });
  } catch (err) {
    self.postMessage({ t: 'error', msg: String(err?.message || err) });
    return;
  } finally { bmp.close(); }
  self.postMessage({ t: 'pose', world, image, ms: performance.now() - t0 }, world ? [world.buffer, image.buffer] : []);
}

self.addEventListener('message', (e) => {
  const m = e.data;
  if (m?.t === 'init' && typeof m.base === 'string' && typeof m.model === 'string') init(m.base, m.model);
  else if (m?.t === 'frame' && m.bmp) frame(m.bmp, m.ts);
});
