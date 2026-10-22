/* Full-body people for the room call: everything about a body that is plain maths, with no
   imports, so app.js imports it statically and node can import it for checks.

   - The LRAV v1 reader, the head mask and forward kinematics (normative: they mirror
     room_ops/roomcall/avatar_format.py and are checked against its files).
   - A small vector / quaternion kit (vectors [x,y,z], quaternions [x,y,z,w], three.js order).
   - The SENDER's upper-body solver: MediaPipe PoseLandmarker world landmarks -> SMPL-X local
     joint rotations for the spine, neck, head, collars, shoulders, elbows and wrists, packed as
     13 rotation vectors in centiradians for the "pose" message ("j").
   - The RECEIVER's procedural legs (`Gait`): feet planted in world space, the step phase driven
     by distance walked, two-bone IK, nothing on the wire.
   - `relaxedUpperBody`: what a receiver shows while a sender is not tracked, and `assembleLocal`,
     which puts one frame's pose together from the two.

   The body frame is SMPL-X's canonical one: metres, +Y up, the person faces +Z, +X is the
   person's LEFT, rest pose a T-pose. */

// ---- LRAV v1 reader (normative; mirrors room_ops/roomcall/avatar_format.py) -----------------
export const LRAV_MAX_BYTES = 5 * 1024 * 1024, LRAV_MAX_SPLATS = 120000;
export const BODY_JOINT_NAMES = ['Pelvis', 'L_Hip', 'R_Hip', 'Spine_1', 'L_Knee', 'R_Knee', 'Spine_2', 'L_Ankle',
  'R_Ankle', 'Spine_3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder',
  'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist'];
export const HEAD_JOINTS = [15, 22, 23, 24];
const LRAV_ARRAYS = [['positions', '<f4', 3], ['scales', '<f2', 3], ['rotations', '<f2', 4], ['colors', '|u1', 4], ['skin', '<u2', 4]];

let F16 = null;
function f16Table() {
  if (F16) return F16;
  F16 = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
    F16[h] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return F16;
}

/** ArrayBuffer -> {count, kind, floorY, height, meta, names, parents, rest (Float32Array J*3),
 *  positions (Float32Array n*3), scales (Float32Array n*3), rotations (Float32Array n*4, x y z w),
 *  colors (Uint8Array n*4), skin (Uint16Array n*4)}. Throws on anything structurally off. */
export function parseAvatar(buf) {
  if (!(buf instanceof ArrayBuffer) || buf.byteLength < 16 || buf.byteLength > LRAV_MAX_BYTES) throw new Error('avatar: bad size');
  const dv = new DataView(buf);
  if (dv.getUint32(0, true) !== 0x5641524c) throw new Error('avatar: bad magic');      // "LRAV"
  if (dv.getUint32(4, true) !== 1) throw new Error('avatar: unsupported version');
  const hlen = dv.getUint32(8, true);
  if (hlen % 16 || hlen <= 0 || hlen > 65536 || 16 + hlen > buf.byteLength) throw new Error('avatar: bad header length');
  const h = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 16, hlen)));
  const n = h.count, J = h.joints?.names?.length;
  if (h.format !== 'lrav' || h.version !== 1 || !Number.isInteger(n) || n < 1 || n > LRAV_MAX_SPLATS) throw new Error('avatar: bad header');
  if (!(J >= 22 && J <= 64) || BODY_JOINT_NAMES.some((name, i) => h.joints.names[i] !== name)) throw new Error('avatar: bad skeleton');
  const parents = h.joints.parents, restRows = h.joints.rest;
  if (!Array.isArray(parents) || parents.length !== J || parents[0] !== -1 || parents.some((p, j) => j > 0 && !(Number.isInteger(p) && p >= 0 && p < j))) throw new Error('avatar: bad parents');
  if (!Array.isArray(restRows) || restRows.length !== J) throw new Error('avatar: bad rest');
  const rest = new Float32Array(J * 3);
  restRows.forEach((r, j) => { if (!Array.isArray(r) || r.length !== 3 || !r.every(Number.isFinite)) throw new Error('avatar: bad rest'); rest.set(r, 3 * j); });
  const start = 16 + hlen, out = { count: n, kind: String(h.kind), floorY: +h.floor_y, height: +h.height, meta: h.meta || {}, names: h.joints.names, parents, rest };
  if (!Number.isFinite(out.floorY)) throw new Error('avatar: bad floor');
  for (const [name, dtype, cols] of LRAV_ARRAYS) {
    const a = h.arrays?.[name], size = dtype === '<f4' ? 4 : dtype === '|u1' ? 1 : 2;
    if (!a || a.dtype !== dtype || a.shape?.[0] !== n || a.shape?.[1] !== cols || !Number.isInteger(a.offset) || a.offset % 16) throw new Error(`avatar: bad ${name}`);
    const at = start + a.offset, len = n * cols;
    if (at + len * size > buf.byteLength) throw new Error(`avatar: ${name} past the end`);
    if (dtype === '<f4') out[name] = new Float32Array(buf, at, len);
    else if (dtype === '|u1') out[name] = new Uint8Array(buf, at, len);
    else if (name === 'skin') out[name] = new Uint16Array(buf, at, len);
    else { const t = f16Table(), raw = new Uint16Array(buf, at, len), f = new Float32Array(len); for (let i = 0; i < len; i++) f[i] = t[raw[i]]; out[name] = f; }
  }
  for (let i = 0; i < 4 * n; i++) if ((out.skin[i] >> 8) >= J) throw new Error('avatar: skin joint past the skeleton');
  for (let i = 0; i < 3 * n; i++) if (!Number.isFinite(out.positions[i])) throw new Error('avatar: non-finite position');
  return out;
}

/** Splats whose weight on HEAD_JOINTS is >= 128/255 (same rule as Avatar.head_mask). */
export function headMask(skin, n) {
  const m = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    let w = 0;
    for (let k = 0; k < 4; k++) { const v = skin[4 * i + k]; if (HEAD_JOINTS.includes(v >> 8)) w += v & 255; }
    m[i] = w >= 128 ? 1 : 0;
  }
  return m;
}

// ---- forward kinematics and Spark skin transforms (normative) ------------------------------
// Quaternions are [x, y, z, w] arrays. local[j]: joint j's rotation relative to its parent,
// relative to the rest (zero, T-) pose; local[0] and rootPos are the pelvis' WORLD rotation and
// position (three.js world, Y up). Rest rotations are all identity, so the skin transform of joint
// j is S_j = G_j * B_j^-1 with B_j = (identity, rest_j): S_j(x) = q_j (x - rest_j) + p_j.
const qMul = (a, b) => [a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1], a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
  a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3], a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]];
const qRot = (q, v) => { const [x, y, z, w] = q, [vx, vy, vz] = v;
  const tx = 2 * (y * vz - z * vy), ty = 2 * (z * vx - x * vz), tz = 2 * (x * vy - y * vx);
  return [vx + w * tx + y * tz - z * ty, vy + w * ty + z * tx - x * tz, vz + w * tz + x * ty - y * tx]; };

/** -> {q: Float32Array(J*4) world rotations, p: Float32Array(J*3) world positions,
 *      sq: Float32Array(J*4), st: Float32Array(J*3) the skin transforms for setBoneQuatPos}. */
export function poseSkeleton(parents, rest, local, rootPos, out) {
  const J = parents.length;
  out ??= { q: new Float32Array(J * 4), p: new Float32Array(J * 3), sq: new Float32Array(J * 4), st: new Float32Array(J * 3) };
  for (let j = 0; j < J; j++) {
    const L = local[j] || [0, 0, 0, 1], par = parents[j];
    let q, p;
    if (par < 0) { q = L; p = rootPos; } else {
      const pq = [out.q[4 * par], out.q[4 * par + 1], out.q[4 * par + 2], out.q[4 * par + 3]];
      const d = qRot(pq, [rest[3 * j] - rest[3 * par], rest[3 * j + 1] - rest[3 * par + 1], rest[3 * j + 2] - rest[3 * par + 2]]);
      q = qMul(pq, L); p = [out.p[3 * par] + d[0], out.p[3 * par + 1] + d[1], out.p[3 * par + 2] + d[2]];
    }
    out.q.set(q, 4 * j); out.p.set(p, 3 * j); out.sq.set(q, 4 * j);
    const r = qRot(q, [rest[3 * j], rest[3 * j + 1], rest[3 * j + 2]]);
    out.st.set([p[0] - r[0], p[1] - r[1], p[2] - r[2]], 3 * j);
  }
  return out;
}

// ---- vectors and quaternions ----------------------------------------------------------------
// qMul and qRot are the normative ones above. Few allocations-avoidance tricks: ~25 joints a frame.
export { qMul, qRot };
export const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
export const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const len = (a) => Math.hypot(a[0], a[1], a[2]);
export const norm = (a) => { const l = len(a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
export const mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
export const lerp3 = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
export const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
export const smoothstep = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
/** Orthonormal frame from a primary axis u and a hint v (v made perpendicular to u): [u, v', u x v']. */
export const basis = (u, v) => { const x = norm(u); const y = norm(sub(v, scale(x, dot(v, x)))); return [x, y, cross(x, y)]; };

export const QI = () => [0, 0, 0, 1];
export const qInv = (q) => [-q[0], -q[1], -q[2], q[3]];
export const qNorm = (q) => { const l = Math.hypot(...q) || 1; return q.map((c) => c / l); };
export const qAxisAngle = (axis, ang) => { const a = norm(axis), s = Math.sin(ang / 2); return [a[0] * s, a[1] * s, a[2] * s, Math.cos(ang / 2)]; };
/** Shortest rotation taking unit a onto unit b (the "swing"). Antiparallel: 180 deg about any axis normal to a. */
export const qFromTo = (a, b) => {
  const d = dot(a, b);
  if (d < -0.999999) { let ax = cross([1, 0, 0], a); if (len(ax) < 1e-6) ax = cross([0, 1, 0], a); return qAxisAngle(ax, Math.PI); }
  const c = cross(a, b);
  return qNorm([c[0], c[1], c[2], 1 + d]);
};
/** Rotation whose matrix has columns x, y, z (a right-handed orthonormal frame). */
export const qFromBasis = ([x, y, z]) => {
  const m00 = x[0], m10 = x[1], m20 = x[2], m01 = y[0], m11 = y[1], m21 = y[2], m02 = z[0], m12 = z[1], m22 = z[2];
  const tr = m00 + m11 + m22;
  let q;
  if (tr > 0) { const s = 0.5 / Math.sqrt(tr + 1); q = [(m21 - m12) * s, (m02 - m20) * s, (m10 - m01) * s, 0.25 / s]; }
  else if (m00 > m11 && m00 > m22) { const s = 2 * Math.sqrt(1 + m00 - m11 - m22); q = [0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s]; }
  else if (m11 > m22) { const s = 2 * Math.sqrt(1 + m11 - m00 - m22); q = [(m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s]; }
  else { const s = 2 * Math.sqrt(1 + m22 - m00 - m11); q = [(m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s]; }
  return qNorm(q);
};
export const qAngle = (q) => 2 * Math.atan2(Math.hypot(q[0], q[1], q[2]), Math.abs(q[3]));
export const qToRotvec = (q) => {
  const p = q[3] < 0 ? q.map((c) => -c) : q;            // shortest arc
  const s = Math.hypot(p[0], p[1], p[2]);
  if (s < 1e-9) return [0, 0, 0];
  const ang = 2 * Math.atan2(s, p[3]);
  return [p[0] / s * ang, p[1] / s * ang, p[2] / s * ang];
};
export const qFromRotvec = (r) => { const a = len(r); return a < 1e-9 ? QI() : qAxisAngle(r, a); };
/** q^t: same axis, angle scaled by t (q^a * q^b = q^(a+b), so a chain can share one rotation). */
export const qPow = (q, t) => qFromRotvec(scale(qToRotvec(q), t));
export const qClampAngle = (q, max) => { const r = qToRotvec(q), a = len(r); return a > max ? qFromRotvec(scale(r, max / a)) : q; };
export const qSlerp = (a, b, t) => {
  let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  const bb = d < 0 ? b.map((c) => -c) : b; d = Math.abs(d);
  if (d > 0.9995) return qNorm(a.map((c, i) => c + (bb[i] - c) * t));
  const th = Math.acos(d), s = Math.sin(th);
  const wa = Math.sin((1 - t) * th) / s, wb = Math.sin(t * th) / s;
  return a.map((c, i) => wa * c + wb * bb[i]);
};
/** Swing-twist split about unit axis: q = swing * twist, twist a rotation about axis. Returns the twist angle. */
export const twistAngle = (q, axis) => {
  const p = dot([q[0], q[1], q[2]], axis);
  return 2 * Math.atan2(p, q[3]);
};
export const wrapPi = (a) => a - 2 * Math.PI * Math.floor((a + Math.PI) / (2 * Math.PI));

// ---- the sender's upper body: MediaPipe world landmarks -> SMPL-X local rotations ------------
// MediaPipe world landmarks: metres, origin mid-hip, +x image right, +y image down, +z away from
// the camera (checked on an unmirrored photo). A person facing an unmirrored camera has their
// left on image right, so the body frame B = (x, -y, -z): a 180 deg turn about x. Rotations out
// are SMPL-X body_pose style: local, relative to the zero (T-) pose whose joint frames are all
// world-aligned. No mirroring: the person's real left arm drives the avatar's left arm.
const X = [1, 0, 0], Y = [0, 1, 0], Z = [0, 0, 1];

export const MP = { NOSE: 0, L_EYE: 2, R_EYE: 5, L_EAR: 7, R_EAR: 8, L_SH: 11, R_SH: 12, L_EL: 13, R_EL: 14,
  L_WR: 15, R_WR: 16, L_PINKY: 17, R_PINKY: 18, L_INDEX: 19, R_INDEX: 20, L_THUMB: 21, R_THUMB: 22, L_HIP: 23, R_HIP: 24 };
export const J = { pelvis: 0, left_hip: 1, right_hip: 2, spine1: 3, left_knee: 4, right_knee: 5, spine2: 6, left_ankle: 7,
  right_ankle: 8, spine3: 9, left_foot: 10, right_foot: 11, neck: 12, left_collar: 13, right_collar: 14, head: 15,
  left_shoulder: 16, right_shoulder: 17, left_elbow: 18, right_elbow: 19, left_wrist: 20, right_wrist: 21, jaw: 22,
  left_eye: 23, right_eye: 24 };
/** SMPL-X kintree for joints 0-24 (hands 25-54 hang off the wrists). Take the real one from the model file. */
export const PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 15, 15, 15];
/** What goes on the wire, in this order: 13 joints x rotation vector (centiradians). */
export const SENT = [3, 6, 9, 12, 15, 13, 14, 16, 17, 18, 19, 20, 21];

const USED = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20];

/** One Euro filter (Casiez et al., CHI 2012): little lag when moving fast, little jitter when still. */
export class OneEuro {
  constructor(minCutoff = 1.2, beta = 0.6, dCutoff = 1.0) { Object.assign(this, { minCutoff, beta, dCutoff, x: null, dx: 0, t: 0 }); }
  static alpha(dt, cutoff) { const r = 2 * Math.PI * cutoff * dt; return r / (r + 1); }
  filter(x, t) {
    if (this.x === null) { this.x = x; this.t = t; return x; }
    const dt = Math.max(1e-3, t - this.t);
    const dx = (x - this.x) / dt;
    this.dx += OneEuro.alpha(dt, this.dCutoff) * (dx - this.dx);
    this.x += OneEuro.alpha(dt, this.minCutoff + this.beta * Math.abs(this.dx)) * (x - this.x);
    this.t = t;
    return this.x;
  }
}

/** Confidence of one landmark: visibility ramp times an in-frame ramp. Out-of-frame joints come back
 *  extrapolated WITH high visibility (measured: wrists 25 % outside the frame at v 0.97), so the
 *  normalized image position has to be checked too. */
export function landmarkConf(vis, nx, ny) {
  const m = Math.min(nx, 1 - nx, ny, 1 - ny);               // >0 inside the image
  const inFrame = m >= 0.02 ? 1 : m >= -0.15 ? 0.5 * (m + 0.15) / 0.17 : 0;
  return smoothstep(0.5, 0.85, vis) * inFrame;
}

export const DEFAULTS = {
  restDir: null,            // fn(jointIndex) -> unit rest bone direction in B (from the avatar's zero-pose joints); null: ideal T-pose
  level: true,              // auto-level camera pitch/roll from the long-run neck and shoulder line
  neckPitch0: 0.28,         // forward tilt (rad) of ear-mid minus shoulder-mid, upright (measured 0.22-0.39); calibrate at join
  headPitch0: 0.30,         // nose below the ear line (rad), level head (measured 0.12-0.47, noisy); calibrate at join
  spineSplit: [0.3, 0.3, 0.4],
  neckSplit: 0.4,
  maxTorsoYaw: 0.9, maxTorsoRoll: 0.5, maxHead: 1.3, maxElbow: 2.5, maxWrist: 1.0,
  restArmDown: 1.2,         // rad below the T-pose for an arm we cannot see
  restElbow: 0.25,
};

export class UpperBodySolver {
  constructor(opts = {}) {
    this.o = { ...DEFAULTS, ...opts };
    this.f = new Map(USED.map((i) => [i, [new OneEuro(), new OneEuro(), new OneEuro()]]));
    this.conf = new Map(USED.map((i) => [i, 0]));
    this.lvl = { pitch: null, roll: null };
    this.twist = { 1: 0, [-1]: 0 };
    this.head = QI();
  }

  restDir(j, side) {
    if (this.o.restDir) return this.o.restDir(j);
    return [side, 0, 0];                                   // arms along +-x in the T-pose
  }

  /** world: 33 {x,y,z,visibility} (poseWorldLandmarks[0]); image: 33 normalized landmarks; t: seconds.
   *  Returns { local: {jointIndex: quat}, conf: {torso, head, left, right} } */
  solve(world, image, t, dt = 1 / 15) {
    const o = this.o, P = {}, C = {};
    for (const i of USED) {
      const w = world[i], f = this.f.get(i);
      P[i] = [f[0].filter(w.x, t), f[1].filter(-w.y, t), f[2].filter(-w.z, t)];      // MediaPipe world -> B
      const c = landmarkConf(w.visibility ?? 1, image[i].x, image[i].y);
      const k = 1 - Math.exp(-dt / 0.15);                    // confidence eased, no flicker at the gate
      C[i] = this.conf.get(i) + k * (c - this.conf.get(i)); this.conf.set(i, C[i]);
    }
    // 1. Level: a laptop camera looks up at you; a level, upright person is the long-run average.
    let L = QI();
    const shL = sub(P[MP.L_SH], P[MP.R_SH]), neck = sub(mid(P[MP.L_EAR], P[MP.R_EAR]), mid(P[MP.L_SH], P[MP.R_SH]));
    if (o.level) {
      const good = Math.min(C[MP.L_SH], C[MP.R_SH], C[MP.L_EAR], C[MP.R_EAR]) > 0.8;
      const pitch = Math.atan2(neck[2], neck[1]), roll = Math.atan2(shL[1], Math.hypot(shL[0], shL[2]));
      const k = 1 - Math.exp(-dt / 8);                      // ~8 s memory: a lean held longer is absorbed
      if (this.lvl.pitch === null) Object.assign(this.lvl, { pitch, roll });
      else if (good) { this.lvl.pitch += k * (pitch - this.lvl.pitch); this.lvl.roll += k * (roll - this.lvl.roll); }
      L = qMul(qAxisAngle(Z, -clamp(this.lvl.roll, -0.45, 0.45)), qAxisAngle(X, -clamp(this.lvl.pitch - o.neckPitch0, -0.45, 0.45)));
      for (const i of USED) P[i] = qRot(L, P[i]);
    }
    const local = {};
    // 2. Torso: yaw and roll from the shoulder line only. Hips are never used: with a webcam they are
    //    out of frame and MediaPipe's guess can sit inside the frame at visibility 0.99.
    const a = norm(sub(P[MP.L_SH], P[MP.R_SH]));
    const yaw = clamp(Math.atan2(-a[2], a[0]), -o.maxTorsoYaw, o.maxTorsoYaw);
    const roll = clamp(Math.asin(clamp(a[1], -1, 1)), -o.maxTorsoRoll, o.maxTorsoRoll);
    const cT = Math.min(C[MP.L_SH], C[MP.R_SH]);
    const torso = qSlerp(QI(), qMul(qAxisAngle(Y, yaw), qAxisAngle(Z, roll)), cT);
    [J.spine1, J.spine2, J.spine3].forEach((j, k) => { local[j] = qPow(torso, o.spineSplit[k]); });
    const G9 = torso;
    // 3. Head from ears + nose: x = left ear - right ear, z = nose direction made normal to x.
    const xh = norm(sub(P[MP.L_EAR], P[MP.R_EAR]));
    const fwd = sub(P[MP.NOSE], mid(P[MP.L_EAR], P[MP.R_EAR]));
    const zh = norm(sub(fwd, scale(xh, dot(fwd, xh))));
    const headMeas = qMul(qFromBasis([xh, cross(zh, xh), zh]), qAxisAngle(X, -o.headPitch0));
    const cH = C[MP.NOSE] * Math.min(C[MP.L_EAR], C[MP.R_EAR]);
    this.head = qSlerp(this.head, qClampAngle(qMul(qInv(G9), headMeas), o.maxHead), cH);   // held when unseen
    local[J.neck] = qPow(this.head, o.neckSplit);
    local[J.head] = qPow(this.head, 1 - o.neckSplit);
    // 4. Arms: swing to the upper-arm direction, twist so the elbow hinge matches the arm's bend plane.
    const conf = { torso: cT, head: cH };
    for (const side of [1, -1]) {
      const [iS, iE, iW, iI, iP] = side > 0 ? [11, 13, 15, 19, 17] : [12, 14, 16, 20, 18];
      const [jC, jS, jE, jW] = side > 0 ? [13, 16, 18, 20] : [14, 17, 19, 21];
      const cU = C[iS] * C[iE], cF = cU * C[iW], cHand = cF * Math.min(C[iI], C[iP]) * 0.8;
      const d = norm(sub(P[iE], P[iS])), f = norm(sub(P[iW], P[iE]));
      const t = this.restDir(jS, side), te = this.restDir(jE, side), tw = this.restDir(jW, side);
      const h = norm(cross(t, Z));                         // elbow hinge in the T-pose: flexion swings the forearm forward
      // Collar follows the arm a little (clavicle rhythm): up with elevation, forward with reach.
      const dT = qRot(qInv(G9), d);
      const elev = Math.asin(clamp(dT[1], -1, 1)), reach = Math.max(0, Math.asin(clamp(dT[2], -1, 1)));
      const gam = elev > 0 ? 0.3 * elev : 0.1 * elev;
      const collarMeas = qMul(qAxisAngle(Y, -side * 0.15 * reach), qAxisAngle(Z, side * gam));
      const collarRest = qAxisAngle(Z, side * 0.1 * -o.restArmDown);
      local[jC] = qSlerp(collarRest, collarMeas, cU);
      const Pc = qMul(G9, local[jC]);
      // Shoulder.
      const dl = qRot(qInv(Pc), d), fl = qRot(qInv(Pc), f);
      const s = qFromTo(t, dl);
      const hs = qRot(s, h), n = cross(dl, fl), nl = len(n);
      const wt = smoothstep(0.15, 0.45, nl) * cF;          // a straight arm has no bend plane: keep the last twist
      const tauMeas = nl > 1e-6 ? Math.atan2(dot(dl, cross(hs, scale(n, 1 / nl))), dot(hs, scale(n, 1 / nl))) : 0;
      const tw0 = wrapPi(this.twist[side] + wt * wrapPi(tauMeas - this.twist[side]));
      this.twist[side] = tw0 * (1 - 0.3 * dt * (1 - wt));   // unseen bend plane: drift back to the plain swing
      const shMeas = qMul(qAxisAngle(dl, this.twist[side]), s);
      const shRest = qAxisAngle(Z, -side * o.restArmDown);
      local[jS] = qSlerp(shRest, shMeas, cU);
      // Elbow: a hinge about h, measured in the shoulder's frame.
      const Gs = qMul(Pc, local[jS]);
      const fl2 = qRot(qInv(Gs), f);
      const alpha = clamp(Math.atan2(dot(cross(te, fl2), h), dot(te, fl2)), 0, o.maxElbow);
      local[jE] = qAxisAngle(h, o.restElbow + (alpha - o.restElbow) * cF);
      // Wrist from the pose model's coarse hand points: direction to the knuckles, back-of-hand normal.
      const Ge = qMul(Gs, local[jE]);
      const W = P[iW], u = sub(mid(P[iI], P[iP]), W), b = scale(cross(sub(P[iI], W), sub(P[iP], W)), side);
      let lw = qClampAngle(qMul(qInv(Ge), qMul(qFromBasis(basis(u, b)), qInv(qFromBasis(basis(tw, Y))))), o.maxWrist);
      lw = qSlerp(QI(), lw, cHand);
      // Half the forearm roll goes to the elbow, so the skin twists along the forearm, not at the wrist.
      const roll2 = twistAngle(lw, tw) / 2;
      local[jE] = qMul(local[jE], qAxisAngle(te, roll2));
      local[jW] = qMul(qAxisAngle(tw, -roll2), lw);
      conf[side > 0 ? 'left' : 'right'] = { upper: cU, fore: cF, hand: cHand };
    }
    return { local, conf, level: L };
  }
}

/** Wire format: rotation vectors of SENT joints in centiradians, ints (~170 bytes as JSON). */
export function packJoints(local) {
  const out = [];
  for (const j of SENT) for (const c of qToRotvec(local[j] || QI())) out.push(Math.round(c * 100));
  return out;
}
export function unpackJoints(arr) {
  if (!Array.isArray(arr) || arr.length !== SENT.length * 3 || !arr.every((v) => Number.isFinite(v) && Math.abs(v) <= 400)) return null;
  const local = {};
  SENT.forEach((j, k) => {
    const r = [arr[3 * k] / 100, arr[3 * k + 1] / 100, arr[3 * k + 2] / 100];
    const a = Math.hypot(...r);
    local[j] = a < 1e-9 ? QI() : qAxisAngle(r, a);
  });
  return local;
}

// ---- the receiver's procedural legs ---------------------------------------------------------
// Run in every RECEIVER's browser: nothing about legs goes on the wire. Input each frame: where
// the avatar stands (floor point, three.js world, Y up) and its facing yaw. Output: pelvis world
// pose + SMPL-X local rotations for hips, knees, ankles (+ a spine1 counter-rotation so the chest
// does not wobble with the hips). Feet are planted in WORLD space, so a stance foot cannot slide
// whatever the avatar does; the step phase advances with distance walked, so cadence follows
// speed and stops with the avatar; analytic two-bone IK puts each ankle on its target with the
// knee toward the facing direction, and the pelvis drops just enough that a planted foot stays
// reachable (the gait's bob, for free).

const frac = (x) => x - Math.floor(x);

export class Gait {
  /** J0: zero-pose joints of the avatar (SMPL-X indices, metres, Y up), from the build function.
   *  ankleHeight: sole-to-ankle-joint height (J0[7].y minus the mesh/splat min y). */
  constructor({ J0, ankleHeight = 0.08 }) {
    this.J0 = J0;
    this.L1 = len(sub(J0[4], J0[1])); this.L2 = len(sub(J0[7], J0[4]));
    this.leg = this.L1 + this.L2;
    this.ankleH = ankleHeight;
    this.pelvisH = J0[0][1] - (J0[7][1] - ankleHeight);   // standing pelvis height over the floor
    this.hipH = this.pelvisH - (J0[0][1] - J0[1][1]);       // and the hip joints'
    this.hipHalf = Math.abs(J0[1][0] - J0[2][0]) / 2;
    this.restThigh = [norm(sub(J0[4], J0[1])), norm(sub(J0[5], J0[2]))];
    this.restShin = [norm(sub(J0[7], J0[4])), norm(sub(J0[8], J0[5]))];
    this.reset(null, 0);
  }

  reset(root, yaw) {
    this.root = root; this.vel = [0, 0, 0]; this.speed = 0; this.phase = 0; this.drop = 0;
    this.feet = [0, 1].map((i) => ({ pos: root ? this.home(root, yaw, i) : null, swing: null }));
  }

  /** Where foot i (0 left, 1 right) rests under its hip. Body +x is the person's left. */
  home(root, yaw, i) {
    const side = i === 0 ? 1 : -1;
    return add(root, qRot(qAxisAngle(Y, yaw), [side * this.hipHalf * 0.95, 0, 0]));
  }

  /** dt s, root = floor point under the pelvis (three world), yaw = rotation about +Y of the body's +z. */
  update(dt, root, yaw, t) {
    if (!this.root || len(sub(root, this.root)) > 1.5) this.reset(root, yaw);          // first frame / teleport / room swap
    const step = sub(root, this.root); step[1] = 0;
    const v = scale(step, 1 / Math.max(dt, 1e-3));
    this.vel = lerp3(this.vel, v, 1 - Math.exp(-dt / 0.15));
    this.speed = Math.hypot(this.vel[0], this.vel[2]);
    this.root = root;
    const g = smoothstep(0.08, 0.35, this.speed);                                     // 0 idle .. 1 walking
    const moving = len(step) / Math.max(dt, 1e-3) > 0.1;   // stopped NOW (the EMA'd speed lags ~0.5 s)
    const duty = 0.62 - 0.12 * smoothstep(1.6, 2.8, this.speed);                      // shorter stance when hurrying
    // How far behind the hip an ankle can be with the hips 4 cm down and the heel 10 cm up; a foot
    // lifts off 58 % of a stance behind the hip, which caps the stride (and so the hips' bob).
    const reach = Math.sqrt(Math.max(0, this.leg ** 2 - (this.hipH - 0.04 - this.ankleH - 0.10) ** 2));
    const stride = Math.min(2 * (0.28 + 0.27 * this.speed), 0.95 * reach / (0.58 * duty));   // cycle length: 2 steps
    const walking = g > 0.05 && moving;
    // Starting to walk: both feet are under the hips, which is mid-stance for one and mid-swing for
    // the other. Starting at phase 0 instead leaves a stance foot planted 0.5 stride too long.
    if (walking && !this.walking) this.phase = duty / 2;
    this.walking = walking;
    const dir = this.speed > 1e-3 ? norm([this.vel[0], 0, this.vel[2]]) : [0, 0, 0];
    const fwd = qRot(qAxisAngle(Y, yaw), [0, 0, 1]);
    // Cadence catch-up: a stance foot further behind than half a stance (after a turn, a speed-up,
    // network lag) makes the phase run faster until it lifts off. No jumps, just quicker steps.
    let extra = 0;
    for (const i of [0, 1]) {
      const f = this.feet[i];
      if (walking && !f.swing) extra = Math.max(extra, -dot(sub(f.pos, this.home(root, yaw, i)), dir) - 0.58 * duty * stride);
    }
    this.phase += len(step) / stride + (walking ? 10 * clamp(extra, 0, 0.3) * dt : 0);
    // Feet.
    const order = len(sub(this.feet[0].pos, this.home(root, yaw, 0))) >= len(sub(this.feet[1].pos, this.home(root, yaw, 1))) ? [0, 1] : [1, 0];
    for (const i of order) {
      const f = this.feet[i], home = this.home(root, yaw, i);
      if (walking) {
        const p = frac(this.phase + 0.5 * i);
        if (p >= duty) {                                   // swing: lift-off point -> predicted landing
          const u = (p - duty) / (1 - duty);
          if (!f.swing || f.swing.idle) f.swing = { from: f.pos, idle: false, u0: Math.min(u, 0.9), heel0: f.heel || 0 };   // may start mid-swing
          const ahead = (1 - u) * (1 - duty) * stride + 0.42 * duty * stride;           // body travel to landing + 42 % of a stance
          const land = add(home, scale(dir, ahead));
          const su = clamp((u - f.swing.u0) / (1 - f.swing.u0), 0, 1);
          f.pos = lerp3(f.swing.from, land, smoothstep(0, 1, su));
          f.lift = (0.05 + 0.03 * g) * Math.sin(Math.PI * su);
          f.u = su;
          f.heel = f.swing.heel0 * (1 - smoothstep(0, 0.3, su));
        } else { f.swing = null; f.lift = 0; f.u = -1; }    // stance: planted where it landed
      } else {                                              // idle: step a foot back under its hip, one at a time
        if (f.swing && !f.swing.idle) f.swing = { from: f.pos, idle: true, t: 0, heel0: f.heel || 0 };   // stopped mid-step: finish it home
        const off = len(sub(f.pos, home)), other = this.feet[1 - i];
        if (!f.swing && !other.swing && (off > 0.16 || (off > 0.06 && this.idleFor > 0.6))) {
          f.swing = { from: f.pos, idle: true, t: 0, heel0: f.heel || 0, dur: clamp(0.22 + 0.35 * off, 0.25, 0.45) };
        }
        if (f.swing) {
          f.swing.t += dt / (f.swing.dur || 0.3);
          const u = Math.min(1, f.swing.t);
          f.pos = lerp3(f.swing.from, home, smoothstep(0, 1, u));
          f.lift = 0.05 * Math.sin(Math.PI * u); f.u = u;
          f.heel = f.swing.heel0 * (1 - smoothstep(0, 0.3, u));
          if (u >= 1) { f.swing = null; f.lift = 0; f.u = -1; }
        } else { f.lift = 0; f.u = -1; }
      }
      // A planted foot BEHIND its hip rolls onto its toes (heel-off): the ankle rises up to 9 cm,
      // which is what lets a real stride reach without the hips sinking. One ahead lands on its heel.
      if (!f.swing) {
        const back = -dot(sub(f.pos, home), walking ? dir : fwd);
        const heel = 0.10 * smoothstep(0.10, 0.40, back) + 0.02 * smoothstep(0.15, 0.35, -back);
        f.heel = (f.heel || 0) + (heel - (f.heel || 0)) * (1 - Math.exp(-dt / 0.08));        // eased: a planted ankle never jumps
        f.toesUp = smoothstep(0.15, 0.35, -back);
      } else f.toesUp = 0;
      // Stretch guard: a planted foot can never be left behind out of reach (teleport-ish moves).
      if (!f.swing && len(sub(f.pos, home)) > 0.9 * reach + 0.5 * stride) f.pos = home;
    }
    this.idleFor = walking ? 0 : (this.idleFor || 0) + dt;
    // Pelvis: sway toward the stance foot, idle weight shift, and a height every foot can reach.
    const side = qRot(qAxisAngle(Y, yaw), [1, 0, 0]);
    const swayW = 0.02 * g * Math.sin(2 * Math.PI * (this.phase - 0.06));
    const swayI = (1 - g) * (0.012 * Math.sin(2 * Math.PI * t / 4.7) + 0.006 * Math.sin(2 * Math.PI * t / 2.3 + 1));
    let pelvis = add(root, scale(side, swayW + swayI));
    let y = this.pelvisH - 0.015 - 0.02 * g;                                          // soft knees
    const hipW = (i, py) => add([pelvis[0], py, pelvis[2]], qRot(qAxisAngle(Y, yaw), sub(this.J0[i === 0 ? 1 : 2], this.J0[0])));
    for (const i of [0, 1]) {
      const a = this.ankleTarget(i, yaw, fwd);
      const h = hipW(i, y), dx = Math.hypot(a[0] - h[0], a[2] - h[2]);
      const need = a[1] + Math.sqrt(Math.max(0, (0.995 * this.leg) ** 2 - dx * dx)) - (h[1] - y);
      y = Math.min(y, need);
    }
    // Down at once (a foot must stay reachable), back up smoothly.
    const want = this.pelvisH - y;
    this.drop = want > this.drop ? want : this.drop + (want - this.drop) * (1 - Math.exp(-dt / 0.12));
    pelvis = [pelvis[0], root[1] + this.pelvisH - this.drop, pelvis[2]];
    const pelvisYaw = 0.06 * g * Math.sin(2 * Math.PI * (this.phase - 0.06));
    const Gp = qMul(qAxisAngle(Y, yaw + pelvisYaw), qAxisAngle([0, 0, 1], -2.5 * swayW));
    // Legs: two-bone IK, knee toward the facing direction.
    const local = { 0: Gp };
    for (const i of [0, 1]) {
      const [jH, jK, jA] = i === 0 ? [1, 4, 7] : [2, 5, 8];
      const H = add(pelvis, qRot(Gp, sub(this.J0[jH], this.J0[0])));
      const A = this.ankleTarget(i, yaw, fwd);
      const D = sub(A, H), d = clamp(len(D), 0.05, 0.999 * this.leg), Dn = norm(D);
      const cosA = clamp((this.L1 ** 2 + d * d - this.L2 ** 2) / (2 * this.L1 * d), -1, 1);
      const pole = norm(add(fwd, scale(qRot(qAxisAngle(Y, yaw), [i === 0 ? 1 : -1, 0, 0]), 0.1)));
      const pp = norm(sub(pole, scale(Dn, dot(pole, Dn))));
      const thigh = add(scale(Dn, cosA), scale(pp, Math.sqrt(1 - cosA * cosA)));
      const knee = add(H, scale(thigh, this.L1));
      const shin = norm(sub(add(H, scale(Dn, d)), knee));
      // Hip local: swing rest thigh onto the thigh, twist so the knee hinge (+x at rest) is normal to the leg plane.
      const t = this.restThigh[i], hRest = [1, 0, 0];
      const dl = qRot(qInv(Gp), thigh), nl = norm(qRot(qInv(Gp), cross(pp, thigh)));
      const s = qFromTo(t, dl), hs = qRot(s, hRest);
      const tau = Math.atan2(dot(dl, cross(hs, nl)), dot(hs, nl));
      local[jH] = qMul(qAxisAngle(dl, tau), s);
      const Gh = qMul(Gp, local[jH]);
      const sl = qRot(qInv(Gh), shin), tk = this.restShin[i];
      // Knee: the hinge angle, plus the tiny off-hinge swing that the zero pose's own knee/ankle
      // offsets need for the ankle to land exactly (else a planted foot floats ~1 cm).
      local[jK] = qClampAngle(qFromTo(tk, sl), 2.4);
      // Ankle: foot flat (slight toe-out) in stance; toes down at lift-off, up before heel strike.
      const f = this.feet[i];
      const toesDown = f.toesUp > 0 ? 0 : Math.asin(clamp((f.heel || 0) / 0.14, 0, 0.9));          // heel-off about the toes (0.14 m ahead)
      const pitch = toesDown - 0.25 * Math.max(f.toesUp || 0, f.u > 0.75 ? (f.u - 0.75) / 0.25 : 0);  // toes up for heel strike
      const Gfoot = qMul(qAxisAngle(Y, yaw - (i === 0 ? -1 : 1) * 0.1), qAxisAngle([1, 0, 0], pitch));
      local[jA] = qMul(qInv(qMul(Gh, local[jK])), Gfoot);
      local[jA === 7 ? 10 : 11] = QI();
    }
    // Keep the chest pointing where the body faces: undo the pelvis' own sway in spine1.
    local.spineCounter = qMul(qInv(Gp), qAxisAngle(Y, yaw));
    return { pelvis, local, g, speed: this.speed, phase: this.phase };
  }

  ankleTarget(i, yaw, fwd) {
    const f = this.feet[i];
    return add(f.pos, [0, this.ankleH + (f.lift || 0) + (f.heel || 0), 0]);
  }
}

// ---- putting a frame together (receiver) ----------------------------------------------------
/** Local rotations while a sender is not tracked (no "j"): arms relaxed down, elbows a little
 *  bent, the head following the camera pitch (> 0 looks up). The spine stays at rest. */
export function relaxedUpperBody(pitch = 0) {
  const p = Number.isFinite(pitch) ? pitch : 0;
  return {
    3: QI(), 6: QI(), 9: QI(),
    12: qAxisAngle(X, -0.4 * p), 15: qAxisAngle(X, -0.6 * p),
    13: qAxisAngle(Z, -0.12), 14: qAxisAngle(Z, 0.12),
    16: qAxisAngle(Z, -1.2), 17: qAxisAngle(Z, 1.2),
    18: qAxisAngle([0, -1, 0], 0.25), 19: qAxisAngle([0, 1, 0], 0.25),
    20: QI(), 21: QI(),
  };
}

export const LEG_JOINTS = [1, 2, 4, 5, 7, 8, 10, 11];

/** One frame's local rotations for `poseSkeleton`: the pelvis' world rotation and the legs from
 *  the gait, the sent (or relaxed) upper body, Spine_1 also carrying the gait's counter-rotation,
 *  everything else (jaw, eyes, hands) at rest. `J` joints. */
export function assembleLocal(gaitLocal, upper, J) {
  const local = new Array(J);
  for (let j = 0; j < J; j++) local[j] = null;
  local[0] = gaitLocal[0];
  for (const j of LEG_JOINTS) if (j < J) local[j] = gaitLocal[j] || null;
  for (const j of SENT) if (j < J) local[j] = upper[j] || null;
  if (gaitLocal.spineCounter) local[3] = qMul(gaitLocal.spineCounter, upper[3] || QI());
  return local;
}
