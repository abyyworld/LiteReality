"""SPZ files: Niantic's compressed Gaussian splat format, which Spark 2.3.1 reads natively.

Written as version 3 (one gzip stream, the 16-byte header every SPZ reader since 2024 accepts),
following the reference packer (github.com/nianticlabs/spz, `src/cc/load-spz.cc`):

    header   <u4 magic 0x5053474e ("NGSP"), <u4 version 3, <u4 N, u1 sh degree,
             u1 fractional bits (12), u1 flags (1 = antialiased), u1 0
    then     positions  24-bit signed fixed point x, y, z (N * 9 bytes, little-endian)
             alphas     sigmoid(opacity) * 255                     (N bytes)
             colours    SH DC * (0.15 * 255) + 127.5, per channel  (N * 3)
             scales     (ln sigma + 10) * 16, per axis             (N * 3)
             rotations  "smallest three": index of the largest component in the top 2 bits,
                        then 3 x (sign bit + 9-bit magnitude / sqrt(1/2))   (N * 4, little-endian)
             sh         higher SH coefficients, 8-bit, coefficient-major then channel (N * K * 3)

20 bytes a splat before gzip at SH degree 0 (29 at degree 1), against 56 (236 at degree 3) in a
float PLY; gzip takes it to ~10 bytes on a real room (tea_room: 640k splats, 6.2 MB). Coordinates are stored as given: callers write the room's own
frame (MuJoCo, Z up) and the browser places the mesh in its Z-up world group.

Pure numpy + stdlib, Python 3.10 compatible: the Modal GPU trainer imports this module too.
"""

from __future__ import annotations

import gzip
import struct
from dataclasses import dataclass

import numpy as np

MAGIC = 0x5053474E
VERSION = 3
FRACTIONAL_BITS = 12
FLAG_ANTIALIASED = 0x1
COLOR_SCALE = 0.15
SH_C0 = 0.28209479177387814
_SQRT1_2 = float(np.sqrt(0.5))
# 24-bit fixed point with 12 fractional bits: +-2048 m, a quarter of a millimetre.
_POS_LIMIT = (1 << 23) - 1
MAX_SPLATS = 20_000_000


def sh_dim(degree: int) -> int:
    """Higher-order SH coefficients per channel (degree 0 has none)."""
    return (degree + 1) ** 2 - 1


@dataclass
class Splats:
    """Gaussians in plain linear units, the form every step here works in.

    positions  (N, 3) metres
    scales     (N, 3) standard deviations along the splat's own axes, metres
    rotations  (N, 4) unit quaternions x, y, z, w
    colors     (N, 3) the colour as displayed, 0..1 (SH DC as RGB: 0.5 + SH_C0 * dc)
    opacities  (N,)   0..1
    sh         (N, K, 3) higher SH coefficients, or None for degree 0
    """

    positions: np.ndarray
    scales: np.ndarray
    rotations: np.ndarray
    colors: np.ndarray
    opacities: np.ndarray
    sh: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.positions.shape[0])

    @property
    def sh_degree(self) -> int:
        if self.sh is None or self.sh.shape[1] == 0:
            return 0
        k = self.sh.shape[1]
        for degree in (1, 2, 3):
            if sh_dim(degree) == k:
                return degree
        raise ValueError(f"{k} SH coefficients per channel is no SH degree")

    def subset(self, index) -> "Splats":
        return Splats(self.positions[index], self.scales[index], self.rotations[index],
                      self.colors[index], self.opacities[index],
                      None if self.sh is None else self.sh[index])

    def transformed(self, rot: np.ndarray, pos: np.ndarray) -> "Splats":
        """The same splats under the rigid motion x -> rot @ x + pos, their higher SH turned with
        them (`rotate_sh`), so each splat shows the same colour from the same side as before."""
        rot = np.asarray(rot, dtype=np.float64)
        q = matrix_to_quat(rot)
        return Splats((self.positions @ rot.T + pos).astype(np.float32), self.scales.copy(),
                      quat_multiply(np.broadcast_to(q, self.rotations.shape), self.rotations),
                      self.colors.copy(), self.opacities.copy(),
                      None if self.sh is None else rotate_sh(self.sh, rot))

    @staticmethod
    def concat(parts: list["Splats"]) -> "Splats":
        parts = [p for p in parts if len(p)]
        if not parts:
            return empty()
        degree = max(p.sh_degree for p in parts)
        k = sh_dim(degree)

        def sh_of(p: Splats):
            out = np.zeros((len(p), k, 3), np.float32)
            if p.sh is not None and p.sh.shape[1]:
                out[:, :p.sh.shape[1]] = p.sh
            return out

        return Splats(np.concatenate([p.positions for p in parts]),
                      np.concatenate([p.scales for p in parts]),
                      np.concatenate([p.rotations for p in parts]),
                      np.concatenate([p.colors for p in parts]),
                      np.concatenate([p.opacities for p in parts]),
                      np.concatenate([sh_of(p) for p in parts]) if degree else None)


def empty() -> Splats:
    z3 = np.zeros((0, 3), np.float32)
    return Splats(z3, z3.copy(), np.zeros((0, 4), np.float32), z3.copy(), np.zeros(0, np.float32))


# ── spherical harmonics (3DGS's real basis and order, which gsplat trains and Spark draws) ─────

SH_C1 = 0.4886025119029199
SH_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792,
         0.5462742152960396)
SH_C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
         -0.4570457994644658, 1.445305721320277, -0.5900435899266435)


def sh_basis(dirs: np.ndarray, degree: int = 3) -> np.ndarray:
    """The higher SH basis functions (degrees 1..`degree`, no DC) at unit directions `dirs`
    (M x 3): M x sh_dim(degree), in the coefficient order SPZ and 3DGS PLYs store."""
    d = np.asarray(dirs, np.float64)
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    out = [-SH_C1 * y, SH_C1 * z, -SH_C1 * x]
    if degree >= 2:
        xx, yy, zz = x * x, y * y, z * z
        out += [SH_C2[0] * x * y, SH_C2[1] * y * z, SH_C2[2] * (2 * zz - xx - yy),
                SH_C2[3] * x * z, SH_C2[4] * (xx - yy)]
    if degree >= 3:
        out += [SH_C3[0] * y * (3 * xx - yy), SH_C3[1] * x * y * z,
                SH_C3[2] * y * (4 * zz - xx - yy), SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy),
                SH_C3[4] * x * (4 * zz - xx - yy), SH_C3[5] * z * (xx - yy),
                SH_C3[6] * x * (xx - 3 * yy)]
    return np.stack(out[:sh_dim(degree)], axis=1)


def _fibonacci_dirs(count: int) -> np.ndarray:
    k = np.arange(count) + 0.5
    z = 1.0 - 2.0 * k / count
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = np.pi * (3.0 - np.sqrt(5.0)) * k
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def sh_rotation(rot: np.ndarray, degree: int) -> np.ndarray:
    """The sh_dim(degree) x sh_dim(degree) matrix D that turns higher SH coefficients with a
    rotation: coefficients c of a colour f(d) become D @ c, those of f(rot.T @ d) (the colour
    carried along when the splat turns by `rot`). Each band rotates within itself, so D is fitted
    exactly by least squares on well-spread directions."""
    rot = np.asarray(rot, np.float64)
    dirs = _fibonacci_dirs(64)
    a = sh_basis(dirs, degree)                     # Y(d)
    b = sh_basis(dirs @ rot, degree)               # Y(rot.T d), row by row
    d = np.zeros((a.shape[1], a.shape[1]))
    for lo in range(1, degree + 1):                # band by band (block diagonal)
        i, j = sh_dim(lo - 1), sh_dim(lo)
        d[i:j, i:j] = np.linalg.lstsq(a[:, i:j], b[:, i:j], rcond=None)[0]
    return d


def rotate_sh(sh: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """Higher SH coefficients (N x K x 3) of splats turned by the rotation `rot` (`sh_rotation`):
    each keeps showing the colour it showed from each side. float32, like the input."""
    sh = np.asarray(sh)
    if sh.ndim != 3 or sh.shape[1] == 0:
        return sh.copy()
    degree = next(d for d in (1, 2, 3) if sh_dim(d) == sh.shape[1])
    d = sh_rotation(rot, degree)
    return np.einsum("kj,njc->nkc", d, sh.astype(np.float64)).astype(np.float32)


# ── quaternions (x, y, z, w) ───────────────────────────────────────────────────────────────────


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = np.moveaxis(np.asarray(a, np.float64), -1, 0)
    bx, by, bz, bw = np.moveaxis(np.asarray(b, np.float64), -1, 0)
    return np.stack([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz], axis=-1).astype(np.float32)


def matrix_to_quat(m: np.ndarray) -> np.ndarray:
    """One rotation matrix -> unit quaternion x, y, z, w."""
    m = np.asarray(m, dtype=np.float64)
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    q = np.asarray(q)
    return (q / np.linalg.norm(q)).astype(np.float32)


def quat_to_matrices(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


# ── writing ────────────────────────────────────────────────────────────────────────────────────


def _u8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def _pack_rotations(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    largest = np.argmax(np.abs(q), axis=1)
    negate = q[np.arange(len(q)), largest] < 0
    comp = largest.astype(np.uint32)
    for i in range(4):
        keep = largest != i
        neg = ((q[:, i] < 0) ^ negate).astype(np.uint32)
        mag = (511.0 * (np.abs(q[:, i]) / _SQRT1_2) + 0.5).astype(np.uint32)
        comp = np.where(keep, (comp << 10) | (neg << 9) | np.minimum(mag, 511), comp)
    return comp.astype("<u4").view(np.uint8).reshape(-1, 4)


def _quantize_sh(x: np.ndarray, bucket: int) -> np.ndarray:
    q = np.round(x * 128.0).astype(np.int32) + 128
    q = (q + bucket // 2) // bucket * bucket
    return np.clip(q, 0, 255).astype(np.uint8)


def encode(splats: Splats, *, antialiased: bool = True, level: int = 6) -> bytes:
    """`splats` as an SPZ file (bytes). Non-finite splats are dropped."""
    n0 = len(splats)
    if n0 > MAX_SPLATS:
        raise ValueError(f"{n0} splats is more than {MAX_SPLATS}")
    ok = (np.isfinite(splats.positions).all(1) & np.isfinite(splats.scales).all(1)
          & np.isfinite(splats.rotations).all(1) & np.isfinite(splats.colors).all(1)
          & np.isfinite(splats.opacities))
    s = splats.subset(ok) if not ok.all() else splats
    n, degree = len(s), s.sh_degree
    fixed = np.round(np.asarray(s.positions, np.float64) * (1 << FRACTIONAL_BITS)).astype(np.int64)
    if n and np.abs(fixed).max() > _POS_LIMIT:
        raise ValueError("a splat lies more than 2 km from the origin")
    f32 = (fixed & 0xFFFFFF).astype("<u4").view(np.uint8).reshape(n, 3, 4)[:, :, :3]
    dc = (np.asarray(s.colors, np.float64) - 0.5) / SH_C0
    parts = [
        struct.pack("<IIIBBBB", MAGIC, VERSION, n, degree, FRACTIONAL_BITS,
                    FLAG_ANTIALIASED if antialiased else 0, 0),
        np.ascontiguousarray(f32).tobytes(),
        _u8(np.asarray(s.opacities, np.float64) * 255.0).tobytes(),
        _u8(dc * (COLOR_SCALE * 255.0) + 127.5).tobytes(),
        _u8((np.log(np.maximum(np.asarray(s.scales, np.float64), 1e-12)) + 10.0) * 16.0).tobytes(),
        _pack_rotations(s.rotations).tobytes(),
    ]
    if degree:
        sh = np.asarray(s.sh, np.float64)                      # (N, K, 3): coefficient, channel
        out = np.empty(sh.shape, np.uint8)
        out[:, :3] = _quantize_sh(sh[:, :3], 1 << (8 - 5))     # degree 1 at 5 bits
        out[:, 3:] = _quantize_sh(sh[:, 3:], 1 << (8 - 4))     # the rest at 4 bits
        parts.append(out.tobytes())
    return gzip.compress(b"".join(parts), compresslevel=level, mtime=0)


# ── reading (tests, and splitting a trained room) ──────────────────────────────────────────────


def _unpack_rotations(r: np.ndarray) -> np.ndarray:
    comp = r.astype(np.uint8).reshape(-1, 4).copy().view("<u4")[:, 0].astype(np.uint64)
    largest = (comp >> 30).astype(np.int64)
    q = np.zeros((len(comp), 4), np.float64)
    rest = comp.copy()
    for i in (3, 2, 1, 0):
        is_largest = largest == i
        mag = (rest & 511).astype(np.float64) / 511.0 * _SQRT1_2
        neg = ((rest >> 9) & 1).astype(bool)
        val = np.where(neg, -mag, mag)
        q[:, i] = np.where(is_largest, 0.0, val)
        rest = np.where(is_largest, rest, rest >> 10)
    sq = np.sum(q * q, axis=1)
    q[np.arange(len(q)), largest] = np.sqrt(np.maximum(0.0, 1.0 - sq))
    return q.astype(np.float32)


def decode(data: bytes) -> Splats:
    """An SPZ file (version 2 or 3, gzip) back into `Splats`."""
    raw = gzip.decompress(data)
    if len(raw) < 16:
        raise ValueError("not an SPZ file: too short")
    magic, version, n, degree, bits, _flags, _ = struct.unpack_from("<IIIBBBB", raw)
    if magic != MAGIC or version not in (2, 3):
        raise ValueError(f"not an SPZ v2/v3 file (magic {magic:#x}, version {version})")
    if n > MAX_SPLATS or degree > 3:
        raise ValueError("SPZ header out of range")
    k = sh_dim(degree)
    rot_bytes = 4 if version >= 3 else 3
    need = 16 + n * (9 + 1 + 3 + 3 + rot_bytes + 3 * k)
    if len(raw) < need:
        raise ValueError("SPZ file is truncated")
    at = 16

    def take(count):
        nonlocal at
        out = np.frombuffer(raw, np.uint8, count, at)
        at += count
        return out

    p = take(n * 9).reshape(n, 3, 3).astype(np.int32)
    fixed = p[:, :, 0] | (p[:, :, 1] << 8) | (p[:, :, 2] << 16)
    fixed = np.where(fixed & 0x800000, fixed - (1 << 24), fixed)
    positions = (fixed / float(1 << bits)).astype(np.float32)
    opacities = (take(n) / 255.0).astype(np.float32)
    colors = ((take(n * 3).reshape(n, 3) / 255.0 - 0.5) / COLOR_SCALE * SH_C0 + 0.5)
    scales = np.exp(take(n * 3).reshape(n, 3) / 16.0 - 10.0).astype(np.float32)
    r = take(n * rot_bytes).reshape(n, rot_bytes)
    if version >= 3:
        rotations = _unpack_rotations(r)
    else:
        xyz = r / 127.5 - 1.0
        w = np.sqrt(np.maximum(0.0, 1.0 - np.sum(xyz * xyz, axis=1, keepdims=True)))
        rotations = np.concatenate([xyz, w], axis=1).astype(np.float32)
    sh = None
    if degree:
        sh = ((take(n * k * 3).reshape(n, k, 3).astype(np.float32) - 128.0) / 128.0)
    return Splats(positions, scales, rotations, colors.astype(np.float32), opacities, sh)
