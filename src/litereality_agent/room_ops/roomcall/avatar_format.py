"""The full-body avatar file the room call serves to browsers: one splat person, ready to skin.

One file per avatar ("LRAV", version 1), written by the GPU function (LHM++ on Modal) or by the
stand-in generator (`avatars.make_standin`), checked by the room server before it is served, and
read by the browser (`parseAvatar` in web/body.js) straight into Spark's packed splats and
`SplatSkinning` data with no per-splat maths:

    0   4  magic b"LRAV"
    4   4  uint32 LE  version (1)
    8   4  uint32 LE  header length H (a multiple of 16)
    12  4  uint32 LE  0 (reserved)
    16  H  header: UTF-8 JSON, right-padded with spaces
    16+H   data block: the arrays below, each starting at a multiple of 16 bytes from the block
           start (zero padding between), little-endian, row-major, in this order:

    positions  <f4 [N,3]  splat centres, metres
    scales     <f2 [N,3]  Gaussian standard deviations along the splat's own axes, metres (linear)
    rotations  <f2 [N,4]  unit quaternion x, y, z, w (three.js order)
    colors     u1  [N,4]  sRGB colour as displayed (the photo's own values) and opacity * 255
    skin       <u2 [N,4]  up to four (joint, weight) pairs as weight | joint << 8 — exactly Spark
                          2.3.1's SplatSkinning.skinData — heaviest first, weights summing to 255

The frame is the person's own CANONICAL frame, SMPL-X's: metres, +Y up, the person faces +Z, +X
is the person's LEFT, in the SMPL-X zero pose (T-pose, arms straight out, flat hands). The
skeleton travels in the header: joint names, parents and REST positions (each joint's position
in that pose; every joint's rest rotation is the identity). Joints 0..21 are always SMPL-X's body
joints in SMPL-X order, so a pose can be addressed by index whoever built the body; a file may
add jaw and eyes (22-24) and the hands (25-54). `floor_y` is where the soles are.

Splats are stored in a seeded random order, so any PREFIX of them is a uniform subsample: a
browser with less to spend draws the first n and drops the rest (level of detail for free).

Pure numpy + stdlib, Python 3.10 / numpy 1.23 compatible: the GPU container (LHM++ pins both)
imports this module too.
"""

from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass, field

import numpy as np

MAGIC = b"LRAV"
VERSION = 1
PREAMBLE = 16
ALIGN = 16
MAX_HEADER = 64 * 1024
MAX_SPLATS = 120_000
MAX_BYTES = 5 * 1024 * 1024
BODY_JOINTS = 22
MAX_JOINTS = 64
COORD_LIMIT = 3.0       # metres from the canonical origin
MAX_SCALE = 0.5         # metres; a bigger splat is not part of a person
MIN_HEIGHT, MAX_HEIGHT = 0.5, 2.6
# The joints whose splats are the head the live face replaces (Head, Jaw, L_Eye, R_Eye).
HEAD_JOINTS = (15, 22, 23, 24)
_KIND = re.compile(r"[a-z0-9_-]{1,16}")

SMPLX_JOINTS = (
    "Pelvis", "L_Hip", "R_Hip", "Spine_1", "L_Knee", "R_Knee", "Spine_2", "L_Ankle", "R_Ankle",
    "Spine_3", "L_Foot", "R_Foot", "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder",
    "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
    "Jaw", "L_Eye", "R_Eye",
    "L_Index_1", "L_Index_2", "L_Index_3", "L_Middle_1", "L_Middle_2", "L_Middle_3",
    "L_Pinky_1", "L_Pinky_2", "L_Pinky_3", "L_Ring_1", "L_Ring_2", "L_Ring_3",
    "L_Thumb_1", "L_Thumb_2", "L_Thumb_3",
    "R_Index_1", "R_Index_2", "R_Index_3", "R_Middle_1", "R_Middle_2", "R_Middle_3",
    "R_Pinky_1", "R_Pinky_2", "R_Pinky_3", "R_Ring_1", "R_Ring_2", "R_Ring_3",
    "R_Thumb_1", "R_Thumb_2", "R_Thumb_3",
)
SMPLX_PARENTS = (
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
    15, 15, 15,
    20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 20, 37, 38,
    21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52, 53,
)

# (name, dtype, columns), in file order.
ARRAYS = (
    ("positions", "<f4", 3),
    ("scales", "<f2", 3),
    ("rotations", "<f2", 4),
    ("colors", "|u1", 4),
    ("skin", "<u2", 4),
)


class AvatarFormatError(ValueError):
    """The bytes are not a valid avatar file (or an Avatar would not make one)."""


@dataclass
class Avatar:
    """One person, canonical frame (see the module docstring). Per-splat arrays share row order."""

    positions: np.ndarray        # (N, 3) float32, metres
    scales: np.ndarray           # (N, 3) float32, metres, linear
    rotations: np.ndarray        # (N, 4) float32, x y z w, unit
    colors: np.ndarray           # (N, 4) uint8, sRGB + opacity
    skin: np.ndarray             # (N, 4) uint16, weight | joint << 8 (`quantize_skin`)
    joint_names: list[str]
    parents: list[int]
    rest: np.ndarray             # (J, 3) float32, zero-pose joint positions
    floor_y: float
    kind: str = "standin"        # "lhm" | "standin"
    meta: dict = field(default_factory=dict)

    @property
    def count(self) -> int:
        return int(len(self.positions))

    def head_mask(self) -> np.ndarray:
        """Splats whose weight on HEAD_JOINTS is at least half: the head the live face replaces."""
        weights = (self.skin & 0xFF).astype(np.int32)
        joints = (self.skin >> 8).astype(np.int32)
        on_head = np.isin(joints, HEAD_JOINTS) & (weights > 0)
        return (weights * on_head).sum(axis=1) >= 128


def quantize_skin(joints: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Per-splat skinning weights -> the file's (N, 4) uint16 Spark encoding.

    `joints` (N, K) joint indices and `weights` (N, K) non-negative weights, any K >= 1, any
    scale. Keeps the four heaviest, renormalises, rounds to 8-bit weights that sum to exactly 255
    (the rounding remainder goes to the heaviest), heaviest first; unused slots are 0 (weight 0,
    joint 0). A splat with no weight at all is bound to its first joint."""
    joints = np.asarray(joints)
    weights = np.asarray(weights, dtype=np.float64)
    if joints.ndim != 2 or joints.shape != weights.shape or joints.shape[1] < 1:
        raise AvatarFormatError("joints and weights must both be (N, K) with K >= 1")
    if not np.all(np.isfinite(weights)):
        raise AvatarFormatError("skin weights must be finite")
    if joints.size and (joints.min() < 0 or joints.max() > 255):
        raise AvatarFormatError("joint indices must be 0..255")
    n, k = joints.shape
    weights = np.clip(weights, 0.0, None)
    order = np.argsort(-weights, axis=1, kind="stable")[:, :4]
    top_w = np.take_along_axis(weights, order, axis=1)
    top_j = np.take_along_axis(joints, order, axis=1).astype(np.int64)
    if k < 4:
        top_w = np.concatenate([top_w, np.zeros((n, 4 - k))], axis=1)
        top_j = np.concatenate([top_j, np.zeros((n, 4 - k), np.int64)], axis=1)
    total = top_w.sum(axis=1, keepdims=True)
    empty = total[:, 0] <= 0
    top_w[empty] = (1.0, 0.0, 0.0, 0.0)
    total[empty] = 1.0
    w8 = np.floor(top_w / total * 255.0).astype(np.int64)
    w8[:, 0] += 255 - w8.sum(axis=1)
    top_j[w8 == 0] = 0
    return (w8 | (top_j << 8)).astype(np.uint16)


def _fail(message: str) -> None:
    raise AvatarFormatError(message)


def _check_skeleton(names, parents, rest) -> None:
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        _fail("joint names must be a list of strings")
    count = len(names)
    if not BODY_JOINTS <= count <= MAX_JOINTS:
        _fail(f"{count} joints: need {BODY_JOINTS}..{MAX_JOINTS}")
    if len(set(names)) != count:
        _fail("joint names must be unique")
    if tuple(names[:BODY_JOINTS]) != SMPLX_JOINTS[:BODY_JOINTS]:
        _fail("joints 0..21 must be SMPL-X's body joints, in order")
    if (not isinstance(parents, list) or len(parents) != count
            or not all(isinstance(p, int) and not isinstance(p, bool) for p in parents)):
        _fail("parents must be one int per joint")
    if parents[0] != -1 or any(not 0 <= parents[j] < j for j in range(1, count)):
        _fail("parents must form a tree with every parent before its child")
    if tuple(parents[:BODY_JOINTS]) != SMPLX_PARENTS[:BODY_JOINTS]:
        _fail("joints 0..21 must have SMPL-X's parents")
    try:
        rest = np.asarray(rest, dtype=np.float64)
    except (TypeError, ValueError):
        _fail("rest must be numbers")
    if (rest.shape != (count, 3) or not np.all(np.isfinite(rest))
            or np.abs(rest).max() > COORD_LIMIT):
        _fail("rest must be one finite position per joint, within 3 m")


def _check_splats(a: Avatar, joint_count: int) -> float:
    """Value checks shared by pack and unpack; returns the body's height over floor_y."""
    n = a.count
    if not 1 <= n <= MAX_SPLATS:
        _fail(f"{n} splats: need 1..{MAX_SPLATS}")
    for name, _dtype, cols in ARRAYS:
        if getattr(a, name).shape != (n, cols):
            _fail(f"{name} must be ({n}, {cols})")
    if not np.all(np.isfinite(a.positions)) or np.abs(a.positions).max() > COORD_LIMIT:
        _fail("positions must be finite and within 3 m of the origin")
    if not np.all(np.isfinite(a.scales)) or a.scales.min() < 0 or a.scales.max() > MAX_SCALE:
        _fail("scales must be finite, 0..0.5 m")
    norms = np.linalg.norm(a.rotations.astype(np.float64), axis=1)
    if not np.all(np.isfinite(norms)) or norms.min() < 0.5 or norms.max() > 1.5:
        _fail("rotations must be unit quaternions")
    weights = (a.skin & 0xFF).astype(np.int64)
    joints = (a.skin >> 8).astype(np.int64)
    if joints.max() >= joint_count:
        _fail("a skin joint index is past the skeleton")
    if np.any(weights.sum(axis=1) != 255):
        _fail("each splat's skin weights must sum to 255")
    if not (isinstance(a.floor_y, float) and np.isfinite(a.floor_y)
            and abs(a.floor_y) <= COORD_LIMIT):
        _fail("floor_y must be a finite float within 3 m")
    height = float(np.percentile(a.positions[:, 1], 99.8)) - a.floor_y
    if not MIN_HEIGHT <= height <= MAX_HEIGHT:
        _fail(f"the body is {height:.2f} m tall: need {MIN_HEIGHT}..{MAX_HEIGHT}")
    if not isinstance(a.kind, str) or not _KIND.fullmatch(a.kind):
        _fail("kind must be 1-16 of [a-z0-9_-]")
    if not isinstance(a.meta, dict):
        _fail("meta must be a dict")
    return height


def _aligned(offset: int) -> int:
    return (offset + ALIGN - 1) // ALIGN * ALIGN


def pack(avatar: Avatar, *, shuffle: bool = True, seed: int = 0) -> bytes:
    """The avatar as an LRAV v1 file. Raises AvatarFormatError for anything `unpack` would refuse
    (including a file over MAX_BYTES), so a file that packs is a file the server will serve."""
    a = Avatar(
        positions=np.asarray(avatar.positions, dtype=np.float32),
        scales=np.asarray(avatar.scales, dtype=np.float32),
        rotations=np.asarray(avatar.rotations, dtype=np.float32),
        colors=np.asarray(avatar.colors, dtype=np.uint8),
        skin=np.asarray(avatar.skin, dtype=np.uint16),
        joint_names=list(avatar.joint_names), parents=[int(p) for p in avatar.parents],
        rest=np.asarray(avatar.rest, dtype=np.float32), floor_y=float(avatar.floor_y),
        kind=avatar.kind, meta=dict(avatar.meta),
    )
    _check_skeleton(a.joint_names, a.parents, a.rest)
    height = _check_splats(a, len(a.joint_names))
    columns = {"positions": a.positions, "scales": a.scales,
               "rotations": a.rotations / np.linalg.norm(a.rotations, axis=1, keepdims=True),
               "colors": a.colors, "skin": a.skin}
    if shuffle:
        order = np.random.default_rng(seed).permutation(a.count)
        columns = {name: values[order] for name, values in columns.items()}
    blobs, table, offset = [], {}, 0
    for name, dtype, cols in ARRAYS:
        raw = np.ascontiguousarray(columns[name], dtype=np.dtype(dtype)).tobytes()
        offset = _aligned(offset)
        table[name] = {"dtype": dtype, "shape": [a.count, cols], "offset": offset}
        blobs.append((offset, raw))
        offset += len(raw)
    header = {
        "format": "lrav", "version": VERSION, "count": a.count, "kind": a.kind,
        "floor_y": round(a.floor_y, 6), "height": round(height, 4),
        "joints": {"names": a.joint_names, "parents": a.parents,
                   "rest": [[round(float(v), 6) for v in row] for row in a.rest]},
        "arrays": table, "meta": a.meta,
    }
    try:
        text = json.dumps(header, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AvatarFormatError(f"header is not plain JSON: {exc}") from None
    padded = _aligned(len(text))
    if padded > MAX_HEADER:
        _fail("header over 64 KiB (meta too big?)")
    data = bytearray(PREAMBLE + padded + offset)
    data[:PREAMBLE] = MAGIC + struct.pack("<III", VERSION, padded, 0)
    data[PREAMBLE:PREAMBLE + padded] = text.ljust(padded, b" ")
    start = PREAMBLE + padded
    for at, raw in blobs:
        data[start + at:start + at + len(raw)] = raw
    if len(data) > MAX_BYTES:
        _fail(f"{len(data)} bytes: over the {MAX_BYTES} byte budget")
    return bytes(data)


def unpack(data: bytes) -> Avatar:
    """Parse and fully check an LRAV file. Raises AvatarFormatError on anything off."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        _fail("not bytes")
    data = bytes(data)
    if not PREAMBLE <= len(data) <= MAX_BYTES:
        _fail(f"{len(data)} bytes: need {PREAMBLE}..{MAX_BYTES}")
    if data[:4] != MAGIC:
        _fail("not an avatar file (bad magic)")
    version, header_len, _reserved = struct.unpack_from("<III", data, 4)
    if version != VERSION:
        _fail(f"unsupported avatar version {version}")
    if header_len % ALIGN or not 0 < header_len <= MAX_HEADER or PREAMBLE + header_len > len(data):
        _fail("bad header length")
    try:
        header = json.loads(data[PREAMBLE:PREAMBLE + header_len].decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        _fail("header is not JSON")
    if not isinstance(header, dict) or header.get("format") != "lrav" \
            or header.get("version") != VERSION:
        _fail("header is not an lrav v1 header")
    count = header.get("count")
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= MAX_SPLATS:
        _fail("bad splat count")
    joints = header.get("joints")
    if not isinstance(joints, dict):
        _fail("missing skeleton")
    names, parents, rest = joints.get("names"), joints.get("parents"), joints.get("rest")
    _check_skeleton(names, parents, rest)
    table = header.get("arrays")
    if not isinstance(table, dict) or set(table) != {name for name, _, _ in ARRAYS}:
        _fail("arrays must be exactly " + ", ".join(name for name, _, _ in ARRAYS))
    start = PREAMBLE + header_len
    block = len(data) - start
    arrays, end = {}, 0
    for name, dtype, cols in ARRAYS:
        entry = table[name]
        if not isinstance(entry, dict) or entry.get("dtype") != dtype \
                or entry.get("shape") != [count, cols]:
            _fail(f"{name}: wrong dtype or shape")
        offset = entry.get("offset")
        if not isinstance(offset, int) or isinstance(offset, bool) or offset % ALIGN \
                or offset < end:
            _fail(f"{name}: bad offset")
        size = count * cols * np.dtype(dtype).itemsize
        if offset + size > block:
            _fail(f"{name}: runs past the end of the file")
        arrays[name] = np.frombuffer(data, dtype=np.dtype(dtype), count=count * cols,
                                     offset=start + offset).reshape(count, cols)
        end = offset + size
    if block - end >= ALIGN:
        _fail("trailing bytes after the arrays")
    floor_y = header.get("floor_y")
    if isinstance(floor_y, int) and not isinstance(floor_y, bool):
        floor_y = float(floor_y)
    rotations = arrays["rotations"].astype(np.float32)
    avatar = Avatar(
        positions=arrays["positions"].astype(np.float32),
        scales=arrays["scales"].astype(np.float32),
        rotations=rotations,
        colors=arrays["colors"].copy(),
        skin=arrays["skin"].astype(np.uint16),
        joint_names=list(names), parents=list(parents),
        rest=np.asarray(rest, dtype=np.float32), floor_y=floor_y,
        kind=header.get("kind"), meta=header.get("meta", {}),
    )
    _check_splats(avatar, len(names))
    avatar.rotations = rotations / np.linalg.norm(rotations, axis=1, keepdims=True)
    return avatar
