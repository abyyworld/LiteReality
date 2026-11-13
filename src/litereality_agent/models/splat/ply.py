"""Reading the standard 3D Gaussian Splatting PLY (what gsplat's `export_splats` writes).

Binary little-endian vertices with float properties ``x y z``, optional ``nx ny nz``,
``f_dc_0..2`` (SH DC), ``f_rest_*`` (higher SH, channel-major: all of red's coefficients, then
green's, then blue's), ``opacity`` (a logit), ``scale_0..2`` (natural logs) and ``rot_0..3``
(a quaternion w, x, y, z). Returned as `spz.Splats` in linear units.
"""

from __future__ import annotations

import numpy as np

from litereality_agent.models.splat.spz import SH_C0, Splats, sh_dim

_TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1",
          "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4", "short": "<i2",
          "ushort": "<u2"}


def read(data: bytes, *, max_sh_degree: int = 3) -> Splats:
    end = data.find(b"end_header\n")
    if not data.startswith(b"ply") or end < 0:
        raise ValueError("not a PLY file")
    header = data[:end].decode("ascii", "replace").splitlines()
    if "format binary_little_endian 1.0" not in header:
        raise ValueError("only binary little-endian PLY is read")
    count, props, in_vertex = 0, [], False
    for line in header:
        parts = line.split()
        if parts[:2] == ["element", "vertex"]:
            count, in_vertex = int(parts[2]), True
        elif parts and parts[0] == "element":
            in_vertex = False
        elif in_vertex and parts and parts[0] == "property":
            if parts[1] == "list":
                raise ValueError("list properties are not splats")
            props.append((parts[2], _TYPES[parts[1]]))
    v = np.frombuffer(data, dtype=np.dtype(props), count=count, offset=end + len("end_header\n"))
    names = set(v.dtype.names)
    positions = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    dc = np.stack([v[f"f_dc_{i}"] for i in range(3)], 1)
    colors = (0.5 + SH_C0 * dc).astype(np.float32)
    opacities = (1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float64)))).astype(np.float32)
    scales = np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], 1)).astype(np.float32)
    wxyz = np.stack([v[f"rot_{i}"] for i in range(4)], 1).astype(np.float64)
    wxyz /= np.maximum(np.linalg.norm(wxyz, axis=1, keepdims=True), 1e-12)
    rotations = wxyz[:, [1, 2, 3, 0]].astype(np.float32)
    rest = sorted((n for n in names if n.startswith("f_rest_")), key=lambda n: int(n[7:]))
    sh = None
    if rest and max_sh_degree > 0:
        per_channel = len(rest) // 3
        degree = next((d for d in (3, 2, 1) if sh_dim(d) == per_channel), 0)
        keep = sh_dim(min(degree, max_sh_degree))
        if keep:
            r = np.stack([v[n] for n in rest], 1).reshape(count, 3, per_channel)
            sh = np.ascontiguousarray(r[:, :, :keep].transpose(0, 2, 1)).astype(np.float32)
    return Splats(positions, scales, rotations, colors, opacities, sh)


def write(splats: Splats) -> bytes:
    """`splats` as a 3DGS PLY (tests, and handing a room to other splat tools)."""
    n, k = len(splats), (0 if splats.sh is None else splats.sh.shape[1])
    fields = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"]
    fields += [f"f_rest_{i}" for i in range(3 * k)]
    fields += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    cols = [splats.positions, (splats.colors - 0.5) / SH_C0]
    if k:
        cols.append(splats.sh.transpose(0, 2, 1).reshape(n, 3 * k))
    op = np.clip(splats.opacities.astype(np.float64), 1e-6, 1 - 1e-6)
    cols += [np.log(op / (1 - op))[:, None], np.log(np.maximum(splats.scales, 1e-12)),
             splats.rotations[:, [3, 0, 1, 2]]]
    body = np.concatenate([np.asarray(c, np.float32).reshape(n, -1) for c in cols], 1)
    head = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    head += [f"property float {f}" for f in fields] + ["end_header"]
    return ("\n".join(head) + "\n").encode() + body.astype("<f4").tobytes()
