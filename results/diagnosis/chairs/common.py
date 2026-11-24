"""Shared loaders for the chair diagnosis (no SH decoded: memory)."""
import gzip, struct, math, re
from pathlib import Path
import numpy as np
import mujoco

ROOT = Path.home() / "LiteReality"
B = ROOT / "run/atrium_stray_q/v2"
OUT = ROOT / "results/diagnosis/chairs"


def load_spz(path, rot=False):
    raw = gzip.decompress(Path(path).read_bytes())
    magic, version, n, degree, bits, _f, _ = struct.unpack_from("<IIIBBBB", raw)
    at = 16
    p = np.frombuffer(raw, np.uint8, n * 9, at).reshape(n, 3, 3).astype(np.int32); at += n * 9
    fx = p[:, :, 0] | (p[:, :, 1] << 8) | (p[:, :, 2] << 16)
    fx = np.where(fx & 0x800000, fx - (1 << 24), fx)
    pos = (fx / float(1 << bits)).astype(np.float32); del p, fx
    op = (np.frombuffer(raw, np.uint8, n, at) / 255.0).astype(np.float32); at += n
    col = ((np.frombuffer(raw, np.uint8, n * 3, at).reshape(n, 3) / 255.0 - 0.5) / 0.15 * 0.28209479177387814 + 0.5).astype(np.float32); at += n * 3
    sc = np.exp(np.frombuffer(raw, np.uint8, n * 3, at).reshape(n, 3) / 16.0 - 10.0).astype(np.float32); at += n * 3
    del raw
    return dict(pos=pos, op=op, col=np.clip(col, 0, 1), sc=sc)


def model_data(xml):
    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    return m, d


def body_pose(m, d, name):
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    return np.asarray(d.xmat[b]).reshape(3, 3).copy(), np.asarray(d.xpos[b]).copy()


def bundle():
    import json
    return json.loads((B / "splat.json").read_text())
