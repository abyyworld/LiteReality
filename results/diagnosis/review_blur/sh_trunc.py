"""Reviewer check: how much does exporting SH degree 3 -> 1 (what every served SPZ does) cost at a
TRAINING view? Uses a LichtFeld SH3 PLY of the same atrium (q206 dataset) since no gsplat PLY is
local. CPU numpy EWA crop renderer (3DGS maths, +0.3 px dilation, no compensation), 480x360 crop at
full 1920x1440 resolution, centre of frame.
   uv run --frozen python results/diagnosis/review_blur/sh_trunc.py RUN FRAME_STEM [cx cy]"""
import json, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
from litereality_agent.pipeline.evaluate import fidelity as fd
from litereality_agent.models.splat import spz
HERE = Path(__file__).resolve().parent
RUN = Path(sys.argv[1]); STEM = sys.argv[2]
SCAN = Path.home() / "LiteReality-scans/stray_atrium_q206"
C0 = 0.28209479177387814; C1 = 0.4886025119029199
C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658, 1.445305721320277, -0.5900435899266435]
def read_ply(p):
    raw = p.read_bytes(); h = raw.index(b"end_header\n") + len(b"end_header\n")
    head = raw[:h].decode(errors="ignore").splitlines()
    n = int([l for l in head if l.startswith("element vertex")][0].split()[-1])
    props = [l.split()[-1] for l in head if l.startswith("property")]
    a = np.frombuffer(raw, np.float32, n * len(props), h).reshape(n, len(props))
    return {k: a[:, i] for i, k in enumerate(props)}, n
P, n = read_ply(RUN / "splat.ply")
pos = np.stack([P["x"], P["y"], P["z"]], 1).astype(np.float64)
s = spz.decode((RUN / "raw.spz.bin").read_bytes())
print("ply n", n, "bounds", np.percentile(pos, [5, 95], 0).round(2).tolist())
print("spz n", len(s.positions), "bounds", np.percentile(s.positions, [5, 95], 0).round(2).tolist())
m = json.loads((SCAN / f"{STEM}.json").read_text())
pose = fd.camera_pose(m["cameraPoseARFrame"]); K = np.asarray(m["intrinsics"], float).reshape(3, 3)
fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
CW, CH = 480, 360
x0 = int(sys.argv[3]) if len(sys.argv) > 3 else 960 - CW // 2
y0 = int(sys.argv[4]) if len(sys.argv) > 4 else 720 - CH // 2
R, t = pose[:3, :3], pose[:3, 3]
W = np.diag([1.0, -1.0, -1.0]) @ R.T
pc = (pos - t) @ W.T; z = pc[:, 2]
op_all = 1 / (1 + np.exp(-P["opacity"].astype(np.float64)))
keep = (z > 0.05) & (op_all > 1 / 255)
u = fx * pc[:, 0] / np.maximum(z, 1e-6) + cx; v = fy * pc[:, 1] / np.maximum(z, 1e-6) + cy
keep &= (u > x0 - 400) & (u < x0 + CW + 400) & (v > y0 - 400) & (v < y0 + CH + 400)
idx = np.flatnonzero(keep)
q = np.stack([P["rot_0"], P["rot_1"], P["rot_2"], P["rot_3"]], 1)[idx].astype(np.float64)  # w x y z
q /= np.linalg.norm(q, axis=1, keepdims=True); w_, x, y, zq = q.T
Rg = np.stack([np.stack([1-2*(y*y+zq*zq), 2*(x*y-zq*w_), 2*(x*zq+y*w_)], -1),
               np.stack([2*(x*y+zq*w_), 1-2*(x*x+zq*zq), 2*(y*zq-x*w_)], -1),
               np.stack([2*(x*zq-y*w_), 2*(y*zq+x*w_), 1-2*(x*x+y*y)], -1)], 1)
S2 = np.exp(2 * np.stack([P["scale_0"], P["scale_1"], P["scale_2"]], 1)[idx].astype(np.float64))
Sig = np.einsum("nij,nj,nkj->nik", Rg, S2, Rg); Sc = np.einsum("ij,njk,lk->nil", W, Sig, W)
zi, xi, yi = z[idx], pc[idx, 0], pc[idx, 1]
J = np.zeros((len(idx), 2, 3)); J[:, 0, 0] = fx / zi; J[:, 0, 2] = -fx * xi / zi**2; J[:, 1, 1] = fy / zi; J[:, 1, 2] = -fy * yi / zi**2
S2d = np.einsum("nij,njk,nlk->nil", J, Sc, J); S2d[:, 0, 0] += 0.3; S2d[:, 1, 1] += 0.3
det = S2d[:, 0, 0] * S2d[:, 1, 1] - S2d[:, 0, 1]**2
inv = np.stack([S2d[:, 1, 1], -S2d[:, 0, 1], S2d[:, 0, 0]], 1) / det[:, None]
rad = 3 * np.sqrt(np.maximum(S2d[:, 0, 0], S2d[:, 1, 1])); ui, vi = u[idx], v[idx]
hit = (ui + rad > x0) & (ui - rad < x0 + CW) & (vi + rad > y0) & (vi - rad < y0 + CH)
d = pos[idx] - t; d /= np.linalg.norm(d, axis=1, keepdims=True); dx, dy, dz = d.T
dc = np.stack([P[f"f_dc_{i}"] for i in range(3)], 1)[idx].astype(np.float64)
rest = np.stack([P[f"f_rest_{i}"] for i in range(45)], 1)[idx].astype(np.float64).reshape(-1, 3, 15)  # channel-major
B = [-C1*dy, C1*dz, -C1*dx,
     C2[0]*dx*dy, C2[1]*dy*dz, C2[2]*(2*dz*dz-dx*dx-dy*dy), C2[3]*dx*dz, C2[4]*(dx*dx-dy*dy),
     C3[0]*dy*(3*dx*dx-dy*dy), C3[1]*dx*dy*dz, C3[2]*dy*(4*dz*dz-dx*dx-dy*dy), C3[3]*dz*(2*dz*dz-3*dx*dx-3*dy*dy),
     C3[4]*dx*(4*dz*dz-dx*dx-dy*dy), C3[5]*dz*(dx*dx-dy*dy), C3[6]*dx*(dx*dx-3*dy*dy)]
B = np.stack(B, 1)
cols = {}
for deg, k in [(0, 0), (1, 3), (3, 15)]:
    c = 0.5 + C0 * dc + (np.einsum("nck,nk->nc", rest[:, :, :k], B[:, :k]) if k else 0)
    cols[deg] = np.clip(c, 0, 1)
order = np.flatnonzero(hit)[np.argsort(zi[hit])]
op = op_all[idx]
print(len(order), "splats touch the crop", flush=True)
imgs = {}
for deg in (0, 1, 3):
    C = np.zeros((CH, CW, 3)); T = np.ones((CH, CW)); col = cols[deg]
    for kk in order:
        r = rad[kk]
        ax0, ax1 = int(max(x0, np.floor(ui[kk]-r))), int(min(x0+CW, np.ceil(ui[kk]+r)+1))
        ay0, ay1 = int(max(y0, np.floor(vi[kk]-r))), int(min(y0+CH, np.ceil(vi[kk]+r)+1))
        if ax1 <= ax0 or ay1 <= ay0: continue
        xs = np.arange(ax0, ax1) + 0.5 - ui[kk]; ys = np.arange(ay0, ay1) + 0.5 - vi[kk]
        a, b, c = inv[kk]
        al = np.minimum(0.99, op[kk] * np.exp(-0.5*(a*xs[None]**2 + 2*b*xs[None]*ys[:, None] + c*ys[:, None]**2)))
        al[al < 1/255] = 0
        sl = (slice(ay0-y0, ay1-y0), slice(ax0-x0, ax1-x0)); Tl = T[sl]
        C[sl] += (Tl*al)[..., None] * col[kk]; T[sl] = Tl*(1-al)
    imgs[deg] = np.clip(C + T[..., None]*0.0, 0, 1)
photo = np.asarray(Image.open(SCAN / f"{STEM}.jpg").convert("RGB").resize((1920, 1440)), np.float64)[y0:y0+CH, x0:x0+CW] / 255
def cfit(a, b):
    A = np.c_[a.reshape(-1, 3), np.ones(a.size//3)]; M, *_ = np.linalg.lstsq(A, b.reshape(-1, 3), rcond=None)
    return np.clip(A @ M, 0, 1).reshape(a.shape)
def psnr(a, b): return round(float(10*np.log10(1/np.mean((a-b)**2))), 2)
res = {"run": str(RUN), "frame": STEM, "crop_xy": [x0, y0]}
for deg, im in imgs.items():
    res[f"sh{deg}"] = {"psnr_vs_photo": psnr(im, photo), "psnr_cc_vs_photo": psnr(cfit(im, photo), photo)}
res["sh1_vs_sh3_psnr"] = psnr(imgs[1], imgs[3]); res["sh0_vs_sh3_psnr"] = psnr(imgs[0], imgs[3])
print(json.dumps(res))
tag = f"{RUN.name}_{STEM}_{x0}_{y0}"
Image.fromarray((np.concatenate([photo, imgs[3], imgs[1], imgs[0]], 1)*255).astype(np.uint8)).save(HERE / f"sh_{tag}.png")
(HERE / f"sh_{tag}.json").write_text(json.dumps(res, indent=1))
