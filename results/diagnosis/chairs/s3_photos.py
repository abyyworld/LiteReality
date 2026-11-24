"""Stray photos that see each chair best, with the scanned (red) and corrected (green) chair boxes projected."""
import sys, numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path.home() / "LiteReality/src")) if False else None
from common import *
from litereality_agent.models.splat import rgbd
frames = rgbd.load_frames(Path.home() / "LiteReality-scans/stray_atrium_q600")
print(len(frames))
mc, dc = model_data(B / "physics/scene_splat.xml")
mo, do = model_data(B / "physics/scene.xml")

def corners(m, d, name):
    from litereality_agent.pipeline.splatroom.masks import _box_corners
    return _box_corners(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name))

def proj(f, pts):
    w2c = np.linalg.inv(f.c2w); cam = pts @ w2c[:3, :3].T + w2c[:3, 3]
    z = -cam[:, 2]; K = f.K
    return K[0, 0] * cam[:, 0] / z + K[0, 2], -K[1, 1] * cam[:, 1] / z + K[1, 2], z

E = [(0,1),(1,3),(3,2),(2,0),(4,5),(5,7),(7,6),(6,4),(0,4),(1,5),(2,6),(3,7)]
names = ["Chair0", "Chair1", "Chair2"]
# one overview frame that sees all three chairs, plus the best frame per chair
allc = {n: corners(mc, dc, n) for n in names}
def score(f, pts):
    u, v, z = proj(f, pts)
    W, H = f.size
    if np.any(z < 0.3): return -1
    inside = np.mean((u > 0) & (u < W) & (v > 0) & (v < H))
    return inside
cands = []
for f in frames:
    s = [score(f, allc[n]) for n in names]
    cands.append((min(s), f))
cands.sort(key=lambda t: -t[0])
picks = [("all", cands[0][1]), ("all2", cands[len(cands)//50][1])]
for n in names:
    best = max(frames, key=lambda f: score(f, allc[n]) - 0.1 * abs(np.linalg.norm(f.c2w[:3, 3] - allc[n].mean(0)) - 1.6))
    picks.append((n, best))
for tag, f in picks:
    im = Image.open(f.image).convert("RGB")
    dr = ImageDraw.Draw(im)
    for n in names:
        for (m, d, col) in ((mo, do, (255, 60, 60)), (mc, dc, (60, 255, 60))):
            u, v, z = proj(f, corners(m, d, n))
            if np.any(z < 0.05): continue
            for a, b in E: dr.line([(u[a], v[a]), (u[b], v[b])], fill=col, width=4)
            dr.text((u.mean(), v.min() - 30), n, fill=col)
    im.thumbnail((1200, 1200))
    im.save(OUT / f"photo_{tag}_f{f.index:05d}.jpg", quality=85)
    print(tag, f.index, f.image.name)
