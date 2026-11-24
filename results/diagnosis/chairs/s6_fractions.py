"""Per real chair (found from the splat, not from the boxes): where its splats went in the served
bundle (room / Chair0 / Chair1 / Chair2), by count and opacity-weighted. And per body: what its
splats are."""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.cluster.vq import kmeans2
from common import *
from litereality_agent.pipeline.splatroom.split import _static_boxes, _inside_any
from litereality_agent.room_ops.roomcall.model_info import movable_body_ids
man = bundle()
mc, dc = model_data(B / "physics/scene_splat.xml")
names = ["room", "Chair0", "Chair1", "Chair2", "Wall0_Door_0"]
P, O, L, S = [], [], [], []
room = load_spz(B / man["room"])
nfill = man["counts"].get("floor_fill", 0)
keep = slice(0, len(room["pos"]) - nfill)            # fill splats are appended last (Splats.concat)
P.append(room["pos"][keep]); O.append(room["op"][keep]); S.append(room["sc"][keep]); L.append(np.zeros(len(room["pos"]) - nfill, np.int8))
for i, nm in enumerate(names[1:], 1):
    s = load_spz(B / man["bodies"][nm]); R, x = body_pose(mc, dc, nm)
    with np.errstate(all="ignore"):
        P.append((s["pos"].astype(np.float64) @ R.T + x).astype(np.float32))
    O.append(s["op"]); S.append(s["sc"]); L.append(np.full(len(s["pos"]), i, np.int8))
P = np.concatenate(P); O = np.concatenate(O); L = np.concatenate(L); S = np.concatenate(S)
del room
FLOOR = -1.2677
static = _static_boxes(mc, dc, sorted(movable_body_ids(mc)))
region = (P[:, 2] > FLOOR + 0.04) & (P[:, 2] < FLOOR + 1.10) & (P[:, 0] > -1.9) & (P[:, 0] < 1.3) & (P[:, 1] > -0.4) & (P[:, 1] < 2.1)
ridx = np.flatnonzero(region)
with np.errstate(all="ignore"):
    ridx = ridx[~_inside_any(P[ridx].astype(np.float64), static, 0.03)]
# seeds: opaque splats -> 3 cm voxel components (as s5), chairs = components containing the seed points
v = 0.03; lo = P[ridx].min(0)
g = ((P[ridx] - lo) / v).astype(int); shape = g.max(0) + 1
op = O[ridx] > 0.3
occ = np.zeros(shape, np.int32); np.add.at(occ, tuple(g[op].T), 1)
lab, n = ndimage.label(occ >= 2, structure=np.ones((3, 3, 3)))
# grow each component by one voxel so faint splats hugging the chair count too
grown = ndimage.grey_dilation(lab, size=(3, 3, 3))
lab_all = grown[tuple(g.T)]
def comp_at(xy):
    d = np.hypot(P[ridx, 0] - xy[0], P[ridx, 1] - xy[1])
    near = (d < 0.12) & (lab_all > 0) & op
    vals, cnt = np.unique(lab_all[near], return_counts=True)
    return int(vals[np.argmax(cnt)])
c_blue = comp_at((0.57, 1.37)); c_pair = comp_at((-1.1, 0.15)); c_dark = comp_at((-0.65, 0.45))
print("components: blue", c_blue, "teal", c_pair, "dark", c_dark, "(same component = the two chairs touch)")
true = {}
true["blue chair (near Chair0)"] = ridx[lab_all == c_blue]
pair = ridx[lab_all == c_pair]
if c_dark == c_pair:
    # split the merged pair: 2-means on xy seeded at the two seats
    cent, lbl = kmeans2(P[pair, :2].astype(np.float64), np.array([[-1.1, 0.15], [-0.7, 0.5]]), minit="matrix", seed=0)
    print("2-means centres", cent.round(3))
    true["teal chair (near Chair2)"] = pair[lbl == 0]
    true["dark chair (no body; 0.45 m from Chair1 box)"] = pair[lbl == 1]
else:
    true["teal chair (near Chair2)"] = pair; true["dark chair (no body; 0.45 m from Chair1 box)"] = ridx[lab_all == c_dark]
rows = {}
for k, ix in true.items():
    cnt = np.array([(L[ix] == i).sum() for i in range(len(names))])
    w = np.array([O[ix][L[ix] == i].sum() for i in range(len(names))])
    rows[k] = {"splats": int(len(ix)), "centre": P[ix].mean(0).round(3).tolist(),
               "by_count": {names[i]: round(float(cnt[i] / cnt.sum()), 3) for i in range(len(names))},
               "by_opacity": {names[i]: round(float(w[i] / w.sum()), 3) for i in range(len(names))}}
    print(k, json.dumps(rows[k]))
# per body: what its splats are
inchair = np.zeros(len(P), np.int8) - 1
for j, ix in enumerate(true.values()): inchair[ix] = j
per_body = {}
for i, nm in enumerate(names[1:4], 1):
    mine = L == i
    tot = mine.sum(); w = O[mine].sum()
    d = {"splats": int(tot)}
    for j, k in enumerate(true):
        m = mine & (inchair == j)
        d[k] = {"count": round(float(m.sum() / tot), 3), "opacity": round(float(O[m].sum() / w), 3)}
    m = mine & (inchair == -1)
    d["none of the real chairs (floor, table, wall, floaters)"] = {"count": round(float(m.sum() / tot), 3), "opacity": round(float(O[m].sum() / w), 3)}
    big = mine & (S.max(1) > 0.05)
    d["big gaussians (>5 cm sigma)"] = int(big.sum())
    per_body[nm] = d
    print(nm, json.dumps(d))
json.dump({"true_chairs": rows, "bodies": per_body}, open(OUT / "fractions.json", "w"), indent=1)
# picture
fig, axs = plt.subplots(1, 3, figsize=(21, 7))
cl = {0: (0.75, 0.75, 0.75), 1: (1, 0.15, 0.15), 2: (0.1, 0.75, 0.1), 3: (0.15, 0.35, 1), 4: (1, 0.8, 0)}
for ax, (k, ix) in zip(axs, true.items()):
    for i in range(4):
        m = ix[L[ix] == i]
        ax.scatter(P[m, 0], P[m, 1], s=0.4, color=cl[i], label=f"{names[i]} {rows[k]['by_count'][names[i]]*100:.0f}%", rasterized=True)
    for nm, c in (("Chair0", "r"), ("Chair1", "g"), ("Chair2", "b")):
        b = mujoco.mj_name2id(mc, mujoco.mjtObj.mjOBJ_BODY, nm)
        for gg in range(mc.body_geomadr[b], mc.body_geomadr[b] + mc.body_geomnum[b]):
            R = dc.geom_xmat[gg].reshape(3, 3); c0 = dc.geom_xpos[gg]; h = mc.geom_size[gg]
            cs = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]) * h[:2]; w2 = cs @ R[:2, :2].T + c0[:2]
            ax.plot(w2[:, 0], w2[:, 1], "-", color=c, lw=0.7)
    c = P[ix].mean(0); ax.set_xlim(c[0] - 0.8, c[0] + 0.8); ax.set_ylim(c[1] - 0.8, c[1] + 0.8); ax.set_aspect("equal")
    ax.legend(markerscale=15, fontsize=9); ax.set_title(f"{k}: {len(ix)} splats\nwho owns them in the served bundle (boxes = corrected physics)", fontsize=9)
plt.tight_layout(); plt.savefig(OUT / "true_chairs_ownership.png", dpi=100)

# ---- check the sets against the photos: draw each real-chair set (and each body's splats) onto Stray photos
from PIL import Image, ImageDraw
from litereality_agent.models.splat import rgbd
frames = {f.index: f for f in rgbd.load_frames(Path.home() / "LiteReality-scans/stray_atrium_q600")}
def proj(f, pts):
    w2c = np.linalg.inv(f.c2w); cam = pts @ w2c[:3, :3].T + w2c[:3, 3]
    z = -cam[:, 2]; K = f.K
    return K[0, 0] * cam[:, 0] / z + K[0, 2], -K[1, 1] * cam[:, 1] / z + K[1, 2], z
rng = np.random.default_rng(0)
for fi in (63, 12, 254, 612, 16):
    f = frames[fi]
    for mode in ("truth", "owner"):
        im = Image.open(f.image).convert("RGB"); dr = ImageDraw.Draw(im)
        if mode == "truth":
            sets = [(ix, c) for ix, c in zip(true.values(), [(40, 90, 255), (0, 230, 200), (255, 140, 0)])]
        else:
            sets = [(np.flatnonzero((L == i) & region), c) for i, c in ((1, (255, 40, 40)), (2, (40, 230, 40)), (3, (40, 90, 255)))]
        for ix, c in sets:
            ix = ix[O[ix] > 0.3]
            if len(ix) > 6000: ix = rng.choice(ix, 6000, replace=False)
            u, vv, z = proj(f, P[ix].astype(np.float64))
            ok = z > 0.1
            for a, b in zip(u[ok], vv[ok]): dr.point((a, b), fill=c)
            for a, b in zip(u[ok][::3], vv[ok][::3]): dr.rectangle([a - 1, b - 1, a + 1, b + 1], fill=c)
        im = im.rotate(-90, expand=True); im.thumbnail((1000, 1000))
        im.save(OUT / f"overlay_{mode}_f{fi:05d}.jpg", quality=85)
print("overlays saved")
