"""Render the SAME SPZ (run/atrium_stray_q/v2/raw.spz.bin) with gsplat 1.5.3 (the trainer's
rasteriser, antialiased mode as trained) at the same 9 held-out cameras Spark drew in
variants_9views.json, so Spark vs gsplat is compared on identical splats.

One ephemeral `modal run` on an L4 (a few minutes, well under $0.10). Nothing is deployed.

    uv run --frozen modal run results/diagnosis/export_path/gsplat_render_modal.py
"""
import io
import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
LOCAL = len(HERE.parents) > 2            # False inside the container (/root/<this file>)
REPO = HERE.parents[2] if LOCAL else HERE
SPZ_PY = REPO / "src/litereality_agent/models/splat/spz.py"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("numpy<2", "pillow", "packaging", "ninja", "jaxtyping", "rich")
    .pip_install("torch==2.4.1", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("gsplat==1.5.3+pt24cu124", extra_index_url="https://docs.gsplat.studio/whl/pt24cu124")
    .add_local_file(SPZ_PY, "/root/spz_mod.py")   # read on the local side only (lazy)
)
app = modal.App("diag-gsplat-render")


MODES = [("sh1_aa_clip01", 1, "antialiased", "clip01"), ("sh1_aa_sparkpack", 1, "antialiased", "sparkpack")]


@app.function(image=image, gpu="L4", timeout=900)
def render(spz_bytes: bytes, cams: list) -> list:
    import numpy as np
    import torch
    from gsplat import rasterization
    import spz_mod as spz
    from PIL import Image

    s = spz.decode(spz_bytes)
    dev = "cuda"
    means = torch.tensor(s.positions, device=dev)
    quats = torch.tensor(s.rotations[:, [3, 0, 1, 2]], device=dev)        # xyzw -> wxyz
    scales = torch.tensor(s.scales, device=dev)
    opac = torch.tensor(s.opacities, device=dev)
    dc = (torch.tensor(s.colors, device=dev) - 0.5) / spz.SH_C0
    sh = torch.cat([dc[:, None, :], torch.tensor(s.sh, device=dev)], 1)   # (N, 4, 3)
    # Spark's PackedSplats (16 bytes a splat, read back in variants.py pack_check): colour clipped
    # to [0, 1] and 8-bit, centres float16, ln scale 8-bit over [-12, 9].
    col01 = torch.clamp(torch.tensor(s.colors, device=dev), 0, 1)
    col01 = torch.round(col01 * 255) / 255
    dc01 = (col01 - 0.5) / spz.SH_C0
    sh01 = torch.cat([dc01[:, None, :], torch.tensor(s.sh, device=dev)], 1)
    means16 = means.half().float()
    lns = torch.log(scales).clamp(-12, 9)
    scales8 = torch.exp(torch.round((lns + 12) / 21 * 254) / 254 * 21 - 12)
    sets = {"base": (means, scales, sh), "clip01": (means, scales, sh01),
            "sparkpack": (means16, scales8, sh01)}
    out = []
    for cam in cams:
        vm = torch.tensor(cam["viewmat"], dtype=torch.float32, device=dev)[None]
        K = torch.tensor(cam["K"], dtype=torch.float32, device=dev)[None]
        res = {}
        for name, deg, mode, which in MODES:
            m_, s_, h_ = sets[which]
            col, alpha, _ = rasterization(m_, quats, s_, opac, h_[:, :(deg + 1) ** 2], vm, K,
                                          cam["w"], cam["h"], sh_degree=deg, rasterize_mode=mode)
            rgba = torch.cat([col[0].clamp(0, 1), alpha[0]], -1).cpu().numpy()
            buf = io.BytesIO()
            Image.fromarray((rgba * 255 + 0.5).astype(np.uint8)).save(buf, "PNG")
            res[name] = buf.getvalue()
        out.append(res)
    return out


@app.local_entrypoint()
def main():
    import numpy as np
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from litereality_agent.pipeline.evaluate import fidelity as fd
    scan = Path("~/LiteReality-scans/stray_atrium_q600").expanduser()
    frames = json.loads((HERE / "variants_9views.json").read_text())["setup"]["frames"]
    views = {v.index: v for v in fd.capture_views(scan)}
    cams = []
    for f in frames:
        v = views[f]
        fx, fy, cx, cy = fd.intrinsics_at(v, (960, 720))
        c2w = v.pose @ np.diag([1.0, -1.0, -1.0, 1.0])     # OpenGL camera -> OpenCV camera
        cams.append({"frame": f, "w": 960, "h": 720, "viewmat": np.linalg.inv(c2w).tolist(),
                     "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]})
    data = (REPO / "run/atrium_stray_q/v2/raw.spz.bin").read_bytes()
    results = render.remote(data, cams)
    outdir = HERE / "renders"
    for cam, res in zip(cams, results):
        for name, png in res.items():
            (outdir / f"gsplat_{name}_{cam['frame']:05d}.png").write_bytes(png)
    print("saved", len(results), "views")
