"""RESEARCH ONLY. The Difix3D-style progressive update inside our gsplat trainer (patch_difix.py).

`DifixLoop(runner, cfg)` is made once the trainer has its data and splats. Each step calls
`maybe_fix(step)` (at a round's step: render that round's novel poses, repair each with NVIDIA's
Difix, keep the repaired views as the novel set, replacing the last round's) and `sample()` (a
batch of the novel set, with probability `cfg.difix_prob`, else None).

Our own code; only the Difix pipeline (``src.pipeline_difix.DifixPipeline`` from a clone of
github.com/nv-tlabs/Difix3D, weights nvidia/difix_ref) is NVIDIA's, under its research /
evaluation-only licence. Nothing here is part of the hosted service.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

DIFIX_REPO = os.environ.get("DIFIX_REPO", "/opt/difix3d")
PROMPT = "remove degradation"
TIMESTEP = 199
ALPHA_MIN = 0.5


def load_pipeline(device: str = "cuda"):
    """NVIDIA's difix_ref pipeline (needs the Difix3D clone on the path)."""
    if DIFIX_REPO not in sys.path:
        sys.path.insert(0, DIFIX_REPO)
    from src.pipeline_difix import DifixPipeline

    pipe = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)


def repair(pipe, image, ref, width: int = 0):
    """One render (PIL) repaired by Difix with `ref` (PIL) as reference, at `width` (0 = the
    render's own; either way rounded down to a multiple of 8), returned at the render's size."""
    from PIL import Image

    w0, h0 = image.size
    w = (width or w0) // 8 * 8
    h = int(round(h0 * w / w0)) // 8 * 8
    img = image.resize((w, h), Image.LANCZOS) if (w, h) != (w0, h0) else image
    rf = ref.resize((w, h), Image.LANCZOS) if ref.size != (w, h) else ref
    out = pipe(prompt=PROMPT, image=img, ref_image=rf, num_inference_steps=1,
               timesteps=[TIMESTEP], guidance_scale=0.0).images[0]
    return out.resize((w0, h0), Image.LANCZOS) if out.size != (w0, h0) else out


def nearest(train_c2w: np.ndarray, c2w: np.ndarray) -> int:
    """The training camera nearest a pose: position distance plus 0.5 m per radian of the angle
    between the viewing directions (OpenCV cameras look along +z)."""
    d = np.linalg.norm(train_c2w[:, :3, 3] - c2w[:3, 3], axis=1)
    cos = np.clip(train_c2w[:, :3, 2] @ c2w[:3, 2], -1.0, 1.0)
    return int(np.argmin(d + 0.5 * np.arccos(cos)))


class DifixLoop:
    def __init__(self, runner, cfg) -> None:
        self.runner, self.cfg = runner, cfg
        self.poses = np.load(cfg.difix_poses).astype(np.float32)        # [R, V, 4, 4]
        self.steps = [int(s) for s in cfg.difix_steps]
        if self.poses.ndim != 4 or self.poses.shape[0] != len(self.steps):
            raise ValueError(f"{cfg.difix_poses}: {self.poses.shape} poses for "
                             f"{len(self.steps)} steps")
        parser = runner.parser
        idx = np.asarray(runner.trainset.indices)
        self.train_c2w = np.asarray(parser.camtoworlds)[idx]
        self.train_paths = [parser.image_paths[i] for i in idx]
        cam = parser.camera_ids[idx[0]]
        self.K = np.asarray(parser.Ks_dict[cam], dtype=np.float32)
        self.width, self.height = parser.imsize_dict[cam]
        self.out = Path(cfg.result_dir) / "difix"
        self.out.mkdir(parents=True, exist_ok=True)
        self.rng = random.Random(1234)
        self.pipe = None
        self.novel = None             # (c2w [V,4,4], images uint8 [V,H,W,3], masks bool [V,H,W])
        self.log: list[dict] = []
        print(f"[difix] {self.poses.shape[1]} novel views x {len(self.steps)} rounds at steps "
              f"{self.steps}; {self.width}x{self.height}; prob {cfg.difix_prob}, weight "
              f"{cfg.difix_lambda}, Difix width {cfg.difix_width or 'native'}", flush=True)

    @torch.no_grad()
    def maybe_fix(self, step: int) -> None:
        if step not in self.steps:
            return
        from PIL import Image

        r = self.steps.index(step)
        t0 = time.perf_counter()
        if self.pipe is None:
            self.pipe = load_pipeline()
        runner, cfg = self.runner, self.cfg
        dev = runner.device
        K = torch.from_numpy(self.K)[None].to(dev)
        keep_c2w, images, masks, coverage = [], [], [], []
        rdir = self.out / f"round{r}_step{step}"
        rdir.mkdir(parents=True, exist_ok=True)
        for v, c2w in enumerate(self.poses[r]):
            colors, alphas, _ = runner.rasterize_splats(
                camtoworlds=torch.from_numpy(c2w)[None].to(dev), Ks=K, width=self.width,
                height=self.height, sh_degree=cfg.sh_degree, near_plane=cfg.near_plane,
                far_plane=cfg.far_plane)
            rgb = (colors[0, ..., :3].clamp(0, 1) * 255).byte().cpu().numpy()
            mask = (alphas[0, ..., 0] > ALPHA_MIN).cpu().numpy()
            render = Image.fromarray(rgb)
            ref = Image.open(self.train_paths[nearest(self.train_c2w, c2w)]).convert("RGB")
            fixed = repair(self.pipe, render, ref, cfg.difix_width)
            keep_c2w.append(c2w)
            images.append(np.asarray(fixed, dtype=np.uint8))
            masks.append(mask)
            coverage.append(float(mask.mean()))
            if v % 4 == 0:                       # a quarter of them kept to look at
                render.save(rdir / f"{v:03d}_render.jpg", quality=90)
                fixed.save(rdir / f"{v:03d}_fixed.jpg", quality=90)
                ref.resize(render.size).save(rdir / f"{v:03d}_ref.jpg", quality=80)
        self.novel = (np.stack(keep_c2w), np.stack(images), np.stack(masks))
        entry = {"round": r, "step": step, "views": len(images),
                 "seconds": round(time.perf_counter() - t0, 1),
                 "coverage_mean": round(float(np.mean(coverage)), 3),
                 "coverage_min": round(float(np.min(coverage)), 3),
                 "gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
        self.log.append(entry)
        (self.out / "rounds.json").write_text(json.dumps(self.log, indent=1))
        print(f"[difix] {json.dumps(entry)}", flush=True)
        torch.cuda.empty_cache()

    def sample(self):
        if self.novel is None or self.rng.random() >= self.cfg.difix_prob:
            return None
        c2w, images, masks = self.novel
        pick = [self.rng.randrange(len(images)) for _ in range(self.cfg.batch_size)]
        m = torch.from_numpy(masks[pick])
        img = torch.from_numpy(images[pick]).float() * m[..., None]   # loss only where splats are
        return {"camtoworld": torch.from_numpy(c2w[pick]),
                "K": torch.from_numpy(self.K)[None].expand(len(pick), 3, 3).contiguous(),
                "image": img, "image_id": torch.zeros(len(pick), dtype=torch.long),
                "mask": m}
