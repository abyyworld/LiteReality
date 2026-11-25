"""The Difix research scripts (scripts/difix, research only): the trainer patch applies to the
real gsplat v1.5.3 trainer after our own, and the novel poses are rolled as the scan's photos and
reach their targets in 0.5 m steps."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_splat_trainer_patch import patch_trainer, upstream_trainer  # noqa: E402,F401

DIFIX = Path(__file__).resolve().parents[1] / "scripts" / "difix"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


patch_difix = _load("litereality_patch_difix", DIFIX / "patch_difix.py")
poses = _load("litereality_difix_poses", DIFIX / "poses.py")


def test_the_difix_patch_applies_after_ours_and_guards_every_photo_only_step(
        upstream_trainer):  # noqa: F811 (the fixture)
    ours = patch_trainer.patch(upstream_trainer)
    out = patch_difix.patch(ours)
    # every use of the photo-only inputs is skipped on a novel-view step
    assert "if cfg.depth_loss and not is_novel:\n                points = data" in out
    assert "if cfg.use_bilateral_grid and not is_novel:\n                grid_y" in out
    assert "if cfg.depth_loss and not is_novel:\n                # query depths" in out
    assert "loss = loss * cfg.difix_lambda" in out
    with pytest.raises(ValueError):
        patch_difix.patch(out)                              # twice
    with pytest.raises(ValueError):
        patch_difix.patch(upstream_trainer)                 # without ours first


def test_a_novel_view_is_rolled_as_the_photos_and_looks_where_asked():
    c2w = poses.look_at([0.0, 0.0, 0.0], [1.0, 1.0, -0.2])
    r = c2w[:3, :3]
    assert np.allclose(r.T @ r, np.eye(3), atol=1e-9) and np.linalg.det(r) > 0
    fwd = np.array([1.0, 1.0, -0.2]) / np.linalg.norm([1.0, 1.0, -0.2])
    assert np.allclose(r[:, 2], fwd)
    assert r[2, 0] < -0.95                                  # image x points at the floor


def test_a_pose_moves_half_a_metre_a_round_and_turns_in_step():
    start = poses.look_at([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    goal = poses.look_at([2.0, 0.0, 0.0], [2.0, 1.0, 0.0])
    half = poses.toward(start, goal, 0.25)
    assert np.allclose(half[:3, 3], [0.5, 0.0, 0.0])
    cos = half[:3, 2] @ goal[:3, 2]
    assert np.isclose(np.degrees(np.arccos(cos)), 90 * 0.75, atol=0.5)
    assert np.allclose(poses.toward(start, goal, 1.0), goal, atol=1e-9)
