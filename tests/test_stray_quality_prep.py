"""The per-run overrides used to compare splat variants fairly: import-stray's held-out frames and
near-duplicate thresholds, splatroom's training width and raw trainer options."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_quickroom import _frame_looking_down  # noqa: E402

from litereality_agent.models.splat import dataset, rgbd  # noqa: E402
from litereality_agent.pipeline.capture import excluded_mask, read_frame_list  # noqa: E402


@pytest.fixture
def scan(tmp_path):
    folder = tmp_path / "scan"
    folder.mkdir()
    _frame_looking_down(folder)
    return folder


def test_read_frame_list(tmp_path):
    path = tmp_path / "eval.txt"
    path.write_text("# held out\n001234  pool frame_00012\n\n  56 # note\n", encoding="utf-8")
    assert read_frame_list(path) == [1234, 56]


def test_excluded_mask_spans_seconds():
    listed = [SimpleNamespace(row=r) for r in range(10)]
    times = np.arange(10) / 10.0
    assert not excluded_mask(listed, times, None).any()
    assert excluded_mask(listed, times, [4]).tolist() == [r == 4 for r in range(10)]
    held = excluded_mask(listed, times, [4, 99], span=0.15)
    assert np.flatnonzero(held).tolist() == [3, 4, 5]


def test_train_width_override(scan, monkeypatch):
    frames = rgbd.load_frames(scan)
    assert dataset.build(frames, width=160, points_per_frame=300).size == (160, 120)
    monkeypatch.setattr(dataset, "train_width_override", 320)
    assert dataset.build(frames, points_per_frame=300).size == (320, 240)
    assert dataset.build(frames, width=160, points_per_frame=300).size == (160, 120)


def test_cli_flags_parse():
    from litereality_agent.cli import _parser

    parser = _parser()
    a = parser.parse_args(["import-stray", "rec", "--out", "o", "--min-move", "0.02",
                           "--min-turn", "1", "--exclude", "e.txt", "--exclude-span", "0.15"])
    assert (a.min_move, a.min_turn, a.exclude, a.exclude_span) == (0.02, 1.0, "e.txt", 0.15)
    b = parser.parse_args(["splatroom", "scan", "--train-width", "1920",
                           "--trainer-option", "sh_degree=1", "--trainer-option", "batch_size=2"])
    assert b.train_width == 1920 and b.trainer_option == ["sh_degree=1", "batch_size=2"]
