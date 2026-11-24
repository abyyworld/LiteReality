"""results/stray_quality/score.py's sharpness metric and score naming (no renderer, no GPU).

The metric must read blur and only blur: 1 for the photo itself, falling as the photo is blurred
(also by 1-2 px at 1920, which 480 x 360 PSNR hardly sees), and not lowered by a shift (pose error
is PSNR's business, not sharpness's)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCORE = Path(__file__).resolve().parents[1] / "results" / "stray_quality" / "score.py"


@pytest.fixture(scope="module")
def score():
    spec = importlib.util.spec_from_file_location("stray_quality_score", SCORE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["stray_quality_score"] = module
    spec.loader.exec_module(module)
    return module


def _photo(h=360, w=480, seed=0) -> np.ndarray:
    """A natural-ish image: 1/f noise, colour, in [0, 1]."""
    rng = np.random.default_rng(seed)
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    amp = 1.0 / np.maximum(np.hypot(fy, fx), 1.0 / max(h, w))
    out = []
    for _ in range(3):
        phase = np.exp(2j * np.pi * rng.random((h, w)))
        img = np.real(np.fft.ifft2(amp * phase))
        out.append((img - img.min()) / (img.max() - img.min()))
    return np.stack(out, -1)


def _blur(img: np.ndarray, sigma: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    return np.stack([gaussian_filter(img[..., c], sigma) for c in range(3)], -1)


def test_the_photo_itself_is_fully_sharp(score):
    photo = _photo()
    s = score.sharpness(photo, photo)
    assert s["hf_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert s["lap_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert s["half_power_cyc_px1920"] is None


def test_blur_lowers_the_high_frequency_ratio_in_order(score):
    photo = _photo()
    ratios = [score.sharpness(_blur(photo, s), photo)["hf_ratio"] for s in (0.5, 1.0, 2.0)]
    assert 1.0 > ratios[0] > ratios[1] > ratios[2] > 0.0
    assert ratios[1] < 0.5                       # sigma 1 px already loses most of the top band
    half = score.sharpness(_blur(photo, 2.0), photo)["half_power_cyc_px1920"]
    assert half is not None and half < 0.25 * photo.shape[1] / 1920.0 * 4


def test_a_shift_or_a_colour_change_is_not_read_as_blur(score):
    photo = _photo(seed=1)
    shifted = np.roll(photo, (3, -5), axis=(0, 1))
    assert score.sharpness(shifted, photo)["hf_ratio"] == pytest.approx(1.0, abs=0.05)
    dimmer = np.clip(0.7 * photo + 0.1, 0, 1)
    assert score.sharpness(dimmer, photo)["hf_ratio"] == pytest.approx(1.0, abs=0.02)


def test_scores_are_named_by_set_and_size_and_the_old_name_is_kept(score):
    assert score.score_name("v2") == "v2"
    assert score.score_name("v2", "random") == "v2_random"
    assert score.score_name("v2", "sharp", full=True) == "v2_sharp_full"
    assert score.score_name("v4", "random", full=True) == "v4_random_full"
    assert set(score.EVAL_SETS) == {"sharp", "random"}
    assert score._spread(list(range(29)), 3) == [0, 14, 28]
    assert score._spread([1, 2], None) == [1, 2]


def test_box_downscale_averages_blocks(score):
    img = np.arange(16, dtype=float).reshape(4, 4, 1).repeat(3, -1)
    small = score._box(img, (2, 2))
    assert small.shape == (2, 2, 3)
    assert small[0, 0, 0] == pytest.approx(np.mean([0, 1, 4, 5]))
