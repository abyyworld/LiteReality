"""Higher SH in the exported splat: kept to degree 3 (the q_* presets) and turned with a moved
splat (`spz.rotate_sh`), so a body's view-dependent colour stays where it was fitted."""

from __future__ import annotations

import numpy as np
import pytest

from litereality_agent.models.splat import ply, spz


def _rotation(seed: int) -> np.ndarray:
    q = np.random.default_rng(seed).normal(size=4)
    return spz.quat_to_matrices(q / np.linalg.norm(q))


def _dirs(n: int, seed: int) -> np.ndarray:
    d = np.random.default_rng(seed).normal(size=(n, 3))
    return d / np.linalg.norm(d, axis=1, keepdims=True)


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_rotated_sh_show_the_same_colour_from_the_turned_direction(degree):
    rng = np.random.default_rng(degree)
    sh = rng.normal(scale=0.2, size=(50, spz.sh_dim(degree), 3)).astype(np.float32)
    R = _rotation(degree)
    turned = spz.rotate_sh(sh, R)
    d = _dirs(200, 7)
    before = np.einsum("mk,nkc->nmc", spz.sh_basis(d, degree), sh)          # f(d)
    after = np.einsum("mk,nkc->nmc", spz.sh_basis(d @ R.T, degree), turned)  # f'(R d)
    assert np.allclose(after, before, atol=1e-5)
    assert np.allclose(spz.rotate_sh(turned, R.T), sh, atol=1e-5)
    assert np.allclose(spz.rotate_sh(sh, np.eye(3)), sh, atol=1e-6)


def test_the_basis_is_3dgs_degree_one_and_orthogonal_per_band():
    d = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    assert np.allclose(spz.sh_basis(d, 1), -spz.SH_C1 * np.array([[0, 0, 1], [1, 0, 0],
                                                                    [0, -1, 0]]))
    # Orthogonal bands (Monte Carlo over the sphere): the rotation fit is exact band by band.
    y = spz.sh_basis(_dirs(200_000, 3), 3)
    gram = y.T @ y * (4 * np.pi / len(y))
    assert np.allclose(gram, np.eye(15), atol=0.03)


def test_transformed_turns_sh_and_back():
    rng = np.random.default_rng(4)
    n = 30
    s = spz.Splats(rng.normal(size=(n, 3)).astype(np.float32),
                   np.full((n, 3), 0.01, np.float32),
                   np.tile(np.array([0, 0, 0, 1.0], np.float32), (n, 1)),
                   rng.uniform(0, 1, (n, 3)).astype(np.float32),
                   rng.uniform(0, 1, n).astype(np.float32),
                   rng.normal(scale=0.1, size=(n, 15, 3)).astype(np.float32))
    R = _rotation(11)
    moved = s.transformed(R, np.array([1.0, 2.0, 3.0]))
    assert not np.allclose(moved.sh, s.sh)
    back = moved.transformed(R.T, -R.T @ np.array([1.0, 2.0, 3.0]))
    assert np.allclose(back.positions, s.positions, atol=1e-5)
    assert np.allclose(back.sh, s.sh, atol=1e-5)


def test_sh3_survives_ply_and_spz_at_about_4_bytes_a_splat_more():
    rng = np.random.default_rng(5)
    n = 20_000
    sh = (rng.laplace(scale=0.03, size=(n, 15, 3))).astype(np.float32)
    s = spz.Splats(rng.uniform(-2, 2, (n, 3)).astype(np.float32),
                   np.exp(rng.normal(-5, 1, (n, 3))).astype(np.float32),
                   np.tile(np.array([0, 0, 0, 1.0], np.float32), (n, 1)),
                   rng.uniform(0, 1, (n, 3)).astype(np.float32),
                   rng.uniform(0, 1, n).astype(np.float32), sh)
    data = ply.write(s)
    sh3 = spz.encode(ply.read(data, max_sh_degree=3))
    sh1 = spz.encode(ply.read(data, max_sh_degree=1))
    back = spz.decode(sh3)
    assert back.sh_degree == 3 and spz.decode(sh1).sh_degree == 1
    # Degree 1 at 5 bits (steps of 1/16), degrees 2-3 at 4 bits (1/8), after rounding to 1/128.
    assert np.abs(back.sh[:, :3] - sh[:, :3]).max() <= 1 / 32 + 1 / 256 + 1e-6
    assert np.abs(back.sh[:, 3:] - sh[:, 3:]).max() <= 1 / 16 + 1 / 256 + 1e-6
    assert len(sh3) > len(sh1)
