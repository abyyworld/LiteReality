"""Named trainer settings for the photoreal room (deploy/modal/splat/app.py's options).

A preset is a set of trainer options, applied over the trainer's defaults and under anything given
beside it (``{"preset": name, **given}``). The defaults are left as they were, so an unflagged run
(and any other caller of the deployed trainer) behaves as before; a preset changes only what it
names. Pure Python: the Modal trainer imports this module too.

Every ``q_*`` preset fixes what results/diagnosis/ found in the served atrium:

* SH degree 3 kept in the export (``export_sh_degree`` 3): the trainer fits degree 3 and the
  export used to keep 1, which threw away the texture held in the higher bands (3-10 dB at
  training views). SPZ stores degree 3 and Spark 2.3.1 draws it (``maxSh`` defaults to 3); in a
  real SH3 room it adds ~4.4 bytes a splat after gzip (16.0 -> 20.4 B a splat at 1M splats).
* a splat budget that grows: ``cap_max`` 2.4M with the seeds held to half of it by the pipeline
  (`pipeline.splatroom.SEED_FRACTION`), so MCMC adds splats where the photos need them instead of
  starting full; 10000 steps of 4 photos. ~50 MB of SPZ at SH3.
* colour correction that cannot hide in the per-photo model. The per-photo bilateral grid
  (16 x 16 x 8 cells, a 3 x 4 affine colour map per cell) may fit any local colour a photo shows,
  and the export, which carries no per-photo correction, then drops it. Three ways out:

      q_sh3_nogrid      no per-photo colour model at all
      q_sh3_affine      one global 3 x 4 affine colour map per photo (exposure and white balance,
                        no spatial term: a 1 x 1 x 1 grid), its mean over the photos pulled to
                        the identity so the splats keep the average photo's colours
      q_sh3_bilagrid_tv the 16 x 16 x 8 grid with 100x gsplat's smoothness weight and the same
                        pull of its mean to the identity, so what all photos share stays in the
                        splats and only per-photo drift is corrected
"""

from __future__ import annotations

# The pieces every q_* preset shares.
_QUALITY = {
    "export_sh_degree": 3,
    "cap_max": 2_400_000,
    "steps": 10_000,
    "batch_size": 4,
}

PRESETS: dict[str, dict] = {
    "q_sh3_nogrid": {
        **_QUALITY,
        "bilateral_grid": False,
        "fused_bilagrid": False,
    },
    "q_sh3_affine": {
        **_QUALITY,
        # gsplat's own (unfused) grid at 1 x 1 x 1 is one 3 x 4 affine map per photo; the fused
        # kernel is not relied on at that size. No total variation: a single cell has none (and
        # gsplat's TV of a 1-cell axis is 0 / 0).
        "bilateral_grid": True,
        "fused_bilagrid": False,
        "bilateral_grid_shape": [1, 1, 1],
        "bilateral_tv_lambda": 0.0,
        "bilateral_mean_lambda": 10.0,
    },
    "q_sh3_bilagrid_tv": {
        **_QUALITY,
        "bilateral_grid": True,
        "fused_bilagrid": True,
        "bilateral_tv_lambda": 1000.0,
        "bilateral_mean_lambda": 10.0,
    },
}


def names() -> list[str]:
    return sorted(PRESETS)


def resolve(options: dict | None) -> dict:
    """`options` with its ``"preset"`` (if any) expanded: the preset's settings, then whatever
    `options` itself gives (that wins). The name is kept under ``"preset"`` for the report.
    ValueError for an unknown preset."""
    options = dict(options or {})
    name = options.get("preset")
    if name in (None, ""):
        options.pop("preset", None)
        return options
    if name not in PRESETS:
        raise ValueError(f"unknown trainer preset {name!r} (known: {', '.join(names())})")
    return {**PRESETS[name], **options}
