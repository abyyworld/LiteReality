# fix_v4: fixes for the blurry atrium and the doubled chairs

The fixes made after `results/diagnosis/` (code in commit 9dc18c7): new trainer presets in
`deploy/modal/splat/app.py` (no bilateral grid, one affine colour map a photo, SH 3 kept, a
splat budget that grows), the MCMC cap/growth fix in `pipeline/splatroom/__init__.py`
(`dense_points_for`), a new Stray frame picker in `models/capture/stray.py`, and the chair mask
vote and split fixes in `pipeline/splatroom/masks.py` and `split.py` (~94% per chair).

- `chairs/`: the chair cut before (v2) and after (`run/atrium_stray_q/v2_chairs`): overlays,
  truth table, move test, physics check.
- `e2e_affine/`: end-to-end smoke run with the affine colour preset (small body splats, reports).
- `retrain/`: the 10000-step retrain logs; it died three times near step 1250 when the local
  Modal client lost its connection.
- `scores/`: the step-1250 snapshot (`run/atrium_stray_q/v4_a1_nogrid`) scores PSNR 19.62 dB vs
  v2's 21.70 dB on full-res random views (`scores/score_all.log`).
- `smoke/`: 300-step smoke trainings of each q_* preset (events and stats).
  `score_smoke/`: a first scoring of v2.

Big files (over 5 MB) are kept out of git, in `~/LiteReality-scans/fix_v4/`:
`e2e_affine/raw.spz.bin`, `e2e_affine/room-31336cffd79b252d.spz`,
`e2e_affine/room-d2a4e2b10e4610f0.spz`.
