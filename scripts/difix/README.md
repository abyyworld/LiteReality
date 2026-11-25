# scripts/difix: NVIDIA Difix3D progressive update (RESEARCH ONLY)

Difix (github.com/nv-tlabs/Difix3D, weights nvidia/difix_ref) is licensed for research and evaluation
only. Nothing here goes into the deployed splat app, and no room trained with it may be served by the
hosted room call. The Modal image clones Difix at build time, so none of NVIDIA's code is in this repo.

1. Rebuild the live room's training set:

   ```
   uv run --env-file .env python scripts/difix/prepare.py
   ```

2. Write the room's layout to `~/LiteReality-scans/difix/poses/layout.json` (see `results/difix`),
   then make the novel poses:

   ```
   uv run python scripts/difix/poses.py DATASET_TAR BUNDLE OUT
   ```

3. Upload `dataset.tar`, `options.json`, `novel_poses.npy` and `live_v2.spz` to volume
   `litereality-difix` under `/atrium`.

4. Run the smoke test, then the two arms:

   ```
   MODAL_IMAGE_BUILDER_VERSION=2025.06 uv run --env-file .env modal run --detach --env main scripts/difix/train_difix.py --mode smoke
   MODAL_IMAGE_BUILDER_VERSION=2025.06 uv run --env-file .env modal run --detach --env main scripts/difix/train_difix.py --mode arms
   ```

Results: `results/difix/README.md`.
