# RealityScan by hand: align the atrium frames yourself, then import the poses

**When you need this:** our headless RealityScan 2.2 run on Modal already works with no Epic login on
our side (see `results/realityscan/README.md`). Its weak point is coverage. On default settings,
RealityScan splits the 400 atrium frames into 23 components, and the largest has only 60 frames.

The two things it cannot do headless are:

- adding control points by hand to join components;
- running the Windows GUI build, with real CUDA.

This page covers both. The importer takes any COLMAP export of these frames, so plain COLMAP works too.

**Licence:** RealityScan's EULA forbids offering it as a hosted service. Use it for paper experiments
only, and never in the room call or the served pipeline. It is free for education, non-commercial use
and companies under $1M revenue.

## 1. Get the input frames (on this Mac)

```bash
cd ~/LiteReality-scans/rs_stray/rs_input && zip -qr ~/Desktop/atrium_rs_input.zip images selection.json
```

- That folder has 400 photos (`frame_XXXXX.jpg`, 1920x1440).
- The 29 fixed eval frames are included, so they get RealityScan poses. The importer marks them as
  `"eval": true`.
- `build_dataset.py` keeps them out of training. Do not remove or rename any photo.

## 2. Install RealityScan (you, once)

- **Windows:**
  1. Install the Epic Games Launcher.
  2. Sign in with your Epic account.
  3. Install RealityScan 2.2 or newer from Library / Unreal Engine.
  4. The first start asks you to accept the EULA. That acceptance must be yours.
- **Linux:** use the `.deb` from https://www.realityscan.com/en-US/linux (Epic sign-in). We already
  hold 2.2 on the private Modal Volume `litereality-tools`, so you only need this route to run the GUI
  locally.
- **GPU:** an NVIDIA GPU with 8 GB or more is needed for meshing. Alignment alone runs on CPU.

## 3. Align (RealityScan GUI)

Menu names below are as in RealityScan 2.x and may differ slightly between versions.

1. **New project.** Workflow, then Inputs/Folder: add the `images` folder.
2. **One camera for all photos.** Select all images (Ctrl+A in the 1Ds panel). In the selection
   inspector, set:
   - Calibration group = 0
   - Lens group = 0
   - Distortion model = Brown3
   - Prior calibration = Approximate, with focal length 25 mm (35 mm equivalent; ARKit fx 1335 px at 1920 px wide)
3. **Settings.** Alignment settings: Image overlap = Low, Detector sensitivity = High. Click
   **Align Images**.
4. **Join the components.**
   - Open Components (1Ds panel) and note the sizes.
   - For each pair of large components, use the Control Points tool. Place at least 3 control points,
     each seen in at least 2 photos of each component. Good spots are door frame corners, the desk
     edge and chair legs.
   - Click **Align Images** again.
   - Repeat until one component holds most of the 400 photos, and check it before exporting:
     - Walls must be straight, and the floor must be one plane.
     - The camera trail must look like the walk: no folded or mirrored piece.
     - A wrong merge is worse than a gap. Our "Ultra / Low overlap" run merged wrongly (6–11 degrees off).
5. **Shortcut:** our Modal run saved `~/LiteReality-scans/rs_stray/rs_out_default/scene.rsproj` and the
   `scene/` folder next to it. You can open that project, and when RealityScan asks, point it at the
   `images` folder. You then start from the 23 components we already have.

## 4. Export COLMAP

1. Select the component to export (the largest, merged one).
2. Choose Alignment, then Export, then **Registration**, and pick **COLMAP**. Set:
   - Export images = **No**
   - Undistort images = **No**
   - File type: text or binary (both import)
   - Layout: standard (`sparse/0`) or flat (both import)
3. If RealityScan asks "export only the selected component?", answer **Yes**. If it asks whether
   "FULL_OPENCV" is OK, answer **Yes**.
4. Copy the export folder to the Mac at `~/LiteReality-scans/_vendor/rs_manual_export/`. This is
   outside git.
5. Also save the project (`.rsproj`) and copy it there. That makes it possible to re-export later.

## 5. Import, compare with ARKit, train, score (on this Mac; Claude can run these)

```bash
cd ~/LiteReality
# COLMAP (txt/bin) -> our pose format + Sim(3) onto ARKit + pose differences
uv run --frozen python scripts/compare_trainers/import_colmap_poses.py \
    ~/LiteReality-scans/_vendor/rs_manual_export --out ~/LiteReality-scans/rs_manual
# full pose report (adds 1 s relative pose error and intrinsics) -> results/realityscan/stray/pose_vs_arkit_manual.json
uv run --frozen python scripts/realityscan/stray_poses.py --scan ~/LiteReality-scans/rs_manual --tag _manual
# training set in our convention (eval frames left out), plus the eval frames at RS poses
uv run --frozen python scripts/realityscan/build_dataset.py --scan ~/LiteReality-scans/rs_manual --long 960 --eval-rs
uv run --frozen python scripts/realityscan/build_dataset.py --scan ~/LiteReality-scans/rs_manual --long 960 --poses arkit
# gsplat 5k on Modal H100 (about $0.06-0.2), RS poses and the ARKit control
uv run --frozen --env-file .env python scripts/realityscan/train_rs.py rs_manual_gs ~/LiteReality-scans/rs_manual/rs_w960.tar.gz --trainer gsplat
uv run --frozen --env-file .env python scripts/realityscan/train_rs.py arkit_manual_gs ~/LiteReality-scans/rs_manual/arkit_w960.tar.gz --trainer gsplat
# score on the 29 fixed eval frames (shared scorer, unchanged)
uv run --frozen --with playwright --with lpips==0.1.4 python scripts/realityscan/score_rs.py eval rs_manual_gs run/realityscan/rs_manual_gs
uv run --frozen --with playwright --with lpips==0.1.4 python scripts/realityscan/score_rs.py eval rs_manual_gs_rspose run/realityscan/rs_manual_gs --poses rs --eval-rs ~/LiteReality-scans/rs_manual/eval_rs
```

### What `import_colmap_poses.py` writes to `--out`

- **`poses/frame_XXXXX.json`:** our format, one file per registered frame.
  - `cameraPoseARFrame`: RealityScan's pose moved into the ARKit (pool) world. Camera to world,
    ARKit/OpenGL camera axes, row-major 4x4, the same as the import's `frame_*.json`.
  - `intrinsics`: RealityScan's K at the photo size, in ARKit's pixel convention.
  - `colmap_camera`: the lens model and its parameters.
  - `eval`: whether the frame is one of the 29 eval frames.
- **`pose_vs_arkit.json`:** per-frame and median centre error (m) and rotation error (degrees), and
  focal length against ARKit.
- **`rs_to_arkit.json`:** the Sim(3), fitted robustly on camera centres plus points on each camera's axes.
- **`rs_out/colmap/sparse/0`:** the model as COLMAP text.
- **`rs_input`:** a symlink to the photos. Steps 2 to 4 of the block above need it.

### Checked on our own exports

- The Modal text export (60 frames) gives 11 mm median centre error, 1.9 degrees rotation error and
  fx −0.36%. This matches `results/realityscan/README.md`.
- A binary COLMAP model (55 frames) gives 10.7 mm and 1.9 degrees.

### What to look for

- **Coverage:** `registered` out of 400, and `eval_registered` out of 29.
- **Pose error vs ARKit:** a merged component should still be about 1–2 cm and 1–2 degrees. Much
  larger values mean a wrong merge.
- **The comparison that counts:** the 29-eval score of the RS-pose gsplat against gsplat_v0 at
  19.15 / 0.738 / 0.286. This comparison is only meaningful if close to all 29 eval frames are
  registered.
