# scripts/

Standalone capture tooling. Nothing here is imported by the package — these are entry points you
run directly. All of them resolve the repo root themselves, so they work from any working
directory.

The supported entrypoint is `uv run litereality`. Files here are maintenance and batch wrappers,
not importable application code. Use `uv run litereality stage <name> <scene>` for one stage.

## Capture utilities

Fuse the raw RoomPlan capture into geometry for render-vs-real comparison.

- `capture/build_meshes.py` — per-scene TSDF mesh from RGB + depth + poses
- `capture/build_pointclouds.py` — per-scene fused colored point cloud
- `capture/colorize_pcd.py` — colorize the scanner's point cloud from RGB frames
- `capture/shrink_scan.py` — reduce capture size without changing pipeline semantics
