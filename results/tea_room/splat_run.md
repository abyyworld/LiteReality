# tea_room: the photoreal room as Gaussian splats (2026-11-13)

```bash
uv run modal deploy --env main deploy/modal/splat/app.py
uv run litereality splatroom ~/LiteReality-Agent/example-scans/tea_room
uv run litereality splatroom ~/LiteReality-Agent/example-scans/tea_room --publish --upload-only
```

Input: the tea_room example scan, 62 photos at 1920×1440 with ARKit poses and 256×192 LiDAR
depth. Physics: `run/tea_room/objects/scene.xml`. Training: gsplat 1.5.3's `simple_trainer.py
mcmc` on one Modal H100 with these settings:
- 7000 steps, 960×720 photos;
- LiDAR depth loss (λ 0.01);
- a bilateral grid per photo;
- antialiased rasterisation;
- MCMC capped at 800k splats;
- SH degree 3 during training, degree 1 in the SPZ files.

## Timeline (from pressing enter on the laptop)

| At (s) | What | Splats | SPZ MB |
|---|---|---|---|
| 10.3 | instant room (photos + LiDAR fused, no GPU), cut into room + 5 bodies | 639,815 | 6.2 |
| 12.8 | training set written (62 photos, 248k LiDAR points, 28 MB) | | |
| 29.4 | trainer up on an H100 | | |
| 49.8 | step 300 snapshot | 248,000 | 3.8 |
| 61.8 | step 1000 | 301,445 | 4.8 |
| 81.3 | step 2000 | 491,017 | 8.2 |
| 112.8 | step 3500 | 800,000 | 14.0 |
| 141.9 | step 5000 | 800,000 | 14.4 |
| 184.3 | step 7000 (final), held-out metrics | 800,000 | 14.6 |

Training itself took 146 s. Each snapshot took 0.7–5 s on the laptop to cut against the physics
and write.

The step-300 and step-1000 snapshots look worse in the call than the instant room: speckle and
holes. The trainer now sends its first snapshot at step 2000.

## Image quality on held-out photos (every 8th photo, 8 of 62, never trained on)

| PSNR | SSIM | LPIPS (AlexNet) | PSNR after colour correction | SSIM after | LPIPS after |
|---|---|---|---|---|---|
| 19.17 dB | 0.778 | 0.359 | 21.63 dB | 0.799 | 0.338 |

For comparison, the quick room's photo walls scored +2.5–2.7 dB on walls/floor in `litereality
evaluate`, but against a mesh baseline, and that is a different metric. The two are not directly
comparable yet: `evaluate` should render the splat from the same held-out poses.

62 photos is a sparse capture: a room scanned slowly, walking around the furniture, will train
better. The held-out frames are also the hardest ones (every 8th photo, often glancing views).

## Cost

| Item | Cost |
|---|---|
| One training run (about 2.5 min of training + 30 s start + 60 s idle on an H100) | ≈ $0.30 |
| Today's whole `litereality-splat` app: two image builds (one on an H100), a failed first run, this run | $0.57 |

## In the call

- Locally: `roomcall run/tea_room/objects/scene.xml --splat run/tea_room/splat`.
- Hosted: the deployed link, with `splatroom --publish`.
- Each snapshot swapped into an open browser tab while it was in the room. Nobody left the room
  or reloaded.
- The 7000-step room downloads as 13 MB, about 6 s on this connection.
- The Photo button switches between the splats and the meshes.

## Known gaps

- **Objects are cut by their RoomPlan boxes.** RoomPlan boxes the chairs 20–30 cm off. A snap
  step moves each box to where it holds the most splats, but around the table it also catches
  table and wall edges. Per-photo masks (SAM 2) are the fix.
- **Holes:** where a moved object stood, nothing was captured behind it, so a hole shows.
