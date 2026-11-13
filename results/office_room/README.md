# office_room (LiteReality example scan): second room for the paper (2026-11-13)

```bash
uv run litereality realroom ~/LiteReality-Agent/example-scans/office_room --pass both
uv run litereality splatroom ~/LiteReality-Agent/example-scans/office_room
uv run --with playwright litereality evaluate run/office_room --render --max-views 16 --out run/office_room/eval
```

## Build

| Step | Result |
|---|---|
| Realistic objects (TRELLIS, 4×H100) | 7 objects, final pass, 410 s, about $2 |
| Instant splat | 900,688 splats, 9 s |
| SAM 2 masks | 97 masks from 46 photos |
| Final trained splat (1250 steps × 4 photos) | 83 s after the start |
| Held-out photos | PSNR 19.28 dB, SSIM 0.662, LPIPS 0.377 |

After the body-to-body collision check (same day), the photos move one object. Its physics
audit: 6/6 parts interactable, no overlap above 5 mm, no drift. The first cut had moved three
objects, and one of them into another body.

## Fidelity, 16 views (same protocol as tea_room)

| Stage | PSNR | SSIM | PSNR walls/floor | PSNR objects |
|---|---|---|---|---|
| Flat-colour baseline | 12.31 | | | |
| Mesh room (photo walls + TRELLIS objects) | 13.48 | 0.569 | 14.88 | 9.77 |
| Splat room | 18.57 | 0.648 | 18.91 | 17.93 |
| Splat − mesh, paired (95% bootstrap) | +5.09 [3.30, 6.65], better in 15/16 views | +0.079 | +4.04 | +8.16 [5.48, 10.63] |
