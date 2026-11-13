# Physics study: how wrong is estimated physics?

For each object in a scanned room, compare three values for mass, static friction and
restitution:
1. **LiteReality's estimate**: a density table by category, size and material
   (`run/<scan>/objects/quick.json`, the same table every stage uses).
2. **An LLM's estimate** from the object's own masked photo, its category and its scanned size
   (`scripts/physics_study/llm_estimates.py`).
3. **What you measure** with the protocol below.

```bash
uv run python scripts/physics_study/llm_estimates.py run/<scan> --model gpt-5-mini
```

This writes `results/<scan>/physics_study.csv` (LiteReality, LLM, and empty measured columns) and
a markdown table. tea_room with gpt-5-mini: 12 objects, 17.5k tokens, under $0.02.

## Bench protocol (per object)

Measure in the scanned room itself, on its own floor, so friction is against the surface the
simulation uses.

| Quantity | Tools | Procedure | Value |
|---|---|---|---|
| Mass | Kitchen or luggage scale (bathroom scale for > 20 kg) | Weigh 3 times; for heavy objects weigh yourself holding it and subtract your own weight | mean, kg |
| Static friction μ<sub>s</sub> | Phone inclinometer app, a board of the room's floor material | Put the object (or a small object of the same material, for furniture) on the board. Raise one end slowly until it starts to slide. Record the angle θ. Repeat 3 times | μ<sub>s</sub> = tan θ |
| Static friction, heavy furniture | Luggage scale used as a force gauge | Pull horizontally at floor level; read the peak force F just before it moves | μ<sub>s</sub> = F / (m g) |
| Restitution e | Phone slow-motion video (240 fps), tape measure on the wall | Drop the object, or a sample of its material, from h<sub>drop</sub> = 0.5 m. Read the first bounce apex h<sub>bounce</sub> from the video | e = √(h<sub>bounce</sub> / h<sub>drop</sub>) |

Notes:
- For objects you can't drop (furniture), drop a sample of the same material (a seat cushion,
  an offcut of laminate), or record "n/a".
- Fill the `measured_*` columns, and note anything unusual (carpet vs tile, castors) in
  `notes`.

## What the study reports

- **Error per method:** absolute log error of mass, |log(m<sub>est</sub>/m<sub>true</sub>)|, so
  that 2× too heavy and 2× too light count the same; absolute error of μ and of e.
- **Calibration:** whether the measured value falls inside the LLM's stated range.
- **In the room call:** the same room with estimated vs measured physics side by side (two
  scene versions), as the user-study condition for perceived realism.
- **Caveat:** MuJoCo ignores restitution (nothing bounces); bounce stimuli need Unity or contact
  tuning.

## First run (tea_room, 2026-11-13, measured values not available)

LiteReality and gpt-5-mini agree within 20% on most objects. They disagree most on:
- Storage1: 8.6 vs 22 kg (the LLM read it as sheet metal);
- Chair1: 5.6 vs 10 kg;
- the door leaf: 19 vs 12 kg.

The LLM's friction is mostly 0.40–0.50, LiteReality's 0.42–0.75. Measurement decides which is
right. Full table: [`results/tea_room/physics_study.md`](../results/tea_room/physics_study.md).
