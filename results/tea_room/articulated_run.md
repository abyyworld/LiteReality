# tea_room: openable furniture with Infinigen, local (2026-11-13)

```bash
# one-time: deploy/modal/infinigen/README.md "Local env" (pinned commit 224598d, Python 3.11, bpy 4.2.0)
LR_INFINIGEN_PYTHON=~/infinigen/.venv/bin/python \
  uv run litereality realroom ~/LiteReality-Agent/example-scans/tea_room --articulated --fake \
  --out run/tea_room/articulated
uv run litereality evaluate run/tea_room/articulated/scene.xml --out run/tea_room/eval_articulated
```

`--fake` uses stand-in boxes for the TRELLIS objects, so this costs nothing. The Infinigen
assets themselves are real.

## Build

| Step | Time |
|---|---|
| Infinigen-Sim install (clone, venv, `[sim]`, scikit-image from source) | about 10 min, once |
| 7 openable assets built in parallel on this Mac | 91 s wall clock in total |

Per asset:

| Asset | Factory | Time (s) |
|---|---|---|
| Dishwasher0 | dishwasher | 38.1 |
| Wall10_Window_0 | window | 38.6 |
| Storage0 | cabinet | 44.1 |
| Sink_Storage0 | cabinet | 45.6 |
| Wall10_Door_0 | door | 10.5 |
| Storage1 | cabinet | 12.6 |
| Storage3 | cabinet | 20.5 |

## Physics audit

- **Moving parts:** 17 in all: 4 free chairs, 9 hinges, 4 slides.
- **Interactable** (grabbed, moved by the standard pull, then at rest): **12/17**.
  - Hinges: 8/9 (every cabinet door, the room door).
  - Slides: 2/4 (both window sashes).
  - Free bodies: 2/4.
- **Failed:**
  - Chair1 and Chair3: tucked under the table, so the pull cannot free them.
  - Dishwasher0_door: a drop-down door is given just enough friction to stay where it is left,
    and that is more than the standard pull.
  - Dishwasher0_rack0 and rack1: they cannot slide out while the door is closed, which is
    correct.
- **Mass outliers:** 4/17.
  - The two dishwasher racks (0.15 kg) were checked against the whole dishwasher's range
    (30–60 kg). That is an audit flaw: a part needs its own category.
  - The two window sashes (2.8 kg) are just under the 3.55 kg minimum.
- **Stability:** no overlaps above 5 mm and no drift over 10 s.
- **Cost:** 0.24 ms per step, 8× real time per room on one thread. The server held 16 rooms (its
  cap) in real time, measured while other jobs kept the machine busy.

## To fix

- Done: the audit checks a rack against a rack's range (0.5-4 kg), and the articulated build
  gives a rack at least 1.2 kg (`RACK_MIN_MASS`): it is thin wire, so its share of the material
  volume was next to nothing (0.15 kg). Re-audited, mass outliers are 2/17: only the window
  sashes, 2.8 kg against a 3.55 kg minimum.
- The dishwasher door's holding friction should let a person's pull open it, while still holding
  it at rest.
