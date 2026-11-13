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
  - Dishwasher0_door: opens 32.2 of the 51.6 deg needed, then hits Chair1 (see "The dishwasher
    door" below). Its holding friction is not the cause.
  - Dishwasher0_rack0 and rack1 (0.176 and 0.152 of 0.199 m): after 6 cm a rack pushes the
    closed door open from inside, and the door then hits Chair1.
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
- Measured, not a friction problem: see "The dishwasher door" below. Left open: keep chairs out
  of an appliance door's swing, and stop a rack from pushing its closed door open.

## The dishwasher door (2026-11-13)

The door is a drop-down flap: frictionloss = `FLAP_HOLD` (1.25) x m g r = 19.2 N m (3.94 kg,
centre of mass 0.40 m from the hinge), so it holds its own weight even lying flat. Measured
through the audit's own `pull_test` and `RoomSim`:

- **Opening is fine.** The grab spring starts the door turning within 0.2 s, at 19 N m (about
  30 N at the handle; the grab's force cap is 25 g x 3.94 kg = 966 N). It then turns freely to
  32 deg.
- **Chair1 stops it.** The dishwasher faces the table, and Chair1, pushed in under the table,
  reaches about 3 cm into the door's swing. At 32 deg the door hits the chair's seat and the hand
  presses with up to 450 N m against a chair wedged under the table. Both racks fail the same
  way: after 6 cm a rack touches the door's inside, pushes it open, and the door hits Chair1.
- **With Chair1 excluded from colliding with the dishwasher** (nothing else changed): door
  66.3/51.6 deg in 1.16 s, **pass**, stable, 0.04 deg drift after release. rack0 0.200/0.199 m
  **pass** (by pushing the door open). rack1 0.190/0.199 m, still short.
- **It stays where it is left.** Released and stepped for 10 s (plain `mj_step`, Chair1 excluded):

  | Released at (deg) | 0 | 10 | 30 | 45 | 60 | 85.9 (flat) |
  |---|---|---|---|---|---|---|
  | Moved in 10 s (deg) | +0.01 | +0.04 | +0.11 | +0.15 | +0.18 | +0.07 |

  There are no reversals and the peak speed is 0.0004 rad/s after 0.2 s, so it does not
  oscillate. Torque to move it: 19 N m to open from closed, 32 N m to lift it from 45 deg,
  36.5 N m to lift it from flat (about 46 N at its top edge).

So the hold already meets all three requirements, and a lower `FLAP_HOLD` with a spring would not
pass tea_room: the chair would still be in the way. `FLAP_HOLD` is unchanged.
`tests/test_articulated.py::test_a_person_can_pull_a_drop_down_door_open_and_it_stays_where_left`
runs the audit's pull through `RoomSim` on a built dishwasher and checks the door holds when
released shut and when flat. It fails, with 18.9 deg drift, at `FLAP_HOLD = 0.5`.

Before and after the rebuild: 12/17 interactable, the same five failures (Chair1, Chair3,
Dishwasher0_door, Dishwasher0_rack0, Dishwasher0_rack1).
