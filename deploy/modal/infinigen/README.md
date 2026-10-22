# Infinigen-Sim: openable furniture for the objects stage

`litereality realroom <scan> --articulated` and `litereality roomcall <scan> --realistic --articulated`
replace the quick room's cabinets, dishwasher, fridge, oven, door and windows with articulated
assets from [Infinigen-Sim](https://github.com/princeton-vl/infinigen) (BSD-3): real hinges,
drawers, racks and sliding sashes, sized to the RoomPlan box, with our masses, joint dynamics and
the photo's colour (`src/litereality_agent/models/articulated/infinigen.py`,
`src/litereality_agent/pipeline/realroom/articulated.py`).

Infinigen needs its own Python 3.11 with exactly `bpy==4.2.0`, so it never runs inside our env. It
runs in one of two places:

| | where | set up once | per room (tea_room, 7 assets) |
|---|---|---|---|
| **Local env** | a subprocess of `infinigen_worker.py` | ~10 min install, 2.7 GB | ~1.5 min on 3 workers (4 CPUs), free |
| **Modal** | the `generate` CPU function of the `litereality-infinigen` app | `modal deploy` (~10-15 min image build) | one container per asset, ~1 min + container starts, a few US cents |

The local env wins when `LR_INFINIGEN_PYTHON` is exported; otherwise Modal is used. Finished
assets are cached in `~/.cache/litereality/infinigen` (`LR_INFINIGEN_CACHE`) by factory, size to
the centimetre and seed, so a re-run of the same room builds nothing. The offline stand-ins
(`--fake` without `LR_INFINIGEN_PYTHON`) are cached apart, under `fake/`, and a real run never
takes one for a real asset. The test suite clears `LR_ARTICULATED` and `LR_INFINIGEN_PYTHON` and
uses its own cache, so exporting them for real work does not make `pytest` build assets.

## Local env (macOS or Linux)

```bash
git clone https://github.com/princeton-vl/infinigen ~/infinigen
git -C ~/infinigen checkout 224598d4a2ff95b4fc928e72c222598b7122c698
uv venv -p 3.11 ~/infinigen/.venv
INFINIGEN_MINIMAL_INSTALL=True uv pip install -p ~/infinigen/.venv/bin/python -e "$HOME/infinigen[sim]"
export LR_INFINIGEN_PYTHON=~/infinigen/.venv/bin/python   # in the shell: .env does not reach it
uv run litereality realroom tea_room --articulated --fake  # TRELLIS stand-ins, real Infinigen
```

- The install compiles scikit-image 0.19 from source (Infinigen's v1 extra pins `<0.20`, which
  has no Python 3.11 wheel): on a Mac that needs the Xcode command line tools
  (`xcode-select --install`). It took 8.5 min on Linux.
- `LR_INFINIGEN_WORKERS` sets how many assets build at once (default: half the CPUs, at most 4).
  Each worker is one Blender process (~1-2 GB).
- The worker never installs Blender add-ons and gives Blender a throw-away user folder, so
  nothing is written to `~/.config/blender` (or `~/Library/Application Support/Blender`).

## Modal

```bash
uv run modal deploy --env main deploy/modal/infinigen/app.py   # once, and after updating the worker
uv run litereality roomcall tea_room --realistic --articulated --publish
```

- It uses the same Modal login as the rest (`MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET` or
  `MODAL_PROFILE` in `.env`). The CLI checks that the app is deployed before it plans anything.
- **Container settings:** 2 CPUs and 6 GB per container, at most 8 at once, scaling to zero
  60 s after the last asset.
- **Cost:** about 10-90 s per asset. At Modal's CPU and memory rates a room of seven costs well
  under $0.10, plus container start-ups.
- **Stop it:** `uv run modal app stop litereality-infinigen`.
- **Not tried yet:** the image build and a real call were not run from this repo. It is the
  study's environment, pinned, with the same worker script, and the request handling is
  unit-tested, as is importing `app.py` the way each container does (alone, as
  `/root/app.py`).

## What is overridden, and why

Every finding below comes from the feasibility study (30 assets, 2026-10-21):
- **Masses:** exported masses are 3-30x too heavy (metal densities on solid hulls). The total is
  the box volume x `OCCUPANCY_DENSITY`, shared over the links by material volume. A door leaf
  takes all of the door's estimate.
- **Joints:** joints ship with damping up to 5000 and frictionloss 1000. They get
  `JOINT_DEFAULTS` instead. A drop-down door (dishwasher, oven) gets just enough friction to stay
  where it is left.
- **Small parts:** handles, knobs, buttons and lock levers are welded to their part. Parts that a
  sizing bug left outside the object are dropped.
- **Dishwasher:** it stands on an 8 cm plinth, and its racks' slide is clamped to the tub.
- **Fit:** objects are scaled into the RoomPlan box per axis. Doors and windows keep their own
  thickness in the wall.
- **Room to swing:** a cabinet RoomPlan put flush against a side wall is moved in by up to 6 cm on
  that side, so its door can swing open.
- **Contacts between parts:** a part does not collide with what it hangs on, or with a part it
  touches at rest (a cabinet's two doors meet by ~1 mm). Every other pair collides, so a
  dishwasher's rack is held in by its closed door (it pushes an unlatched door open) and the
  door cannot close through a pulled-out rack.
- **Doorway:** the door frame's bottom bar (6 cm in both scans) goes under the floor without a
  collider, and the leaf hangs 1 cm above the floor, as the quick room's does.
- **Movable furniture in the way:** a chair or stool the asset starts inside (handles stand up
  to 12 cm proud of the scanned box) is moved out along the floor, up to 30 cm, and the move
  is recorded in `objects.json` (`check.moved_out_of_its_way_m`).
- **Failures:** an object whose asset fails, or is refused, keeps its template and is given
  back to TRELLIS, so `--articulated` never leaves an object worse than `--realistic` alone.
- **Masses in objects.json:** `mass_kg` is the estimate the moving parts (or the whole object)
  share. A door's or window's frame is welded and reported apart (`static_mass_kg`); a kept
  worktop is `kept_mass_kg`; the plinth carries no mass. Compiled masses add up to these.
- **Colour:** the diffuse textures are tinted with quickroom's photo colour for the object.

Gaps:
- no countertop, sink or corner unit. A counter's cabinet goes under the quick room's worktop.
- no mixed door-and-drawer unit, and no cabinet with more than 2 doors.
- a tall fridge gets a single door and no freezer.
- drawer chests only with `--drawers HANDLE,...`.
