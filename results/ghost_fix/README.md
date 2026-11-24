# ghost_fix: the chair (and door) that stayed behind when grabbed

The complaint: grab a chair in the room call and it doubles. One copy moves with your hand and the
other stays where the chair was. The door did the same: a full closed copy, window and handle
included, stayed in the frame while the leaf swung.

## Root cause

The room splat itself drew the copy. Nothing else in the call draws anything at a body's rest
pose: no mesh, no `raw.spz.bin`, no backdrop and no second body copy. Both earlier diagnoses agreed
on this. The room kept the copy for three reasons.

1. **Chairs were cut by their cluster only.**
   - A body matched to an object found in the splat took only that object's voxels
     (`split.py`, the `claim` from `instances.Found.labels`).
   - An object's cluster is its opaque core: voxels with 2 or more splats of opacity 0.3 or more.
     For a real chair that is the seat and back.
   - The thin legs and the rim and underside of the seat never join the cluster, so they stayed in
     the room.
   - The move test is scored on that same core, so it reported "100% in body".
2. **The door was cut to its 4 cm leaf.**
   - The trained splat spreads the door's face from 10 cm in front of the leaf's mid-plane to 6 cm
     behind it.
   - The cut took splats only within ±4 cm (the ±2 cm leaf plus `MARGIN`), so the room kept the
     room-facing face of the closed door.
3. **The wall behind a chair was never seen.**
   - After the first two fixes, Chair0's spot still showed a white chair-shaped shell. That is no
     splat at all: the pixels are exactly the scene background colour, (223, 229, 232).
   - It is a window through the room onto the empty background. The cameras never saw the strip of
     wall and desk end behind the chair, so the trained room has nothing there.
   - Evidence: `room_only_rows_old_cut_new_cut_wallfill.jpg` shows the room with every body
     emptied, from Chair0's two viewpoints. Its rows are the old cut, the new cut, and the new cut
     with the wall fill.

A fourth problem made "the chairs move together". `instances.fit_scene` fitted Chair1 and Chair2
so that their seats overlapped in plan at different heights, 4 cm apart vertically. Lifting Chair2
straight up drove its seat into Chair1's seat: Chair1 moved 3.7 cm and turned 6.8°
(`before/movecheck.json`).

## Fix (code)

- `pipeline/splatroom/split.py`
  - **Box cut for every body:** every body, matched or not, is cut by its boxes (each geom plus one
    union box, `_cut_boxes`). A matched body's cluster still decides the splats the cluster holds.
    The floor under a matched body stays the room's (`found.floor + CUT_FLOOR`).
  - **Plates:** a plate's thin side (door leaf, seat, back) is grown to `PLATE_HALF` = 8 cm
    (`_plate`).
  - **Splats near static boxes:** a splat inside a body's own box is locked to the room only if it
    is inside a static box proper. The 3 cm `STATIC_KEEP` zone still applies in the margin around
    the box. This keeps the door's edge, which stands close to the wall it closes.
  - **`fill_walls`:**
    - Applies to a free body's nearby walls (shell walls within 1 m of it).
    - It patches the strip of wall behind the body that no splat covers: the body's span plus 0.35 m,
      from the floor to the body's top plus 0.15 m.
    - The patch is flat wall splats at the depth of the seen wall, coloured from the nearest seen
      wall splats.
    - `floor_fill` already covered the footprint.
  - **`left_behind`:** a new check of what the room keeps at each body's rest pose:
    - It measures the room's share of the opacity inside the body's cut boxes plus 5 cm, above the
      floor and clear of static boxes.
    - It does not depend on how the cut was made.
    - It is part of the move test. A body passes at `LEFT_MAX` = 8% or less.
- `pipeline/splatroom/instances.py`: `_settle` also narrows fitted bodies until they are at least
  1 cm apart in plan (`split.plan_overlap`). Before, it only checked that they did not
  interpenetrate.
- `pipeline/splatroom/__init__.py`:
  - `bundle_move_test` reports `left_behind`.
  - `splat.json`'s `move_test` carries it.
  - The `wall_fill` splats are left out of the check, like `floor_fill`.

## The fixed bundle

`run/atrium_stray_q/v2_chairs_fix` (out of git):

```
uv run litereality splatcut run/atrium_stray_q/v2 --out run/atrium_stray_q/v2_chairs_fix
```

It is the same trained v2 snapshot (`raw.spz.bin`), re-cut locally with no Modal. It is not
published. It replaces `v2_chairs`.

| body | v2 left | v2_chairs left | v2_chairs_fix left | v2_chairs splats | v2_chairs_fix splats |
|---|---|---|---|---|---|
| Chair0 | 0.138 | 0.124 | 0.024 | 10,745 | 13,042 |
| Chair1 | 0.918 | 0.064 | 0.008 | 67,604 | 73,175 |
| Chair2 | 0.797 | 0.105 | 0.015 | 37,394 | 47,105 |
| Wall0_Door_0 | 0.080 | 0.409 | 0.053 | 49,560 | 102,243 |

- **"left"** is the room's share of the splats at the body's rest pose, by opacity
  (`check_*.json`: `litereality splatcut DIR --check`).
- **Move test:** v2 FAIL, v2_chairs FAIL, v2_chairs_fix pass.
- **Room fill in v2_chairs_fix:** 478 floor splats and 10,727 wall splats.

## Grab test in the real room call

- **Script:** `movecheck.py`, a copy of `fix_v4/movecheck/movecheck2.py`.
- **Setup:**
  - Served locally on Metal with headless Chromium.
  - A chair is grabbed at its centre of mass.
  - The hand rises 0.95 m, so the held body rises about 0.9 m.
  - It is then carried 0.75 m sideways, away from the other chairs and the walls.
  - A chair the desk blocks is first pulled out from under it, then lifted.
  - Two viewpoints per body.
- **Outputs:**
  - `after/` (v2_chairs_fix) and `before/` (v2_chairs): sheets and `movecheck.json`.
  - `compare_<body>.jpg`: before and after side by side.
  - Raw PNGs are in `~/LiteReality-scans/ghost_fix/{before,after}/`.

| body (after) | rise | carried (plan) | others moved on the straight lift | ghost at rest pose |
|---|---|---|---|---|
| Chair0 | 0.895 m | 0.819 m | none | none (see below) |
| Chair1 | 0.176 m straight up (desk top); pulled out 0.70 m, then 0.95 m | 1.304 m | none (none while pulled either) | none: empty floor under the desk |
| Chair2 | 0.934 m | 0.868 m | none | none: floor |
| Wall0_Door_0 | swung 48.2° | | none | none: the closed copy is gone, the doorway shows the corridor |

Before (v2_chairs):

- Chair2's straight lift moved Chair1 by 3.7 cm / 6.8° (adjacent check FAIL).
- Pulling out Chair1 moved Chair2 by 5.7 cm / 8.8°.
- Chair0 left a background-coloured chair shell.
- The door left its closed copy.

## Still not right

- **Chair0's spot:**
  - Once the chair is carried off, you see the desk's metal end frame and the new wall patch, which
    has a yellowish blob.
  - I checked that the frame is not left-over chair: the room keeps 2.4% of the opacity at the
    chair's rest pose, and recolouring by position puts those legs 0.4 m or more from the chair's
    box.
  - The blue chair's own thin metal legs are barely in the trained splat. Its body has almost
    nothing below 0.2 m.
- **Held chairs tip over:** a free chair tips 65-85° while held, even when grabbed at its centre
  of mass with nothing touching it (reproduced headless with `RoomSim`). This is the grab
  physics in `roomcall/sim.py`, not the cut, and it is not fixed here.
- **Carpet carried with chairs:** chair bodies now take a little carpet fuzz 3-8 cm above the floor
  (Chair2 opacity 341, Chair1 671), which goes with the chair.
- **Dropping Chair2:** let go from 0.93 m, it falls onto Chair1 and moves it 5.4 cm. That is a
  real drop, not the two chairs being bound.
- **Narrower boxes:** Chair1 and Chair2's fitted boxes are now narrowed to 0.734 of their footprint
  so they keep apart. The cut still holds 98-99% of what is at their rest poses.
