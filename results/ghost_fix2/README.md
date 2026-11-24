# ghost_fix2: legs left standing, the desk carried off, chairs too small, chairs that tip

Round 1 (`results/ghost_fix`, a4a4ea4) made the room give up most of each grabbed object. Two
skeptics failed it for five concrete reasons (F1-F5) and a few smaller ones (F6). This round fixes
them at the root, re-cuts the same trained v2 snapshot as `run/atrium_stray_q/v2_chairs_fix2`
(local, no Modal, not published), and checks the result in the real room call.

```
uv run litereality splatcut run/atrium_stray_q/v2 --out run/atrium_stray_q/v2_chairs_fix2
```

## Root causes

- **F1: thin legs and a seat lip stayed standing.**
  - `instances._settle` narrowed Chair1 and Chair2 to 0.734 of their fitted footprint so that
    they kept apart in plan.
  - The same narrowed geoms were the boxes the splat was cut by. Legs 5-20 cm outside them went
    to the room.
  - Even unnarrowed, a fitted box is the object's opaque core. Thin legs splay past it, and so
    does an office chair's star base.
  - `left_behind` only looked 5 cm past the box, and only by total opacity. Thin legs have very
    little opacity, so it could not see them.
- **F2: Chair2 carried a strip of the desk.**
  - `split` locked a splat inside a body's own box to the room only if it was strictly inside a
    static box.
  - The trained desk edge spreads up to 3 cm past the desk's box, into the box of the chair tucked
    under it. As a result Chair2 carried 2,870 splats of the desk edge and Chair1 carried 1,769.
- **F3: chairs passed into each other and the desk.**
  - The collision boxes were the narrowed ones, about 27% smaller than the chairs.
- **F4: held chairs tipped 58-86°.**
  - The grab spring in `roomcall/sim.py` pulls only the grabbed point and damps spin. Nothing held
    the chair's tilt.
  - So any torque left the chair at whatever angle it reached: lifting a seat whose edge is under
    the desk top, accelerating with the hand off the centre of mass, or the legs dragging on lift-off.
  - The skeptic's "nothing touching" case (Chair0 grabbed at its centre of mass) does touch: the
    seat's front edge is under the desk top and catches it at a 0.21 m rise.
- **F5: the open doorway showed the scene background.**
  - The cameras never saw past the closed door, so nothing was trained there.

## What changed (code)

### `pipeline/splatroom/instances.py`

- **`_mark_extents`:** every fitted body gets an `<name>_extent` box site. It is invisible, has no
  mass and takes no contacts, and it records the object as fitted, before anything moves.
- **`_settle`:** two fitted movable bodies that overlap in plan are **slid apart**.
  - The slide is along the axis that separates them best (`_separation`), half each, `SETTLE_GAP`
    apart.
  - Each body moves at most `SETTLE_SHIFT` = 10 cm, and never deeper into anything static.
  - Narrowing is now only the fallback, for a body sunk into something static or one that cannot be
    slid.
- **`_shift_body`:** moves the body and shifts its extent site back, so the extent stays on the
  object.
- **Atrium:** Chair1 and Chair2 each slid 5.1 cm (`[0.031, 0.041]` m). The collision boxes keep the
  fitted scale (Chair1 1.058/1.117, Chair2 1.046/1.04), with no `shrunk_to_clear`.

### `pipeline/splatroom/split.py`

- **Cut is separate from collision:** `_extent_boxes`, `_shape_boxes` and `_cut_boxes` cut a fitted
  body by its extent, not by its (slid or narrowed) geoms. Floor fill and wall fill use the extent
  too (`_corners`, `_footprint`).
- **`_leg_zones`, `_rim_zones` and `_grow_legs`:**
  - **Leg zone:** a free body's leg zone is its extent grown `LEG_REACH` = 20 cm in plan, from the
    floor up to `LEG_ZONE` = 60% of its height.
  - **Rim zone:** the box grown `RIM_REACH` = 10 cm every way but down.
  - **Growth:** the body's splats are grown voxel by voxel (3 cm voxels, 26-connected) through room
    splats in its zones. A voxel carries the growth if it holds a splat of opacity 0.1 or more.
  - **Carpet:** below `LEG_FUZZ` = 8 cm the growth only goes straight down (from a leg to its
    caster), never along the carpet pile.
  - **What it never takes:** the floor, anything within `STATIC_KEEP` of static furniture, or what
    another body holds.
  - **Neighbours:** bodies grow together, and a voxel reached in the same step goes to the nearer
    body core.
  - **Overlapping boxes:** a splat inside two bodies' boxes goes to the box it is deeper in (signed
    distance).
- **Static protection (F2):** a splat within `STATIC_KEEP` of static furniture stays in the room
  even inside a body's box. The exception is a body with no object found for it (the door), whose
  box is all the evidence there is.
- **`left_behind`:**
  - It also takes the connected leg and rim zone (`_grow_legs`, so the next chair and the carpet do
    not count).
  - It reports `legs_left`, the room's share of what stands under the seat (below 40% of the height).
  - A body fails above `LEGS_MAX` = 0.25 even when its total `left` is small.
- **`fill_doorways` (F5):**
  - Behind every hinged upright plate at least 1.4 m tall (a door), on the side away from the
    room's splats, it builds a box-shaped space. The space is as wide as the opening plus the swing
    and is `DOOR_BACK` deeper than the leaf is wide.
  - Its floor takes the colour of the room's floor at the door. Its walls and ceiling take the
    colour of the wall around the opening, each dimmed (`DOOR_DIM`).
  - Splats go only where nothing was seen.
- **`fill_walls` (part of F6):**
  - The wall patch's colour is now the median of the 200 nearest opaque (0.3 or more) wall splats.
    It was an inverse-distance mix of the 8 nearest, which painted the patch with haze and edge
    splats into a shiny chair-shaped shell.
  - A cell counts as seen only through opaque splats at the wall face or up to 30 cm behind it, so a
    patch never covers trained content behind a wall that RoomPlan put in front of the real one.

### `room_ops/roomcall/sim.py` (F4)

- **The tilt hold:** a grab of a free body records world-up in the body's frame (`_up_in`). Each
  step pulls that axis back to vertical (`_tilt`):
  - stiffness `GRAB_TILT_KP` = 4 x `GRAB_KP`, critically damped about the horizontal axes, through
    the inertia about the grabbed point;
  - the torque is clamped like the force.
- **Heading:** left free, so a chair pulled by one side still trails the hand.
- **Hinged and sliding bodies:** not affected; the door swings as before.
- **Several hands on one body:** they share the weight cancellation and the hold.
- **Live version swaps:** `adopt` carries the hold through a swap.

### Other files

- **`pipeline/splatroom/__init__.py`:** `FILLS` (floor, wall and door fill) are subtracted in
  `bundle_move_test`.
- **`results/fix_v4/chairs/truth_table.py`:** `served()` now subtracts wall fill and door fill, not
  only floor fill (F6).

### Tests

These tests fail on a4a4ea4 and pass now (checked with a worktree of a4a4ea4 and the new tests):

| test | what it checks | a4a4ea4 result |
|---|---|---|
| `test_legs_that_splay_past_the_fitted_box_go_with_the_chair_and_the_carpet_stays` (F1) | legs reach 15 cm past the seat to the floor, with carpet pile everywhere | legs 26% in body |
| `test_two_chairs_whose_legs_reach_into_each_others_zone_each_keep_their_own` (F1) | each chair keeps its own legs | 31% |
| `test_the_desk_edge_over_a_tucked_chair_stays_with_the_desk` (F2) | the desk edge stays with the desk | 27% of the edge carried |
| `test_chairs_too_close_are_slid_apart_not_shrunk_and_cut_by_what_they_hold` (F3) | chairs are slid apart, not narrowed | `shrunk_to_clear` |
| `test_a_chair_held_where_a_user_clicks_is_carried_upright[back_top / seat_corner]` (F4) | tilt while held | 19.9° and 19.8°, against under 12° now (under 1° in fact) |
| `test_an_opened_door_shows_a_dim_space_not_the_empty_background` (F5) | the doorway is filled | no fill |
| `test_a_wall_patch_never_hides_what_was_seen_behind_the_wall_face` (F6) | the patch skips seen content | patch over seen content |

One existing test changed: `test_adopt_relaxes_a_carried_overlap_and_lets_go_of_what_became_static`
now pushes the chair against the table lifted 0.15 m instead of 0.5 m.

- At 0.5 m the chair's legs stand at the table top's height.
- Held upright (as it now is), the 30%-grown chair of the new version is wedged through the top,
  which no relax can undo. Before, the chair tipped 93° and fell clear.
- The test still checks a carried overlap that has to be relaxed: 4.1 cm before the relax, 0.24 cm
  after.

## Results on v2_chairs_fix2

### Left behind at rest

From `bundle_move_test`: the bundle as served, body splats put back at their poses.

| body | v2 left / legs_left | v2_chairs_fix (a4a4ea4) left / legs_left | v2_chairs_fix2 left / legs_left |
|---|---|---|---|
| Chair0 | 0.194 / 0.726 | 0.084 / **0.391 FAIL** | **0.010 / 0.034** |
| Chair1 | 0.941 / 0.751 | 0.021 / 0.060 | **0.002 / 0.009** |
| Chair2 | 0.829 / 0.620 | 0.038 / **0.354 FAIL** | **0.007 / 0.049** |
| Wall0_Door_0 | 0.080 / n/a | 0.053 / n/a | 0.053 / n/a |

### Desk edge carried by a chair (F2)

Body splats within 3 cm of a static box:

| | Chair0 | Chair1 | Chair2 |
|---|---|---|---|
| v2_chairs_fix | 321 | 1,769 | 2,870 |
| v2_chairs_fix2 | 0 | 0 | 1 |

### Truth table

`results/fix_v4/chairs/truth_table.py`; shares are of opacity.

| chair | v2_chairs_fix | v2_chairs_fix2 |
|---|---|---|
| Blue chair | 96.8% in Chair0, room 3.2% | 98.8% in Chair0, room 1.2% |
| Teal chair | 98.3% in Chair2, room 0.9% | 99.1% in Chair2, room 0.02% |
| Dark chair | 96.6%, room 0.4%, Chair2 3.0% | 96.9%, room 0.01%, Chair2 3.1% |

### Splat counts

| | count |
|---|---|
| Chair0 | 13,383 |
| Chair1 | 77,947 |
| Chair2 | 48,553 |
| door | 102,243 |
| floor fill | 659 |
| wall fill | 17,626 |
| door fill | 10,469 |

### Grab test in the real room call

- **Setup:** local server, headless Chromium on Metal. Raw PNGs are in
  `~/LiteReality-scans/ghost_fix2/grab_fix2/`.
- **Script:** `callcheck.py`.
- **Grab points:** every chair is grabbed at 3 points a user would click: the centre of mass, the
  top of the back, and a front seat corner.
- **The desk:** it overhangs all three chairs. A slow lift stops within 3 cm, so each chair is first
  pulled out 0.6 m backwards, then lifted 0.9 m and carried 0.8 m clear of the others.
- **Shots:** from the capture path, the far side and low (`after/sheet_*`). Tilt is sampled at 10 Hz
  throughout.

| body / grab point | max tilt held (call) | rise | others moved on lift / while held / after release |
|---|---|---|---|
| Chair0 centre of mass | 0.0° | 0.900 m | 0 / 0 / 0 |
| Chair0 back top | 0.4° | 0.900 m | 0 / 0 / 0 |
| Chair0 seat corner | 0.5° | 0.900 m | 0 / 0 / 0 |
| Chair1 centre of mass | 0.0° | 0.900 m | 0 / 0 / 0 |
| Chair1 back top | 0.3° | 0.900 m | 0 / 0 / 0 |
| Chair1 seat corner | 0.3° | 0.900 m | 0 / 0 / 0 |
| Chair2 centre of mass | 0.0° | 0.900 m | 0 / 0 / 0 |
| Chair2 back top | 0.3° | 0.900 m | 0 / 0 / 0 |
| Chair2 seat corner | 0.3° | 0.900 m | 0 / 0 / 0 |
| Wall0_Door_0 (handle) | swung 57.9° | | 0 / 0 / 0 |

- **Others moved:** exactly zero, with no threshold applied.
- **Reset:** reset restored every body.

#### Headless check

`heldcheck.py` uses `RoomSim` with the same three grab points (`held_before.json` is a4a4ea4's sim
on v2_chairs_fix, `held_after.json` is this round):

| | max tilt (com / back top / seat corner) | deepest contact | rise |
|---|---|---|---|
| a4a4ea4 Chair1 | 29.6 / 6.2 / 6.2° | | |
| a4a4ea4 Chair2 | 33.6 / 7.2 / 8.3° | | |
| fix2, all chairs | 0.3° or less at every point | 0 cm | |
| a4a4ea4, yanked straight up at the origin under the desk: Chair0 | 89.7° | | 0.90 m (pried over) |
| a4a4ea4, same yank: Chair2 | 89.9° | | 0.90 m (pried over) |
| fix2, same yank: Chair0 | 5.6° | 1.65 cm, transient, at the 25 g force cap | 0.29 m (desk stops it) |
| fix2, same yank: Chair1 | 0.4° | 0.35 cm | 0.18 m (desk stops it) |
| fix2, same yank: Chair2 | 1.3° | 0.64 cm | 0.27 m (desk stops it) |

On the fix2 pull-out-first grabs, nothing of the held chair touches anything.

### Doorway (F5)

Share of the bare scene background (223, 229, 232) with the door swung open:

| view | before (v2_chairs_fix), full frame | before, centre crop | after, full frame | after, centre crop |
|---|---|---|---|---|
| cap | 4.7% | 8.0% | 0.0% | 0.0% |
| low | 3.0% | 8.1% | 0.0% | 0.01% |
| far | 1.6% | | 0.9% | |

The far view is mostly the window edge.

### Room at rest against published v2

Thirty views (`rest_diff.json`, `rest_diff_worst.jpg`):

| measure | value |
|---|---|
| mean absolute difference | 1.57 / 255 |
| share of pixels off by more than 40 | 0.94% mean |
| noise floor (same view rendered twice) | 0 |

- **Worst view, Wall0_Door_0_far, 16.7%:** the camera stands 1.3 m outside the doorway, inside the
  new space behind the door, so its grey walls fill the frame where v2 showed the background.
- **Next views, wide0 1.6%, widelow0 1.6% and Chair0_side- 1.2%:** almost all the door's window. The
  glass now shows the dim space behind it instead of the white background.

### Images

| what | where |
|---|---|
| Room-only and carried (centre of mass) frames, a4a4ea4 against this round, 3 viewpoints | `compare_Chair0.jpg`, `compare_Chair1.jpg`, `compare_Chair2.jpg`, `compare_Wall0_Door_0.jpg` |
| Full sheets: before, room only, carried for each grab point, carried with the held body faded | `after/sheet_<body>_{full,crop}.jpg` |
| Raw shots | `~/LiteReality-scans/ghost_fix2/{grab_fix2,rest_fix2,rest_v2}/` |
| Diagnostic tints | `~/LiteReality-scans/ghost_fix2/{tint_fix2,side_tint,side_plain}/`, made with `tint.py` |

## Still not right

- **Two thin grey "legs" at Chair0's spot, seen from the capture path (the cap view) only.**
  - They are not chair splats, the backdrop or a body. Hiding the backdrop meshes changes nothing.
    Deleting every room splat within 0.5 m of the chair, 3-120 cm high, leaves them. Deleting the
    floor layer below 3 cm within 0.5 m removes them.
  - So they are the legs baked into the floor's colour behind the chair, the way a trained splat
    paints a thin occluder onto the surface behind it as the training cameras saw it.
  - They are not in the SH either: zeroing the floor's view-dependent colour leaves them.
  - From the far, top and side views the spot is clean carpet.
  - The same goes for the faint seat outline on the wall and desk end there.
  - Evidence: `~/LiteReality-scans/ghost_fix2/painted_legs/`.
  - Fixing this means repainting trained surfaces along the capture rays past the body. That is not
    a cut, and it is not done here.
- **The door's window now shows the dim space behind it at rest:** a grey pane instead of v2's
  white one. The space is also visible from a camera placed outside the room.
- **Dark slab in Chair2 (F6):** 3.1% of the dark chair's opacity (1,882 splats, seat height) is
  still in Chair2. They lie on the seam where the two chairs touch, 0.29 m from both centres, and
  the truth table's own 2-means split puts them on the dark side. Neither split is clearly right.
- **Under the desk (F6):** the dark mottled patch under the desk edge after the dark chair is carried
  off is trained floor and under-desk junk. It is not touched here.
- **Forced penetration:** yanked up at the 25 g force cap against the desk edge, a chair still
  presses up to 1.65 cm into the desk for a moment.
