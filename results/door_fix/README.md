# Door fix: the gold streaks and the junk in the doorway

The live atrium (`v2_chairs_fix2`) showed a sheaf of gold, pink and green streaks wherever the door was
seen edge on or swung open, and a white "sheet" and other floaters hanging in the doorway once it opened.

## Causes

The door was only filmed face on, from inside the room.

1. **Colour.** Each splat's higher SH (degree 1) is right for the room's side only. From anywhere
   else it adds pink and green.
2. **The far side.** Training left splats deeper in the leaf than its face, hidden from every camera
   by the face in front of them. Swung open, that layer is all the far side shows.
3. **The doorway.** Past the door's edges the cameras glimpsed the corridor. Training smeared that into
   splats 0.5-2 m behind the leaf: 5,500 of them, hidden while the door was shut. They also punched
   holes in the dim space `fill_doorways` puts there, since it fills only where nothing was seen.

## Fix (`pipeline/splatroom/split.py`)

- **`flatten_plate`.** Every door splat keeps its place and takes the colour the room saw it in: its
  SH evaluated over a 35° cone of views from the room's side, then dropped. Splats deeper than 1.2 cm
  behind the face go. `_backing` closes the leaf with a sheet of flat splats every 1.2 cm, painted
  from the face. It is as opaque as the face is there, so the window's glass stays see-through.
- **`clear_doorways`.** Before the fills, every room splat in the space behind a door, and 1 m past
  it, is removed. The wall the door stands in is kept: nothing nearer than 6 cm past the leaf's
  mid-plane goes. `fill_doorways` then fills the whole space.
- Chairs are cut exactly as before: the same files, byte for byte.

## Tried and dropped

| variant | shut, from the room | open |
|---|---|---|
| drop SH + squash the leaf flat (x0.15) | **worse**: pink and green blotches, muddy handle | streaks remain |
| bake SH only (A) | cleaner than live | far side still junk |
| bake + opaque backing (C) | window turns blotchy brown | real-looking door |
| **bake + backing as opaque as the face (D, shipped)** | as live or cleaner, window reads as glass | real-looking door, doorway clean |

`compare_live_vs_fix.jpg`: live (left), C (middle) and D (right). The door is shut in the top three
rows and pulled open 60° at its handle in the bottom three, in the real room call (local server,
headless Chromium on Metal, `doorcheck.py`). Raw PNGs are in `~/LiteReality-scans/door_fix/`.

## Tests

The three new tests in `tests/test_splat_cutting.py` fail on the old cut. The full suite passes:
1907 passed, 8 skipped.

## Still not fixed

- Seen edge on, a few white needles remain at the door's frame. They are room splats in the wall
  around the opening.
- Opened, the door's far side shows the near side's picture (handle and window mirrored). No camera
  ever saw the far side.
