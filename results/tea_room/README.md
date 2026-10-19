# tea_room — MuJoCo seed scene with estimated physics

This is the result of running the LiteReality-Agent pipeline on the `tea_room` example scan
(a RoomPlan `room.usdz` capture) up through the **seed** stage, then exporting the reconstructed
room to a MuJoCo scene (`--from-seed`) and running the shake measurement.

The scene is "seed" physics: every mass / friction / restitution here is an **estimate made before
anything was watched moving** — a category density over the object's bounding box, a friction from
the chosen material, etc. A later dynamic observation would correct them.

## What's in this folder

| file | what it is |
|---|---|
| `physics_table.md` | the per-object physics tables (simulated + sources), human-readable |
| `physics_table.csv` | the same, machine-readable (`scripts/simready/physics_table.py` output) |
| `export_report.json` | the MuJoCo exporter's report (bodies, colliders, `initial_overlaps`, `loads_clean`) |
| `shake_report.json` | the `mujoco_shake` measurement (what moved under a rising ground shake) |

**Not included:** `scene.xml` + its `meshes/` come to ~53 MB together, over the 50 MB cap, so they
are left out (and `Room.glb`, 43 MB, likewise). Regenerate them with the commands below.

## How it was produced

```
# providers: images via Modal (TRELLIS mesh gen, DINO, Qwen image-edit); reasoning/authoring via Claude
uv run litereality setup                       # deploy the Modal model app (one-time)
uv run python sanity.py                        # SANITY OK

LR_FORCE_TRELLIS=Dishwasher0 \
uv run litereality run tea_room --through seed  # ingest -> reconstruct -> seed
uv run litereality stage simulate run/tea_room --from-seed --shake
uv run python -m litereality_agent.room_ops.export.mujoco_shake \
    --scene run/tea_room/mujoco_seed/scene.xml --cutaway
uv run python scripts/simready/physics_table.py \
    run/tea_room/mujoco_seed/scene.xml --csv run/tea_room/physics_table.csv
```

The room has 12 reconstructed objects: 5 via TRELLIS (3 chair clusters, a sofa, the dishwasher),
5 procedural (sink unit, 3 storage units, a table), and 2 openings (a door, a window).

## Physics table (simulated — what MuJoCo steps)

| body | joint | mass_kg | friction |
|---|---|---|---|
| Door0_Door_leaf | hinge | **128.731** ⚠️ | 0.45 |
| Sink_Storage0_Drawer_Right | slide | 0.702 | 0.45 |
| Sink_Storage0_Door_Right | hinge | 0.324 | 0.45 |
| Sink_Storage0_Door_Left | hinge | 0.527 | 0.45 |
| Storage0_door_r | hinge | 1.418 | 0.45 |
| Storage0_door_l | hinge | 1.544 | 0.45 |
| Storage1_Door_Front | hinge | 2.953 | 0.55 |
| Window0_Sash_Sliding_Right | slide | 4.425 | 0.22 |
| Chair0 | free | 5.106 | 0.55 |
| Chair1 | free | 5.617 | 0.55 |
| Chair2 | free | 5.348 | 0.55 |
| Chair3 | free | 5.617 | 0.55 |
| Dishwasher0 | free | 30.881 | 0.55 |

Source estimates (density × bounding box): Dishwasher0 30.46 kg, Sofa0 18.9 kg, chair clusters
9–11 kg, storage units 24–55 kg, table 19 kg. Restitution 0.15 is carried for other engines;
MuJoCo does not simulate it. Full detail in `physics_table.md` / `.csv`. No inf/nan.

## Shake summary (`shake_report.json`)

- 23 bodies (5 free, 9 static), 211 geoms. **Static bodies stayed put** (`static_moved_max 0.01 cm`).
- Overlaps: 32 geom-level initial → **5 after relax**. Three body pairs:
  `Chair1↔Table0 −74 mm`, `Chair3↔Table0 −73 mm`, `Dishwasher0↔room −12 mm`.
- `lifted_out_of_walls`: Chair1 +0.197 m, Chair3 +0.188 m, Dishwasher0 +0.113 m.
- What moved — settle: **Chair3 102.5 cm** (ejected ~1 m resolving its table penetration),
  Chair1 3.7 cm, rest <1 cm. Shake: Dishwasher0 5.8 cm, Chair3 2.3 cm, others ≤1.2 cm.
- Hinges swung freely under gravity (door leaf 90°, cabinet doors 20–77°, window sash 56°).

## Issues found

1. **Door leaf mass 128.7 kg** — the simulated `Door0_Door_leaf` is far too heavy (a real leaf is
   ~20–40 kg; the source table estimates the whole door at ~19–43 kg). The compiled door collider
   volume is inflated. Not inf/nan, but wrong — worth fixing before trusting door dynamics.
2. **Chair/table overlaps** — Chair1 and Chair3 start ~7 cm inside Table0 (tucked chairs in the
   scan). The `simulate` acceptance gate rejects the scene for these "initial interpenetrations";
   the relax step resolves all but 5 geom contacts, but Chair3 is ejected ~1 m when its penetration
   is resolved. A small per-object inflation or a pre-relax at placement would avoid the ejection.
3. **`Wall10_Door_0/object.py` is truncated** — the generated editable source ends at the
   `OBJECT DEFINITION` marker, missing the whole build body + `export_glb()` call, so the seed
   bake's regen silently produces nothing and `Room.glb` never assembles (which blocks
   `--from-seed`). Worked around by reusing the already-built, sim-passed door glb from
   `reconstructed_objs`. The openings generator should be fixed so `object.py` is complete.
4. **Physics-compile race** — the reconstruct stage's `compile_directory` ran before the TRELLIS
   branch finished writing `Dishwasher0.glb`, so the dishwasher shipped with no `physics.json`
   (the gate only checks the glb exists). Fixed by compiling its physics (`compile_object`) and
   re-seeding so the Room package carries it. A barrier before the physics compile would fix this.
5. **`LR_FORCE_TRELLIS=Dishwasher0`** — the procedural authoring agent could not finish the
   dishwasher: every attempt exhausted its $5 / 80-turn budget and still failed a cosmetic
   door-handle QC. Forcing it to the TRELLIS route (via `LR_FORCE_TRELLIS`) built it as a solid
   body (30.46 kg, 1 collider, sim-check pass).

## Cost

- **Modal (GPU):** ~25–30 min of H100/GPU wall-time total, dominated by the one-time setup TRELLIS
  build; roughly **~$2–3** of credit (no dollar figure is printed in the logs — estimated from GPU
  wall-times; subsequent runs reuse the warm image cache).
- **Claude (Anthropic API, separate from Modal):** procedural authoring spent ~$23 for the 5
  procedural objects plus ~$10 on the abandoned Dishwasher0 attempts.
