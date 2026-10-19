# Project notes — private copy of LiteReality-Agent

Everything needed to pick this work up without the conversation it came from. The upstream
project's own docs still apply: [README.md](README.md), [ARCHITECTURE.md](ARCHITECTURE.md),
[doc/Sim-Ready-intergration/Mujoco.md](doc/Sim-Ready-intergration/Mujoco.md).

Why this copy exists: to get LiteReality running on a student budget and use the **initial,
estimated physics** it gives a scanned room (mass, friction, joints — derived from each object's
category, size and material, never from watching it move) as a starting point for a VR
physics-perception project.

## What this copy adds on top of upstream (github.com/LiteReality/LiteReality-Agent @ 28ffe1f)

| Change | Where |
|---|---|
| `LR_IMAGE_PROVIDER=modal`: reference images from the open Qwen-Image-Edit-2509 on your own Modal workspace, so no OpenAI/Gemini key is needed | `deploy/modal/image_edit/`, `src/litereality_agent/models/image_edit/`, `.../ingest/references/image_gen.py` |
| `litereality setup` now deploys the one `litereality` app the runtime looks up (upstream deployed `litereality-dino`/`litereality-trellis`, which nothing looks up) | `src/litereality_agent/runtimes/setup.py`, `deploy/modal/app.py` |
| `stage simulate --from-seed` runs after `seed` instead of demanding the authored room | `pipeline/stage.py`, `pipeline/runner.py`, `pipeline/simulate/` |
| `--shake` runs without `ffmpeg` (skips only the video) | `pipeline/simulate/__init__.py` |
| Per-object physics table for any exported scene | `scripts/simready/physics_table.py` |
| `LR_FORCE_TRELLIS=<Name>,...`: build named objects with TRELLIS when the procedural agent can't finish them | `.../reconstruct/classify/classify_complexity.py`, `settings.py` |
| First results: the `tea_room` example scan | [`results/tea_room/`](results/tea_room/README.md) |

## Status

- First end-to-end run done on the `tea_room` example scan, macOS on Apple Silicon, Blender 5.2.2,
  Modal Starter (free) plan, Claude Code login. Results and the issues it hit are in
  [`results/tea_room/README.md`](results/tea_room/README.md).
- Modal apps are deployed in the workspace; the first run used about $2–3 of the free monthly
  credit (one-time TRELLIS build + 58 GB image-model download included). Idle apps cost nothing.

## Setting up from scratch (Mac)

1. Install `uv`, git and Blender 5.x (blender.org). Clone this repo **into a permanent folder**
   such as `~/LiteReality-Agent` — not a temp directory (macOS clears those).
2. `uv sync --frozen --extra modal --group dev` then `cp .env.example .env`.
3. Fill `.env` (never commit it; it is gitignored). Keys used:
   - `MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET` (from modal.com/settings/tokens) — or `MODAL_PROFILE`
   - `LR_IMAGE_PROVIDER=modal`
   - `BLENDER_PATH=/Applications/Blender.app/Contents/MacOS` (the folder, not the binary)
   - `LR_SCANS_DIR=<absolute path to the folder holding scans>` (e.g. a clone of
     github.com/LiteReality/example-scans)
   - `LR_AGENT_PROVIDER`, `LR_AUTHOR_PROVIDER`, `LR_MATERIALS_PROVIDER`, `LR_QUALITY_PROVIDER`,
     `LR_REFINE_PROVIDER` all `=claude` (otherwise authoring defaults to Codex)
   - optional: `LR_FORCE_TRELLIS=`, `LR_MODAL_IMAGE_STEPS=20` (cheaper images),
     `HARNESS_MODEL` / `LR_PROCEDURAL_MODEL` / `LR_CHAIR_JUDGE_MODEL` if your Claude plan lacks the
     default model
4. Log the `claude` CLI in once in a terminal (`claude`, then `/login`) — the pipeline's agents use
   that login in the background, even when you drive the work from the desktop app.
5. `uv run litereality setup` (one-time; the cold TRELLIS build is ~11 min on an H100), then
   `uv run python sanity.py` until it passes.

## Running a room

```bash
uv run litereality run <scan> --through seed                        # scan -> seed room (long)
uv run litereality stage simulate run/<scan> --from-seed --shake    # MuJoCo scene + shake
uv run python scripts/simready/physics_table.py run/<scan>/mujoco_seed/scene.xml --csv run/<scan>/physics_table.csv
```

The first room took a few hours including failures; a later room skips all one-time setup. If
`simulate` stops on "initial interpenetrations" (objects overlapping > 5 mm at t=0), the export is
still written; run the shake directly, which relaxes small overlaps:
`uv run python -m litereality_agent.room_ops.export.mujoco_shake --scene run/<scan>/mujoco_seed/scene.xml --cutaway`

## What needs which account

| Step | Runs on | Needs |
|---|---|---|
| Reference images | Modal (Qwen-Image-Edit) | Modal token only |
| Image -> 3D (TRELLIS), detection (DINO) | Modal | Modal token only |
| Classification | Claude, or OpenAI with `LR_CLASSIFY_PROVIDER=openai` | either |
| Doors/windows/cabinets (procedural agent), authoring | Claude, or Codex (`LR_*_PROVIDER=codex`) | Claude login, or Codex + OpenAI key |
| Chair-type judge, layout agent, completeness check | **Claude only** (hardcoded: `chair_type_judge.py`, `layout/agent.py` + `agent_loop.py` call `claude`, `completeness_check.py`) | Claude login or `ANTHROPIC_API_KEY` |
| Simulate, shake, physics table | local | nothing |

So a full room cannot run without Claude today; making it Claude-free means changing those three
call sites. Free hosted models (Groq etc.) are fine for the single-image calls but not for the
long Blender-coding agents.

## Known issues (from the tea_room run)

1. **Door leaf at 128.7 kg** (a real one is ~20–40 kg): its compiled collider volume is inflated. A
   genuine example of wrong estimated physics — and a bug.
2. **Initial overlaps** (chairs ~7 cm into the table): the simulate gate refuses the shake; run the
   shake directly (above).
3. **Truncated `object.py`** for an agent-built door (`Wall10_Door_0`): its bake regeneration silently
   produced nothing; worked around by reusing the already-built GLB.
4. **Physics-compile race**: an object built by TRELLIS in a re-run was compiled before its GLB
   existed; re-running `seed` fixed it.
5. **Procedural agent budget** ($5 per object, hardcoded default): Dishwasher0 ran out on a
   cosmetic QC finding; `LR_FORCE_TRELLIS=Dishwasher0` routed it to TRELLIS.
6. NumPy `matmul` RuntimeWarnings on Apple Silicon during placement — no effect found so far.

## Ideas / next steps

- **Object-only path** (not built yet): photo -> TRELLIS on Modal -> `models/object_generation/sim`
  physics -> MuJoCo drop/tilt test -> table. Needs only Modal; fits single-object stimuli.
- Fix issues 1, 3 and 4.
- Scan your own room with the LiteReality Scanner app (needs an iPhone/iPad Pro with LiDAR).
- For VR: `Room.glb` goes into Unity for Quest 3. MuJoCo ignores restitution (bounce); Unity's
  physics does not, so bounce-related stimuli need Unity or MuJoCo contact tuning.
