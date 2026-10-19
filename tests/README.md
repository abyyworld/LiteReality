# tests

```bash
.venv/bin/python -m pytest          # the whole default suite, offline, ~0.5s
.venv/bin/python -m pytest -m blender   # opt-in: needs $LITEREALITY_BLENDER
```

The default run is offline and fast on purpose: a suite you can run before every commit is the only
kind that catches this repo's characteristic failure, which is **a tool that stops working without
anything failing**. An authoring pass whose `render` is broken does not crash — the model tries once,
gets an error, and quietly authors the room without ever looking at it. The run still costs $7 and
still writes a `Room.py`.

## What is covered

| file | guards |
|---|---|
| `test_scan_inference.py` | `_scan_from_room`, the choke point under `render` / `select_views` / `survey`. Both path spellings (a staging root and its resolved deliverables realpath), `Room.py` vs room-dir input, nested object rooms, `$LITEREALITY_SCAN` fallback, and that no tool grows its own inline copy. |
| `test_capability_tools.py` | The four tools stage 3 hands the model exist, carry SDK-valid schemas, and are reachable: `select_views` must get as far as its data layer rather than dying on layout. |
| `test_surfaces.py` | Surface discovery from `Room.py`: sliver-stub exclusion, numeric wall ordering, opening keys not becoming phantom walls, no hidden wall cap, graceful degradation. |
| `test_agent_trace.py` | The agent evidence trail: one trace file per pass, edit events, MCP prefix stripping, totals, and that a broken trace never breaks a run. |
| `test_projection.py` | The pinhole projection in the vendored preprocessing: points behind the camera (which mirror onto a plausible in-frame pixel) and on the camera plane are not counted visible, valid points are unmoved, and no non-finite value reaches an int cast. That mask picks an object's reference frames, so a false positive seeds the whole object from a frame it is not in. |
| `test_module_paths.py` | Every module named as a STRING (`-m …` in subprocess calls, the CLI and the batch runners) resolves to a file, and no retired top-level spelling comes back. A string is invisible to imports, linters and renames — this exact case has broken a live run twice. |
| `test_console.py` | The stage display and the reconstruction phase's bookkeeping: no escape codes into a non-tty, a distinct colour per stage, a build stage that produced nothing marked ✗ rather than ✓, never raising on an unexpected payload, and the progress denominator counting both GLB layouts. |
| `test_scan_input.py` | A scan named or handed over as a folder must mean the same thing: the folder → (`$LR_SCANS_DIR`, name) translation, every spelling (absolute, relative, trailing slash), refusal of a non-capture directory, and that a bare name is never probed against the CWD. |
| `test_scene_package.py` | The manifest that lets stage 2 launch from a folder: relative (movable) paths, discovery from either path spelling, every stage resolving off `--scene` alone, explicit flags and an exported environment still winning, and the work room never clobbering an edited one. |

## Conventions

- **`stage_tree` (conftest) reproduces the real layout**, including the `run/<scan>/<stage>` →
  `run/<scan>/<stage>` symlinks `the CLI` creates. Most tool bugs here are *path-shape*
  bugs, and a fixture that just makes a directory named `room` cannot catch them.
- **`$LITEREALITY_SCAN` is isolated per test** (autouse fixture). `compose._config_for` *writes* it as
  a side effect, so without isolation one test's write becomes the next test's fallback and the suite
  passes for the wrong reason.
- **Async invocations are driven with `asyncio.run`** inside sync tests, so no asyncio plugin is
  needed.
- **Regression tests name the failure they came from** in the docstring. Each one was checked to fail
  against the pre-fix code — a regression test that passes either way is decoration.
