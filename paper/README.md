# Paper (draft): RoomCall, a live, photoreal, physically shared room on ordinary video calls

Target: UIST 2027 (or ISMAR 2027). System + user study.

| File | What |
|---|---|
| `main.tex` | The draft (ACM `acmart`, sigconf, anonymous review). Measured numbers come from `results/`; `\todo{}` marks what is not done |
| `refs.bib` | References. **Every entry needs checking** against the publisher's page; some came from a web survey and are marked unverified |
| `study_protocol.md` | User-study design: conditions, tasks, measures, analysis, ethics |

Build the PDF (needs a TeX installation with `acmart`):

```bash
cd paper && latexmk -pdf main.tex
```

## Where each number comes from

| Claim | Source |
|---|---|
| Build timeline and costs | `results/tea_room/realistic_run.md`, `results/tea_room/splat_run.md`, `run/tea_room/splat/report.json` |
| Fidelity table | `litereality evaluate run/tea_room --render` → `run/<scan>/eval/comparison.md` |
| Physics study | `scripts/physics_study/llm_estimates.py` → `results/<scan>/physics_study.csv`; protocol in `doc/physics_study.md` |
| Figures | The room call with `?debug=1`: `window.__roomcall.look(from, at)` places the camera exactly (MuJoCo coordinates), and `scripts/roomcall/remote_grab.py` moves an object through the shared physics for before/after shots |

## Still to do before submission

- [ ] Scan the author's own room, and two or three more rooms, so the paper isn't one room.
- [ ] Run the user study (`study_protocol.md`), after ethics approval.
- [ ] Physics study: measure tea_room's objects (or the new rooms') with the bench protocol.
- [ ] Physics/capacity numbers from `litereality evaluate` into §5.2.
- [ ] Architecture figure; before/after interaction figure; progressive-build figure (instant →
      trained).
- [ ] Check every reference.
