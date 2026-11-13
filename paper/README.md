# Paper (draft): RoomCall, a live, photoreal, physically shared room on ordinary video calls

Target: UIST 2027 (or ISMAR 2027). System + user study.

| File | What |
|---|---|
| `main.tex` | The draft (ACM `acmart`, sigconf, anonymous review). Measured numbers come from `results/`; `\todo{}` marks what is not done |
| `refs.bib` | References, each checked on 2026-11-13 against Crossref/DOI, arXiv, CVF, JMLR or the project's repository; the comment above each entry says where. An `UNVERIFIED` note prints where the draft's description does not match the source |
| `figures/` | TikZ figures (architecture, build timeline) and `figures/README.md` with the blocks to paste into `main.tex` |
| `study_protocol.md` | User-study design: conditions, tasks, measures, analysis, ethics |

Build the PDF (needs a TeX installation with `acmart` and TikZ):

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
- [x] Architecture figure and build-timeline figure (`figures/`; not yet compiled, and not yet
      pasted into `main.tex`).
- [ ] Before/after interaction figure; progressive-build screenshots (instant → trained).
- [x] Check every reference (2026-11-13). Open points: GaussianNexus "basic physics" not
      confirmed, Croquet is replicated rather than per-client physics, Hyperfy runs physics on the
      server too (see `figures/README.md`, section 4).
