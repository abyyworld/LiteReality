# Figures

**Both figures are now in `main.tex`, with the faster trainer's numbers (final splat at 104 s);
the blocks below are the originals, kept for reference.**

TikZ figures for `paper/main.tex`. Each `<name>.tex` holds a single `tikzpicture` and is meant to
be pulled in with `\input`. Each `standalone-<name>.tex` wraps one figure so it compiles on its own.

| File | Figure | Width |
|---|---|---|
| `architecture.tex` | System architecture: scan → builders → versioned room → one server → browsers, with the video call alongside | full text width (`figure*`), about 17.2 cm |
| `timeline.tex` | Progressive build of tea_room on a log time axis | one column (`figure`), about 7.8 cm |

**Not compiled yet.** This machine has no TeX (`latexmk`, `pdflatex`, `tectonic` and a TeX
Docker image are all missing), so neither figure has been rendered. Compile both before using them:

```bash
cd paper/figures && latexmk -pdf standalone-architecture.tex && latexmk -pdf standalone-timeline.tex
```

To get TeX quickly on macOS, `brew install tectonic` works, then `tectonic standalone-timeline.tex`.
The standalone wrappers use the article font (10 pt Computer Modern), so text in them looks a
little larger than it will in acmart (9 pt Libertine).

Where the numbers come from: `timeline.tex` gives the source of each number in its header
comment (paper Table `tab:timeline`, `results/tea_room/splat_run.md`,
`results/tea_room/realistic_run.md`).

## 1. Preamble (once)

In `main.tex`, after `\usepackage{xcolor}`:

```latex
\usepackage{tikz}
\usetikzlibrary{arrows.meta,shapes.geometric}
```

## 2. Architecture figure

Where: at the start of `\section{The \sys{} System}`. Replace the
`\todo{architecture figure: ...}` paragraph with this block. Then refer to it in the text with
`Figure~\ref{fig:architecture}`.

```latex
\begin{figure*}[t]
  \centering
  \input{figures/architecture}
  \caption{\sys{} architecture. A phone scan is uploaded to builders: Modal GPUs run the gsplat
  trainer, SAM~2 object masks and TRELLIS, and a CPU (the laptop or Modal) builds the quick room,
  the instant splat and Infinigen's openable furniture. Each stage publishes a new version of the
  room to a Modal Volume. One server container loads it and runs one MuJoCo simulation per room,
  sending body poses to every browser over a WebSocket and taking grabs back; rooms are created
  by link. Browsers draw meshes with three.js and splats with Spark. Webcams go browser to browser
  over WebRTC, and the video call carries voice as usual.}
  \Description{Block diagram, left to right: phone scan; builders split into a GPU group (gsplat
  trainer, SAM 2, TRELLIS) and a CPU group (quick room, instant splat, Infinigen); a Modal
  Volume holding versions of the room; one server container with FastAPI, WebSocket and MuJoCo;
  three browsers with two-way arrows to the server and dashed WebRTC links between them. A bar
  under the server and browsers stands for the Zoom, Meet or Teams call.}
  \label{fig:architecture}
\end{figure*}
```

## 3. Progressive build timeline

Where: in `\section{From Scan to a Photoreal, Interactive Room}`, directly after the
`tab:timeline` table (after its `\end{table}`). It shows the same numbers as the table, so refer
to both together, e.g. "Table~\ref{tab:timeline} and Figure~\ref{fig:timeline}".

```latex
\begin{figure}[t]
  \centering
  \input{figures/timeline}
  \caption{Progressive build of tea\_room (62 photos), on a log time axis. Each stage is timed
  from its own start; all start when the scan arrives and run in parallel. The trained splat
  streams snapshots from step 2000 (81--102\,s) to the final splat at 184\,s; generated objects
  arrive one at a time between 2.0 and 7.2\,min (circle: all previews, 275\,s). Bottom row: the
  deployed system when a room is scanned during a call, timed from the start of the upload.}
  \Description{Horizontal timeline with a logarithmic time axis from 1 second to 10 minutes.
  Quick room ready at 2 seconds; instant splat at 10 seconds; trained splat training from
  29 seconds, first snapshot at 81 to 102 seconds and final at 184 seconds; openable furniture at
  91 seconds; generated object meshes from 2.0 to 7.2 minutes. A separate row for a live call
  marks the mesh room at 10 seconds, the photoreal room at 22 seconds and the first trained
  snapshot at 96 seconds.}
  \label{fig:timeline}
\end{figure}
```

## 4. Related-work citations that are now available

`refs.bib` now has `hubs`, `croquet`, `hyperfy` and `crossgaussian`, for the two `\todo`s in the
introduction and Related Work. Wording to check before citing them (details in the comments in
`refs.bib`):

- **Croquet** replicates one deterministic computation on every client; it does not simulate
  "on each client separately".
- **Hyperfy** runs its PhysX simulation on both the server and every client, so it is not a clean
  example of client-side physics. Its entry prints an `UNVERIFIED` note until the text changes.
- **Hubs** does fit: each client runs ammo.js physics and objects are synced over the network.
  Mozilla shut down the hosted service on 31 May 2024; the code lives on with the Hubs Foundation.
- **GaussianNexus**: "basic physics between a HoloLens and a Quest user" is not confirmed. The
  HoloLens + Quest part is right; the abstract talks about segmenting, tracking and syncing
  moved objects, not physics.
- **AvatarPilot** is a 4-page ISMAR *Adjunct* paper, not a full ISMAR paper.
