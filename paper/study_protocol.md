# User study protocol (draft): a shared, photoreal, physical room on an ordinary video call

**Status:** draft for the supervisor and the ethics board. Nothing here has been run yet.

## Research questions

- **RQ1 — Presence and understanding.** Compared with sharing the same room as a screen-shared
  video on the call, does a shared interactive room make remote participants understand the space
  better and feel more "there"?
- **RQ2 — Realism.** Does the photoreal splat room change presence, co-presence and task
  performance compared with the mesh room (photo walls plus generated objects)?
- **RQ3 — Shared physics.** Do people coordinate through the shared simulation, for example
  handing objects over and rearranging together? Where does the experience break: latency, object
  cut-outs, physics that feels wrong?
- **RQ4 — Estimated physics** (with the physics study). Do people notice the difference between
  estimated and measured physics, and does it matter to them?

## Design

- **Pairs:** 12–16 pairs (24–32 people). One person in each pair is the room's "host": the room
  is theirs, scanned beforehand with the LiteReality Scanner app. The other joins remotely.
- **Platform:** the pair talks on Zoom (audio and webcams, as usual).
- **Within-subjects, three conditions,** counterbalanced with a Latin square:
  - **C1 Video:** the host screen-shares a walkthrough video of the room in the call.
  - **C2 Mesh room:** the room call link, Photo off (quick room plus TRELLIS objects).
  - **C3 Photoreal room:** the room call link, Photo on (trained splat with cut-out objects).
- **RQ4 add-on:** a short block for 8 pairs. The same room with LiteReality's estimated masses
  and with measured masses (two scene versions), presented blind in counterbalanced order.

## Tasks (about 6 min each, a matched variant per condition)

1. **Spatial Q&A:** the remote person answers 6 questions about the room, for example "Which is
   closer to the window, the sink or the sofa?" or "How many chairs fit at the table?".
   - Scored on accuracy and time.
   - In C1 they may ask the host to point the camera.
2. **Rearrangement:** together, rearrange the room to a target layout shown only to the remote
   person, for example "make space for a 4-person meeting".
   - In C2/C3 they move objects in the shared room; in C1 they instruct the host verbally.
   - Scored on final layout error (object positions against the target), completion time and the
     number of conversational turns.
3. **Hand-over** (C2/C3 only): pass a chair from one person to the other across the room.
   - Scored on success and time, plus a log of grabs, throws and resets.

## Measures

- **Questionnaires:**
  - after each condition: presence (IPQ, short form), co-presence/social presence (the
    Networked Minds subset), NASA-TLX raw, and a single-item realism rating;
  - at the end: preference ranking.
- **System logs:**
  - from the room server: grabs, moves and resets per peer, frame round-trip latency, and the
    version and stage on screen;
  - from the client: fps and splat load time.
- **Interviews:** a 10-minute semi-structured interview at the end on what felt real or broke
  presence, and on use cases.
- **Video:** the Zoom recording (with consent), for coordination analysis.

## Analysis

- **Repeated measures across C1–C3:** linear mixed models with condition as fixed effect and
  pair/participant as random effects. Pairwise comparisons are Holm-corrected.
- **Interviews:** thematic analysis, two coders, inter-rater agreement on 20%.

## Apparatus and logistics

- **Hosting:** the room server runs on Modal (one CPU container, about $0.10/h).
- **Building each host's room:** the realistic objects (~$2) and the trained splat (~$0.30),
  built before the session.
- **Clients:** participants use their own laptops with Chrome. Phones are an exploratory
  extra.
- **Network:** no TURN relay yet, so webcams can fail on strict networks. Run sessions on
  networks where WebRTC works, or add coturn first (on the to-do list).

## Ethics and privacy

- Scans show the inside of a participant's home or office: they review the scan and may remove
  objects before the session.
- Scans and recordings are stored on the university drive, never in the public repository.
  `run/` and scans are git-ignored.
- Webcam holograms are processed on each participant's own device. The server relays video
  peer-to-peer and does not record it.
- Informed consent covers scanning, recording and log data.

## Pilot plan

Before the main study, run 2 pilot pairs to:
- check timings (target: 60 min per session, including questionnaires);
- confirm the task variants are of similar difficulty;
- check the server's capacity with 2 browsers per room (`litereality evaluate` measures how many
  rooms one CPU sustains).
