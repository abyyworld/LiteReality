# Room call — a shared physics room for people on a video call

People on a Zoom, Google Meet or Teams call (or on no call at all) open one link in a browser and
land together inside a LiteReality-scanned room. Each person is an avatar with their webcam on its
face; they walk around, grab, drag and throw objects, and open doors and drawers. The room's
physics runs **once, on the server** (MuJoCo, with LiteReality's estimated masses, friction and
joints), so everyone sees the same thing. The call keeps carrying everyone's voice; the room call
adds the room.

Code: `src/litereality_agent/room_ops/roomcall/` — `sim.py` (shared physics), `geometry.py` (scene
→ glb + manifest), `server.py` (HTTP + WebSocket), `web/` (browser client), `demo.py` (built-in
room).

## Run it

```bash
uv run litereality roomcall --demo                 # built-in demo room, opens a browser
uv run litereality roomcall run/tea_room           # a scanned room (finds mujoco_seed/scene.xml)
uv run litereality roomcall path/to/scene.xml --port 8790 --label "Tea room"
```

`SCENE` can be any of these:
- a `scene.xml`;
- a folder holding `scene.xml`, `mujoco_seed/scene.xml` or `mujoco/scene.xml` (the output of
  `litereality stage simulate run/<scan> --from-seed`);
- a scan itself: a scan folder, its `room.usdz`, or a scan name from `$LR_SCANS_DIR` (see below).

Each `?room=<id>` in the URL is its own copy of the room with its own physics (default `lobby`).

## From a fresh scan to the call in seconds

The full pipeline takes an hour or more per room. That's too slow if you scan a room and then
want to hop on a call, so the room call can also build a **quick room** straight from the scan:

```bash
uv run litereality roomcall my_room             # scan name (in $LR_SCANS_DIR) or scan folder
uv run litereality roomcall my_room --publish   # same, onto the deployed link (seconds)
uv run litereality quickroom my_room            # just write run/my_room/quick/scene.xml
```

The quick room is built from the scan's RoomPlan data (`room.usdz`) with no Blender and no AI.
tea_room takes about 2 s. It has:
- the real walls, with the door and window cut out, and the floor and ceiling;
- a door leaf on a hinge, and glass in the windows;
- every object at its scanned position and size, built from simple parts for its category: a table
  is a top on four legs (so tucked chairs still fit under it), a chair is a seat, back and legs;
- colours sampled from the scan's photos;
- masses from the same category density table the full pipeline uses. For tea_room's chairs and
  dishwasher they come out identical to the full pipeline's.

Chairs, stools and small tables can be moved; heavy and wall-hung furniture is fixed in place.

It looks like a blocky sketch of the room, not the real furniture. When the full pipeline later
finishes for the same scan, publish its scene and the link switches to it live:

```bash
uv run litereality roomcall run/my_room --publish   # serves mujoco_seed/ once it exists
```

## Scan straight into the call (no laptop)

During a call, open the room link on the phone you scanned with, tap **Rooms → Add a room you
scanned…** and pick the scan:
a zip of the scan folder (in Files, long-press the folder → Compress), or a bare `room.usdz`
(flat colours, no photos).
1. **Seconds later, the quick room:** everyone in that room id is standing in the scanned room, with
   the photos on its walls and floor. A local test took ~6–8 s from upload to the room being
   offered. Other rooms are not affected.
2. **Then, realistic objects:** with `--uploads realistic` (the default when Modal credentials are
   set), objects stream in as TRELLIS finishes them, with progress shown to everyone ("objects
   3/9…").

```bash
uv run litereality roomcall --demo                          # uploads on; realistic if Modal is set up
uv run litereality roomcall --demo --uploads quick          # quick rooms only, no GPU
uv run litereality roomcall --demo --fake-objects           # try the whole flow without Modal
```

Safety and limits:
- **Before the upload is read:** the server checks the person's key, cooldowns, a per-room build
  slot, per-address and server-wide caps, and a daily cap on realistic builds (6 by default,
  hosted). Over the cap, people still get the quick room.
- **Zip checks:** at most 200 MB uploaded and 1 GB unpacked; no `..`, absolute paths, symlinks or
  >100× compression; only the scanner's file types. Prim names in the usdz must be identifiers.
- **Builds:** they run as niced child processes with timeouts. A newer upload replaces an older
  build, and a room that stays empty for 60 s drops its scan.
- **Hosted:** each room's latest scan is kept on the Volume, so a container restart restores it.

Tried on the hosted link with `scripts/roomcall/upload_scan.py` (quick room ~10 s, photoreal ~22 s
after the upload). Not tried yet: a real iPhone upload over a phone network.

## Ready-made rooms: the room picker

Not everyone has a scan. The hosted link keeps a small library of finished rooms, each with its
photoreal splat, and anyone in a call can move everyone in that call room into one:
- **Joining an empty call room:** the join card shows the rooms as tiles; the one picked is where
  everyone who joins next lands.
- **During a call:** **Rooms** lists them; press a room, then press it again to move everyone
  there (the default room is always first). The same panel has **Add a room you scanned…**.
- A picked room is let go like a scanned one: once the call room has been empty for a minute, it
  is back on the link's default room. Picks are refused while a scan is being built there.

Adding a room to the library (it does not change the link's default room, and needs no redeploy):

```bash
uv run litereality roomcall run/office_room/objects/scene_splat.xml --library --label office_room --thumb run/office_room/thumb.jpg
```

```bash
uv run litereality splatroom ~/LiteReality-Agent/example-scans/office_room --label office_room --out run/office_room/splat --publish --upload-only
```

```bash
uv run litereality roomcall --library-remove office_room
```

The first command uploads the room's physics and thumbnail (a JPEG under 1 MB) to the Volume's
`.library/`; the second uploads its photoreal bundle to `splats/office_room/`, where the server
finds it by the room's name. The label has to match in both. Within ~20 s of a change, anyone
opening Rooms sees it. Versions the library names are never pruned by later publishes.

## Realistic objects, streaming in (needs Modal)

```bash
uv run litereality setup                                    # once: deploys TRELLIS with 4 GPUs
uv run litereality roomcall my_room --realistic --deploy    # first time; later --publish
uv run litereality realroom my_room                         # offline: run/my_room/objects/scene.xml
```

`--realistic` puts the quick room on the link first. It then generates each object from its own
photos in the scan, on TRELLIS on Modal with up to 4 GPUs in parallel. Each object goes through two
passes, a fast preview and then a full-quality final, and each new version swaps into the call live.

- **Inputs:** each object's best views are cut out with its depth mask, refined with GrabCut, and
  sent with a real alpha channel. Dark objects therefore come out solid.
- **Fitting:** the generated mesh is turned to face the right way (a quarter-turn search scored
  against the object's depth masks) and scaled into the scanned box.
- **What you see vs what collides:** the mesh is what you see. The quick room's simple parts stay as
  what you bump into, and mass and joints are unchanged, so tucked chairs still fit under the table.
- **Shortcuts:** identical objects share one mesh. An object seen too little (e.g. a table only seen
  from above) keeps its template.
- **Measured time and cost (tea_room, 2026-11-13):** the first preview arrived 2 min after
  generation started (including the GPU start-up) and the last final at 7.2 min, for $2.24 of
  Modal time. Details: [`results/tea_room/realistic_run.md`](../results/tea_room/realistic_run.md).
- **Offline test:** `--fake` uses a stand-in generator (textured boxes), so the whole flow can be
  tried without Modal.

## The real room, photoreal: Gaussian splats (needs Modal for the sharp version)

```bash
uv run modal deploy --env main deploy/modal/splat/app.py          # once: the gsplat trainer
uv run litereality splatroom my_room                              # instant splat, then trained on Modal
uv run litereality roomcall run/my_room/objects/scene.xml --splat run/my_room/splat
uv run litereality splatroom my_room --publish                    # onto the deployed link, live
uv run litereality splatroom my_room --instant-only               # no GPU, free
```

The room is drawn as Gaussian splats made from the scan's own photos. You see the real room,
not reconstructed meshes. It grows live:

1. **Instant (no GPU, ~10 s):** every LiDAR depth pixel of every photo becomes a small flat disc,
   coloured from the photo and fused on a voxel grid (`models/splat/rgbd.py`). tea_room comes to
   640k splats, a 6 MB SPZ file.
2. **Trained (Modal H100):** gsplat 1.5.3's own trainer (`examples/simple_trainer.py`, unchanged)
   runs on a training set written straight from the scan, with no COLMAP run
   (`models/splat/dataset.py`):
   - ARKit poses and intrinsics as cameras;
   - LiDAR points to start from and to supervise depth;
   - a per-photo bilateral grid for the phone's exposure changes;
   - MCMC densification capped at 800k splats.
   A snapshot streams back every few hundred to thousand steps, and each swaps into the call.
   Every 8th photo is held out, so each room also gets PSNR/SSIM/LPIPS.

**Objects you can grab are real too.** Each bundle is cut against the room's MuJoCo scene
(`pipeline/splatroom/split.py`):
- each movable body gets the splats on its geometry, in its own frame;
- the browser hangs those under the body's node, so grabbing the real chair moves the real chair.

**How objects are cut out** (`pipeline/splatroom/masks.py`, `deploy/modal/segment`):
1. Each body's box is projected into the photos that see it best, grown by 30% because RoomPlan
   boxes chairs 20–30 cm off.
2. SAM 2.1 returns three candidate outlines per box. The best one not lying mostly on static
   furniture is kept, so a chair tucked under a table doesn't come back as the table.
3. Every nearby splat votes by projection, with a LiDAR visibility test.
4. The body is moved to the median of its own splats, measured on the LiDAR room, but never
   deeper into a wall than it was. The moved physics is written as `scene_splat.xml` and
   published with `--publish`.
5. The floor under each object, never photographed, is patched from the floor around it.

Without Modal (`--no-masks`), boxes are used, snapped to where they hold the most splats.

- `--recut` cuts the last trained snapshot again (new masks, new rules) without training.
- `--keep-physics` never moves a body: for a room already live.

**Scanned during a call:** when someone adds a scan under **Rooms**, the server also builds that room's
splat beside the quick room and the objects.
- Locally, measured on tea_room: quick room at 4.6 s after the upload, photoreal at 16.7 s.
- On the deployed link: quick room at 10 s, photoreal at 22 s, the first trained snapshot at 96 s.

**Tools:**
- `scripts/roomcall/remote_grab.py` carries an object through the shared physics as a scripted
  participant.
- `scripts/roomcall/upload_scan.py` sends a scan into a room and prints the build timeline.
- `?debug=1` in the page exposes `window.__roomcall` (what's shown, and a camera you can place
  exactly) for checks and figures.

**In the call:**
- The **Photo** button (or **P**) switches between the splats and the meshes. The meshes still
  pick and collide either way.
- The server watches the bundle folder and tells every room when a newer snapshot lands
  (`{"t": "splat"}`). Browsers load the new files beside the old ones, then swap.

**Gaps:**
- Where a moved object stood there is a hole: nothing was ever captured behind it.
- Objects tucked under a table are hard to separate with boxes alone; per-photo masks (SAM 2) are
  the next step.

### Opening cabinets, doors and drawers (Infinigen)

```bash
uv run litereality roomcall my_room --realistic --articulated      # add --drawers Storage3 for a chest
```

With `--articulated`, every door, window, cabinet, dishwasher, fridge and oven becomes an
[Infinigen](https://github.com/princeton-vl/infinigen) asset (BSD-3) sized to its scanned box, with
working hinges and slides. It runs on the CPU:
- **Time:** about 8–45 s per asset; tea_room's 7 assets take 83 s with 3 workers, and a cached
  rebuild about 10 s.
- **Physics:** masses come from our density table and joint settings from our defaults, so a hand
  can open them. Colours are tinted from the scan photos.
- **Where it runs:** locally with `LR_INFINIGEN_PYTHON` exported to an Infinigen env's python (setup
  in `deploy/modal/infinigen/README.md`), or on Modal after
  `uv run modal deploy --env main deploy/modal/infinigen/app.py`. The Modal path has not been run yet.
- **Gaps:** no counter, sink or corner units; no mixed door-and-drawer unit; at most 2 cabinet doors;
  fridges have one door.

An asset Infinigen fails on is handed back to TRELLIS.

## You as a hologram

With the camera on, everyone else sees you as a life-size cut-out of your real webcam video, standing
at your place in the room and turned to face them, instead of a video square.

- **On your own device:** your browser cuts you out of your background with MediaPipe's selfie
  segmenter (0.10.35, Apache-2.0). The model is served by the room call itself; nothing goes to
  Google. Newer MediaPipe versions send usage statistics, which is why 0.10.35 is pinned.
- **What's sent:** the colour image and the cut-out mask, side by side in one 640×240 video. The
  mask survives video compression, and the stream costs about as much bandwidth as the plain webcam
  (~0.8 Mbit/s per person).
- **Toggle:** the **Hologram** button turns it off and on; the corner shows your own cut-out.
- **When it falls back to the plain webcam on the face plane:**
  - while the model loads;
  - when it can't load (blocked CDN, no WebAssembly or WebGL);
  - on a device too slow for it (after 8 s; click Hologram to retry);
  - when nobody is in frame;
  - while the tab is in the background.

Limits:
- It's flat video. Room lights don't affect it, it casts no shadow, and you never see someone's side
  or back.
- Arms at the frame edge are cut off.
- The first cut-out freezes the page for about half a second while the segmenter starts.
- Full 3D photoreal people would need something like LAM (a one-shot animatable 3D head, SIGGRAPH
  2025, Apache-2.0). That isn't built.

## Full-body people

A webcam frames only head and shoulders, so on its own the hologram floats. When you join, the
**body** button (the figure in the toolbar) gives you a 3D body:

1. **Make your body.** Step back about 2 m and turn slowly through a 3-2-1 countdown (the page grabs
   a few frames), or upload one or more photos.
2. **The server builds it:**
   - `--avatars standin` (the default): a simple splat body on our skeleton, tinted with your
     clothes' colours, in about 2 s, with no GPU. It isn't photoreal.
   - `--avatars modal`: a photoreal body from your photos, made by LHM++ on a Modal L4 GPU in about
     5–15 s warm (1–2 min cold). You deploy it once:
     ```bash
     uv run modal run --env main deploy/modal/avatar/app.py::populate     # image + weights, once
     uv run modal deploy --env main deploy/modal/avatar/app.py
     uv run litereality roomcall my_room --avatars modal
     ```
     See `deploy/modal/avatar/README.md`. The LHM++ weights are CC-BY-NC-4.0 and SMPL-X is
     non-commercial, which is fine for research but not for a product.
3. **Everyone sees your body move live:**
   - Spark 2.3.1 draws it in every browser, on three.js 0.180.
   - Your upper body follows MediaPipe pose tracking in your own browser, sent as joint rotations
     with your position.
   - Your legs walk when you move.
   - Your live hologram face sits on the body's head.

Limits and protections:
- **Limits:** uploads are rate-limited per person, per room, per client address and server-wide.
  GPU builds have a daily cap and fall back to the stand-in.
- **Tried:** a real GPU body from LHM++'s sample photos (120k splats, 4.1 s of inference,
  80 s cold). **Not tried yet:** a real webcam turnaround and real hardware.
- **Speed:** two bodies took a software-rendered test page from about 18 to 8 fps; real GPUs
  should do far better.

## Switching rooms live, mid-call

Every publish is a new **version** of the same link. People in the room are never disconnected:
- their browser fetches the new version in the background and swaps it in;
- objects keep the positions people moved them to (matched by name), and an object someone is
  holding stays in their hand;
- everyone keeps their camera position and their webcams;
- objects that don't exist in the new version disappear, and new ones appear at their scanned
  position.

- `--publish` uploads a new version and switches the deployed link to it within about 10 s. It
  doesn't redeploy anything, so it's safe during a call.
- `--deploy` publishes and also redeploys the server code. Use it the first time and after updating
  this repo, not during a call.
- `--stage quick|objects|full|scene` labels the version. People see a "quick preview" tag for
  quick rooms. By default it comes from the folder (`quick/` is quick, `mujoco_seed/` is full).
- `--watch` does the same locally, without Modal. It re-reads `SCENE` whenever its files change, so
  you can try a swap on your own machine.

In the room: **WASD / arrows** walk (Shift hurries, Q/E turn), **drag empty space** to look,
**click and drag an object** to grab it (doors and drawers too), the **mouse wheel** pulls or pushes
what you hold, **let go while moving** to throw. On a phone: drag to look, press and hold an
object to grab, joystick or two fingers to walk. **Reset** puts the room back for everyone.

## Use it with Zoom, Meet or Teams: a permanent link (recommended)

Deploy the room server to your Modal workspace once. It gets a permanent `https://` link, so
nobody on the call has to install anything, and no tunnel is needed:

```bash
uv sync --frozen --extra modal --group dev        # once, if the Modal client isn't installed
uv run litereality roomcall run/tea_room --deploy  # upload the scene, deploy, print the link
uv run litereality roomcall --demo --deploy        # or serve the demo room
```

It uses the Modal token from `.env` (`MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET`, or `MODAL_PROFILE`),
the same as `litereality setup`. It prints two links:
- the server's address, which stays the same across redeploys;
- a link into a fresh room with an unguessable `?room=` id.

Paste the second link into the call chat. Everyone who opens it lands in the same room, with their
webcam on their avatar and the same physics. That's because the link is https.

- **What gets uploaded:** the scene goes into the Modal Volume `litereality-roomcall-scenes`. Only
  `scene.xml` and the meshes, textures and include files it names are sent, never the rest of the
  export folder.
- **Switching scenes:** `--publish` with another scene (or a scan) switches the link live, with
  no redeploy (see "Switching rooms live, mid-call").
- **The server:** it runs as its own Modal app, `litereality-roomcall`
  (`deploy/modal/roomcall/app.py`), in one CPU container that holds every room.
- **Cost:** the container shuts down 15 minutes after the last person leaves, so an unused link
  costs nothing. The first visit after that waits a few seconds while it starts. While people are
  in it, it uses 2 CPUs and 2 GB, roughly $0.10 an hour at Modal's CPU rates.
- **Stop it:** `uv run modal app stop litereality-roomcall`.

## Use it with Zoom, Meet or Teams: from your own machine

The local server listens on `127.0.0.1` only. For other people on the call to reach it, give it an HTTPS
address. HTTPS is also what browsers require before they allow a webcam.

```bash
uv run litereality roomcall run/tea_room --no-open
cloudflared tunnel --url http://127.0.0.1:8790     # or: ngrok http 8790
```

Open the `https://…` address the tunnel prints, join, press **Invite** and paste the link into the
call chat. Everyone who opens it lands in the same room. When asked, keep **Send my microphone**
off, because the call already carries your voice (otherwise you get echo). Anyone who can't open
the link can follow along when you share your browser tab in the call.

## Run it inside Zoom as a Zoom App (optional)

Inside the Zoom client the page reads the meeting ID, so everyone in the meeting lands in the same
room automatically and uses their Zoom name. The code for this is written and was checked against a
stand-in SDK only. It has **not** been tried in the real Zoom client yet: that needs a Zoom
developer account.

1. Go to marketplace.zoom.us → **Develop → Build App** and create a **General App** (user-managed).
2. Under **Features → Surface**, select **Meetings**.
3. Set the **Home URL** to your tunnel address followed by `/?zoom=1`. The `?zoom=1` makes sure the
   page switches to Zoom mode even if Zoom's user-agent string changes.
4. Add the tunnel's domain to the **Domain Allow List**. Also add `cdn.jsdelivr.net` (three.js)
   and `appssdk.zoom.us` (the Zoom Apps SDK).
5. Under **Zoom App SDK**, add the APIs `getRunningContext`, `getMeetingUUID` and
   `getUserContext`.
6. Fill in the OAuth redirect URL that Zoom asks for. The tunnel address is enough for testing,
   because the room call does not use the OAuth token.
7. Use **Local Test → Add App** to install it on your own account.
8. Start a meeting and open it from **Apps**.

The server already sends the headers Zoom Apps require:

- `Strict-Transport-Security`
- `X-Content-Type-Options`
- `Referrer-Policy`
- a `Content-Security-Policy` that allows only these: the page's own import map, three.js, the Zoom SDK,
  and the pinned MediaPipe runtime. All three come from jsDelivr or Zoom, which are already on the
  domain allow list. It also allows WebAssembly (`'wasm-unsafe-eval'`) for the hologram's segmenter;
  JavaScript `eval` stays blocked.

Inside Zoom the microphone option is off, because Zoom carries the audio. The webcam on your avatar
may be unavailable while Zoom itself is using the camera.

## What to know

- **The physics is LiteReality's estimate.** Masses, friction and joints come from each object's
  category, size and material and were never measured (see `results/tea_room/`). The tea_room door
  leaf, for example, simulates at 128.7 kg. MuJoCo ignores restitution, so nothing bounces.
- **Changes are virtual only.** Opening a drawer in the room call does not move the real one.
- **The servo-held room body is static.** An exported room hangs its walls and fixtures on a 50 t
  `room` body pinned by position servos (the shake rig). The room call treats that body as static
  and never lets anyone grab it. Doors and drawers, which have motors, and free objects can be
  moved.
- **Video between people is peer-to-peer.** It uses WebRTC with a public STUN server and no TURN
  relay. On networks that block peer-to-peer traffic (some university and corporate networks), the
  webcams on the avatars won't connect. The room, the avatars and the physics still work, because
  they go through the server. Peer-to-peer suits a handful of webcams; the server allows 16 people
  per room.
- **Walking has no collision.** Your camera is kept inside the room's bounds, so you can pass
  through furniture.
- **Rooms are not saved.** A room's physics stops, and the room is dropped, when its last person
  leaves. **Reset** returns it to the settled start state.
