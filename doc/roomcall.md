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
- **Expected time and cost (estimates, not measured yet):** first previews about 2.5 min after
  start (including the GPU start-up), all finals about 5 min, roughly $2 of H100 time for tea_room.
- **Offline test:** `--fake` uses a stand-in generator (textured boxes), so the whole flow can be
  tried without Modal.

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
