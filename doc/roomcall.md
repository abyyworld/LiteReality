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

`SCENE` may be a `scene.xml` or a folder holding `scene.xml`, `mujoco_seed/scene.xml` or
`mujoco/scene.xml` — the output of `litereality stage simulate run/<scan> --from-seed`. Each
`?room=<id>` in the URL is its own copy of the room with its own physics (default `lobby`).

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
- **Switching scenes:** uploading another scene, or `--demo`, and redeploying switches the link.
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
- a `Content-Security-Policy` that allows only the page's own import map, three.js and the Zoom SDK

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
