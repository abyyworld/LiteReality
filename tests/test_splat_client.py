"""The photoreal room in the browser (roomcall web/app.js): its own functions, cut out of the page
and run under Node against stand-ins for three.js, Spark and the network. No browser here, so what
is checked is the bookkeeping: which splats stand over which meshes, and when they go."""

from __future__ import annotations

import json
import re
import shutil
import subprocess

import pytest

from litereality_agent.room_ops.roomcall import server

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="needs Node to run the page's functions")

APP_JS = (server.WEB_DIR / "app.js").read_text("utf-8")


def _block(start: int) -> str:
    """From `start` to the brace closing the first one opened after it (none of these functions
    has a brace inside a string or a regex)."""
    i, depth = APP_JS.index("{", APP_JS.index(")", start)), 0
    for j in range(i, len(APP_JS)):
        depth += {"{": 1, "}": -1}.get(APP_JS[j], 0)
        if depth == 0:
            return APP_JS[start:j + 1]
    raise AssertionError("unbalanced braces")


def _function(name: str) -> str:
    m = re.search(rf"^(?:async )?function {name}\(", APP_JS, re.M)
    assert m, name
    return _block(m.start())


def _line(prefix: str) -> str:
    m = re.search(rf"^{re.escape(prefix)}.*$", APP_JS, re.M)
    assert m, prefix
    return m.group(0)


def _keydown() -> str:
    start = APP_JS.index("addEventListener('keydown'")
    return _block(start) + ");"


HARNESS = r"""
const out = {};
const toasts = [];
const listeners = {};
const addEventListener = (t, f) => { listeners[t] = f; };
class HTMLInputElement {}
const localStorage = { setItem() {}, getItem() { return null; } };
const sessionStorage = { setItem() {}, getItem() { return null; } };
const $ = () => null;
const toast = (m) => toasts.push(m);
const showStageTag = () => {};
const applySplatView = () => {};
const retrySplat = () => {};
const console = { warn() {}, info() {}, log() {} };
const keys = new Set();
let capture = null, joined = true, welcomed = true, grab = null, sceneReady = false, bodies = new Map();
const closePanels = () => {};
const world = { kids: new Set(), add(o) { this.kids.add(o); o.parent = this; },
                remove(o) { this.kids.delete(o); if (o.parent === this) o.parent = null; } };
const disposeTree = () => {};
const spark = { failed: false };
const ver = { quiet: undefined, frame: null, gone: new Set(), pickV: undefined, pickLabel: '' };
const scans = { complete: true };
const SCAN_ENDED = new Set();
let otherWallsSays = false;
const scanJobOf = () => null;
const otherWalls = () => otherWallsSays;
const spawns = [];                         // applyManifest's spawn: true for a room new to the page
const setHover = () => {}, endGrab = () => {}, setHighlight = () => {};
const applyManifest = (man, spawn) => spawns.push(spawn);
const applyBodies = () => {}, setLoad = () => {};
const scanSwapToast = () => null, onScanVersion = () => {};
const loadProgressEl = { classList: { add() {} } };
const rooms = { list: [], pick: '', busy: '', seen: 0 };
const renderRooms = () => {}, reconsider = () => {}, onScanOffer = () => {};

// One version: a static wall (no body id) and a chair body, as the meshes the page builds.
function version(v) {
  const wall = { isMesh: true, visible: true, userData: {} };
  const chairMesh = { isMesh: true, visible: true, userData: { bodyId: 1 } };
  const chair = { kids: [], add(m) { this.kids.push(m); m.parent = this; },
                  remove(m) { this.kids = this.kids.filter((k) => k !== m); m.parent = null; } };
  const root = { traverse(f) { f(wall); f(chairMesh); } };
  return { v, manifest: {}, root, wall, bodies: new Map([[1, { id: 1, name: 'chair', node: chair }]]),
           byName: new Map([['chair', 1]]) };
}
let shown = null, sceneRoot = null;

const made = [];
class SplatMesh {
  constructor(opts) { this.opts = opts; this.opacity = 1; this.parent = null; made.push(this);
                      this.initialized = Promise.resolve(); }
  dispose() { this.disposed = true; }
}
const loadSpark = async () => ({ SplatMesh });
let answer = null;                         // what splat.json says next
const fetch = async () => {
  const a = answer;
  if (a === 'throw') throw new Error('offline');
  return { ok: a.status === 200, status: a.status, json: async () => a.body };
};
const bundle = (created) => ({ status: 200, body: {
  name: 'tea_room', created, room: 'splat/tea_room/room.spz', bodies: { chair: 'splat/tea_room/chair.spz' } } });
const settle = () => new Promise((r) => setTimeout(r, 0));
const state = () => ({ room: !!photo.room, v: photo.v, wallShown: shown.wall.visible,
                       bodies: [...photo.bodies.keys()] });

__FUNCTIONS__

(async () => {
  // The photoreal room of version "a" up.
  installVersion(version('a'));
  answer = bundle(1);
  await refreshSplat(); await settle();
  out.loaded = state();
  out.raycastable = made.map((m) => m.opts.raycastable);

  // A held P: the first press switches, the repeats do not.
  const press = (repeat) => listeners.keydown({ code: 'KeyP', repeat, target: null, preventDefault() {} });
  press(false);
  out.afterPress = photo.want;
  for (let i = 0; i < 7; i++) press(true);
  out.afterHeld = photo.want;
  press(false);
  out.afterSecondPress = photo.want;

  // Same version, a later bundle that fails: what is up stays; a 404 says there is none.
  answer = { status: 500 };
  await refreshSplat(); await settle();
  out.sameVersion500 = state();

  // Another room swapped in (someone joined into it): the old room's splat is gone at once, and
  // its meshes are drawn, before its own bundle answers (here: never).
  answer = 'throw';
  ver.quiet = 'b';
  installVersion(version('b'));
  out.otherRoomAtOnce = state();
  await settle(); await settle();
  out.otherRoomAfterFailure = state();

  // A later stage of one room: the splat stays while the new bundle loads...
  installVersion(version('c0'));
  answer = bundle(2);
  await refreshSplat(); await settle();
  answer = 'throw';
  installVersion(version('c1'));
  out.sameRoomWhileLoading = state();
  // ...but a version whose bundle cannot be had is not left under another version's splat.
  await settle(); await settle();
  out.sameRoomAfterFailure = state();

  // 404 for the version on screen: no splat for it any more.
  installVersion(version('d'));
  answer = bundle(3);
  await refreshSplat(); await settle();
  answer = { status: 404 };
  await refreshSplat(); await settle();
  out.gone404 = state();

  // A "scene" nobody picked is no news about rooms.
  onScene({ t: 'scene', v: 'e0' });
  out.unpicked = ver.pickV === undefined && rooms.pick === '';

  // Someone picks a room whose walls match this one's: it still goes in as another room. The old
  // room's splats go at once (before its own bundle answers), and everyone starts at its spawn.
  installVersion(version('e0'));
  answer = bundle(4);
  await refreshSplat(); await settle();
  out.beforePick = state();
  onScene({ t: 'scene', v: 'e1', label: 'Office room', pick: { room: 'office_room', by: 'Ada' } });
  out.pickedV = ver.pickV;
  answer = 'throw';
  const told = toasts.length;
  installVersion(version('e1'));
  out.pickAtOnce = state();
  out.pickSpawn = spawns[spawns.length - 1];
  out.pickToasts = toasts.slice(told);
  out.pickAfter = { pickV: ver.pickV === undefined ? null : ver.pickV, here: rooms.pick };
  await settle(); await settle();

  // ...and the picked room's next version is a later stage of it again: its splats stay.
  answer = bundle(5);
  await refreshSplat(); await settle();
  answer = 'throw';
  installVersion(version('e2'));
  out.pickedRoomLaterStage = state();
  out.laterSpawn = spawns[spawns.length - 1];

  // A reconnect into the same room: no news. Into a call room someone moved to another room
  // meanwhile, walls alike: that welcome's version goes in as another room too.
  const list = [{ name: 'office_room', label: 'Office room' }, { name: 'loft', label: 'Loft' }];
  noteRejoin({ pick: 'office_room', rooms: list }, 'f0');
  out.rejoinSame = ver.pickV === undefined;
  welcomed = false;                                // the first welcome is the join's own business
  noteRejoin({ pick: 'loft', rooms: list }, 'f0');
  out.firstWelcome = ver.pickV === undefined;
  welcomed = true;
  noteRejoin({ pick: 'loft', rooms: list }, 'f1');
  out.rejoinV = ver.pickV;
  const before = toasts.length;
  installVersion(version('f1'));
  out.rejoinSpawn = spawns[spawns.length - 1];
  out.rejoinToasts = toasts.slice(before);
  out.rejoinSplat = state();

  process.stdout.write(JSON.stringify(out));
})().catch((e) => { process.stderr.write(String(e && e.stack || e)); process.exit(1); });
"""


@pytest.fixture(scope="module")
def run(tmp_path_factory) -> dict:
    functions = "\n".join([
        _line("const SPLAT_URL = "), _line("const photo = "), _line("const photoShown = "),
        _line("const versionOf = "), _line("const ROOM_NAME_RE = "), _line("const pickName = "),
        *(_function(n) for n in ("disposeSplat", "clearSplat", "detachBodySplats",
                                 "attachBodySplats", "applyPhoto", "refreshSplat", "togglePhoto",
                                 "installVersion", "roomByName", "defaultRoom", "notePick",
                                 "onScene", "parseRooms", "noteRejoin")),
        _keydown(),
    ])
    # The page's `$('loadProgress')` is the only lookup installVersion needs answered.
    js = HARNESS.replace("const $ = () => null;",
                         "const $ = (id) => id === 'loadProgress' ? loadProgressEl : null;")
    path = tmp_path_factory.mktemp("splat_client") / "harness.cjs"
    path.write_text(js.replace("__FUNCTIONS__", functions), "utf-8")
    done = subprocess.run([NODE, str(path)], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_splats_load_over_the_meshes_and_are_never_raycast(run: dict) -> None:
    assert run["loaded"] == {"room": True, "v": "a", "wallShown": False, "bodies": ["chair"]}
    # pick() keeps only meshes, so Spark testing every splat on each hover would be thrown away.
    assert run["raycastable"] and all(r is False for r in run["raycastable"])


def test_holding_p_switches_the_photoreal_room_once(run: dict) -> None:
    assert run["afterPress"] is False
    assert run["afterHeld"] is False
    assert run["afterSecondPress"] is True


def test_a_failed_refresh_keeps_this_versions_own_splats(run: dict) -> None:
    assert run["sameVersion500"]["room"] is True and run["sameVersion500"]["v"] == "a"


def test_another_room_is_not_drawn_under_the_old_rooms_splats(run: dict) -> None:
    assert run["otherRoomAtOnce"] == {"room": False, "v": None, "wallShown": True, "bodies": []}
    assert run["otherRoomAfterFailure"] == run["otherRoomAtOnce"]


def test_a_later_stage_keeps_the_splats_until_its_bundle_fails(run: dict) -> None:
    assert run["sameRoomWhileLoading"]["room"] is True
    assert run["sameRoomWhileLoading"]["v"] == "c0"
    assert run["sameRoomAfterFailure"] == {"room": False, "v": None, "wallShown": True,
                                           "bodies": []}


def test_a_404_takes_the_splats_away(run: dict) -> None:
    assert run["gone404"]["room"] is False and run["gone404"]["wallShown"] is True


def test_a_picked_room_goes_in_as_another_room_even_with_the_same_walls(run: dict) -> None:
    assert run["unpicked"] is True
    assert run["beforePick"]["room"] is True and run["beforePick"]["v"] == "e0"
    assert run["pickedV"] == "e1"
    # The old room's splats are gone at once and its meshes drawn; everyone starts at the spawn.
    assert run["pickAtOnce"] == {"room": False, "v": None, "wallShown": True, "bodies": []}
    assert run["pickSpawn"] is True
    assert run["pickToasts"] == ["Now in Office room"]
    assert run["pickAfter"] == {"pickV": None, "here": "office_room"}
    # Only that one swap: the picked room's own next stage keeps its splats and people in place.
    assert run["pickedRoomLaterStage"]["room"] is True and run["pickedRoomLaterStage"]["v"] == "e1"
    assert run["laterSpawn"] is False


def test_a_reconnect_into_a_room_moved_meanwhile_goes_in_as_another_room(run: dict) -> None:
    assert run["rejoinSame"] is True and run["firstWelcome"] is True
    assert run["rejoinV"] == "f1"
    assert run["rejoinSpawn"] is True and run["rejoinToasts"] == ["Now in Loft"]
    assert run["rejoinSplat"] == {"room": False, "v": None, "wallShown": True, "bodies": []}


VIEW_HARNESS = r"""
const out = {};
let eyeY = 1.6;
const vec = (x, y, z) => ({ x, y, z, distanceTo(o) { return Math.hypot(this.x - o.x, this.y - o.y, this.z - o.z); } });
const walker = { pos: vec(0, 1.6, 0), yaw: 0, pitch: 0, spawnFrom: null };
const mjToThree = (x, y, z, o) => { o.x = x; o.y = z; o.z = -y; return o; };   // MuJoCo z-up -> three y-up
const clampWalker = () => {};
__FUNCTIONS__
const view = { pos: [2, 3, 0.1], yaw: 1.2, pitch: -0.9 };
applySplatView({ view });
out.noSpawn = [walker.pos.x, walker.yaw];                 // never spawned: stays put
walker.spawnFrom = vec(0, 1.6, 0);
applySplatView({ view: { pos: [2, 3], yaw: 1, pitch: 0 } });
out.badView = walker.spawnFrom !== null && walker.pos.x === 0;   // malformed: ignored, still armed
applySplatView({ view });
out.moved = { near: Math.hypot(walker.pos.x - 2, -walker.pos.z - 3) <= 0.31,
              y: walker.pos.y, yaw: walker.yaw, pitch: walker.pitch, once: walker.spawnFrom === null };
walker.spawnFrom = vec(5, 1.6, 5);                        // spawned, then walked away
applySplatView({ view });
out.walked = walker.spawnFrom !== null;
process.stdout.write(JSON.stringify(out));
"""


def test_a_newcomer_starts_at_the_scans_best_view_until_they_walk(tmp_path) -> None:
    functions = "\n".join([_line("const clamp = "), _line("const finite = "),
                           _function("applySplatView")])
    path = tmp_path / "view.cjs"
    path.write_text(VIEW_HARNESS.replace("__FUNCTIONS__", functions), "utf-8")
    done = subprocess.run([NODE, str(path)], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    got = json.loads(done.stdout)
    assert got["noSpawn"] == [0, 0] and got["badView"] is True
    assert got["moved"] == {"near": True, "y": 1.6, "yaw": 1.2, "pitch": -0.35, "once": True}
    assert got["walked"] is True
