"""Grabbing in the photoreal room (roomcall web/app.js), run under Node like test_splat_client: the
body picked is tinted on its own splat (its meshes are hidden under it, so tinting them alone shows
nothing), the tint follows a new bundle's splat, and a body the bundle marks as holding no real
object (splat.json "grabbable": {name: false}) is never offered for grabbing."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_splat_client import NODE, _function, _line  # noqa: E402

pytestmark = pytest.mark.skipif(NODE is None, reason="needs Node to run the page's functions")

HARNESS = r"""
const out = {};
const BACKDROP = true;
const depthCut = { visible: false };
const showStageTag = () => {}, $ = () => null, sceneRoot = null;
let sceneReady = true, grab = null, hoverBody = null;
const inStep = () => true;
const setNdc = () => {};
let hits = [];
const raycaster = { intersectObject: () => hits };
class Recolor { constructor() { this.rgb = [1, 1, 1]; } setRGB(r, g, b) { this.rgb = [r, g, b]; } }
const splat = () => ({ recolor: new Recolor(), opacity: 1, parent: null });
const node = () => ({ add(m) { m.parent = this; } });
const mat = () => ({ emissive: { r: 0, setRGB(r) { this.r = r; }, copy(c) { this.r = c.r; } },
                     userData: { emissive0: { r: 0 } } });
const bodies = new Map([[1, { id: 1, name: 'Chair0', node: node(), mats: [mat()] }],
                        [2, { id: 2, name: 'Chair1', node: node(), mats: [mat()] }],
                        [3, { id: 3, name: 'Lamp0', node: node(), mats: [mat()] }]]);
let shown = { bodies, byName: new Map([['Chair0', 1], ['Chair1', 2], ['Lamp0', 3]]) };
__FUNCTIONS__
const applyPhoto = () => {};
photo.room = {};
photo.bodies = new Map([['Chair0', splat()], ['Chair1', splat()]]);
photo.grabbable = new Map([['Chair0', true], ['Chair1', false]]);
const hit = (id) => [{ object: { isMesh: true, visible: false, userData: { bodyId: id, photoHidden: true } },
                       point: { clone() { return this; } }, distance: 1 }];

setHighlight(1, true);
out.grabbedTint = photo.bodies.get('Chair0').recolor.rgb;
out.otherTint = photo.bodies.get('Chair1').recolor.rgb;
out.meshTint = bodies.get(1).mats[0].emissive.r;
setHighlight(1, false);
out.releasedTint = photo.bodies.get('Chair0').recolor.rgb;
setHighlight(3, true);                     // a body with no splat: its meshes, nothing else
out.lampMesh = bodies.get(3).mats[0].emissive.r;

hits = hit(1); out.pickGrabbable = pick(0, 0)?.body ?? null;
hits = hit(2); out.pickNotGrabbable = pick(0, 0)?.body ?? null;
hits = hit(3); out.pickNoSplat = pick(0, 0)?.body ?? null;
photo.want = false;                        // the meshes are drawn: every body is offered again
hits = hit(2); out.pickMeshRoom = pick(0, 0)?.body ?? null;
photo.want = true;

// A new bundle lands while Chair0 is held: its new splat comes in tinted, the others plain.
grab = { body: 1 };
photo.bodies = new Map([['Chair0', splat()], ['Chair1', splat()]]);
attachBodySplats();
out.newBundleHeld = photo.bodies.get('Chair0').recolor.rgb;
out.newBundleOther = photo.bodies.get('Chair1').recolor.rgb;
process.stdout.write(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def run(tmp_path_factory) -> dict:
    functions = "\n".join([
        _line("const photo = "), _line("const SPLAT_GLOW = "), _line("const photoShown = "),
        *(_function(n) for n in ("setHighlight", "glowSplat", "pick", "offered",
                                 "attachBodySplats")),
    ])
    path = tmp_path_factory.mktemp("splat_grab") / "harness.cjs"
    path.write_text(HARNESS.replace("__FUNCTIONS__", functions), "utf-8")
    done = subprocess.run([NODE, str(path)], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_the_picked_bodys_own_splat_is_tinted_and_untinted(run: dict) -> None:
    assert run["grabbedTint"] != [1, 1, 1] and min(run["grabbedTint"]) >= 1.0
    assert run["otherTint"] == [1, 1, 1]
    assert run["meshTint"] > 0                     # the hidden meshes as before
    assert run["releasedTint"] == [1, 1, 1]
    assert run["lampMesh"] > 0


def test_a_body_holding_no_real_object_is_not_offered_in_the_photoreal_room(run: dict) -> None:
    assert run["pickGrabbable"] == 1
    assert run["pickNotGrabbable"] is None
    assert run["pickNoSplat"] == 3                 # no splat of its own: a mesh, grabbable
    assert run["pickMeshRoom"] == 2                # photoreal room off: the meshes decide


def test_a_new_bundle_keeps_the_held_bodys_tint(run: dict) -> None:
    assert run["newBundleHeld"] != [1, 1, 1]
    assert run["newBundleOther"] == [1, 1, 1]


def test_the_server_passes_the_bundles_grabbable_bodies_through(tmp_path) -> None:
    from litereality_agent.room_ops.roomcall.splats import SplatSource

    room, chair = "room-" + "0" * 16 + ".spz", "body-Chair1-" + "1" * 16 + ".spz"
    (tmp_path / "splat.json").write_text(json.dumps({
        "stage": "trained", "room": room, "bodies": {"Chair1": chair},
        "grabbable": {"Chair1": False, "Ghost": True, "Chair2": "no"}}))
    out = SplatSource.fixed(tmp_path).public("atrium", "atrium")
    assert out["grabbable"] == {"Chair1": False}          # only its own bodies, only booleans
    (tmp_path / "splat.json").write_text(json.dumps({"room": room, "bodies": {"Chair1": chair}}))
    assert "grabbable" not in SplatSource.fixed(tmp_path).public("atrium", "atrium")
