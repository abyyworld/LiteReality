"""The physics table a room is reported with — what MuJoCo steps, and where each number came from."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "simready" / "physics_table.py"

# A shaker carrying the room (the `room_*` joints the export always adds), a free cabinet with a
# drawer on a slide, a static wall, and a visual-only geom that must not count as friction.
SCENE = """
<mujoco>
  <worldbody>
    <body name="room">
      <joint name="room_x" type="slide" axis="1 0 0"/>
      <geom name="wall" type="box" size="1 0.1 1" mass="100"/>
    </body>
    <body name="Cabinet0" pos="0 0 0.5">
      <freejoint/>
      <geom type="box" size="0.3 0.3 0.4" mass="20" friction="0.55 0.005 0.0001"/>
      <geom type="box" size="0.31 0.31 0.41" mass="0" contype="0" conaffinity="0"
            friction="0.9 0.005 0.0001"/>
      <body name="Cabinet0_drawer" pos="0 0.3 0">
        <joint type="slide" axis="0 1 0" range="0 0.3"/>
        <geom type="box" size="0.25 0.05 0.1" mass="2" friction="0.42 0.005 0.0001"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def table():
    return runpy.run_path(str(SCRIPT))


@pytest.fixture
def scene(tmp_path: Path) -> Path:
    out = tmp_path / "scan" / "mujoco_seed"
    out.mkdir(parents=True)
    (out / "scene.xml").write_text(SCENE)
    return out / "scene.xml"


def test_simulated_lists_only_what_can_move_with_the_mass_its_joint_moves(table, scene):
    rows = {row["body"]: row for row in table["simulated"](scene)}

    assert set(rows) == {"Cabinet0", "Cabinet0_drawer"}       # not the room, not the wall
    assert rows["Cabinet0"]["joint"] == "free"
    assert rows["Cabinet0"]["mass_kg"] == pytest.approx(22.0)  # carcass + drawer
    assert rows["Cabinet0"]["friction"] == "0.55"              # the visual-only geom is ignored
    assert rows["Cabinet0_drawer"] == {"body": "Cabinet0_drawer", "joint": "slide",
                                       "mass_kg": pytest.approx(2.0), "friction": "0.42"}


def test_sources_read_each_sidecar_and_the_room_is_found_beside_a_seed_export(table, scene):
    room = table["default_room"](scene)
    assert room == scene.parent.parent / "scene_init" / "scene_stage" / "room_init" / "room"

    sim = room / "Objects" / "Procedural" / "Cabinet0" / "sim"
    sim.mkdir(parents=True)
    (sim / "Cabinet0.physics.json").write_text(json.dumps({
        "name": "Cabinet0", "category": "cabinet", "total_mass": 22.0,
        "notes": ["total mass 22.00 kg derived from cabinet occupancy density over a 0.400 m^3 box"],
        "links": [
            {"mass_source": "derived", "friction": 0.55, "restitution": 0.2, "material": "oak"},
            {"mass_source": "derived", "friction": 0.42, "restitution": 0.25, "material": "steel"},
        ],
    }))
    (rows,) = table["sources"](room)

    assert rows["mass_kg"] == 22.0 and rows["mass_source"] == "derived"
    assert rows["friction"] == "0.42, 0.55" and rows["restitution"] == "0.2, 0.25"
    assert rows["note"].startswith("total mass 22.00 kg derived from cabinet")


def test_cli_prints_both_tables_and_writes_the_csv(table, scene, tmp_path, capsys):
    out = tmp_path / "physics.csv"
    assert table["main"]([str(scene), "--csv", str(out)]) == 0

    printed = capsys.readouterr().out
    assert "| Cabinet0 | free | 22.0 | 0.55 |" in printed
    assert "no Room package found" in printed                  # no sidecars were written here
    assert out.read_text().splitlines()[0] == "body,joint,mass_kg,friction"
