"""A small built-in room for the room call, so it can be tried without a scan.

    write_demo_scene(out_dir) -> out_dir/scene.xml

It is deliberately the same KIND of scene LiteReality exports — a static shell, static furniture,
free props, and articulated parts on hinge and slide joints — so everything the call does with a
real room (grab a chair, open a door, pull a drawer, throw a ball) can be exercised on it. It is
plain MJCF with box/sphere/plane primitives and rgba materials: no meshes, no textures, nothing to
copy alongside, so the file alone is the whole room.

Layout (MuJoCo world frame, Z up, metres). The interior is x in [-2.45, 2.45], y in [-1.95, 1.95]:

* floor plane, four 2.6 m walls named ``wall_*`` (the east wall is split around a door opening),
  no ceiling so the room can be seen from above;
* a static table at (-0.6, 0.2) with two free chairs and a mug on it;
* a free ball and a free cardboard box on the floor;
* a static cabinet against the north wall with a hinged door (upper part) and a sliding drawer
  (lower part), both with damping, friction loss and joint limits so they stay where they are left;
* the room door in the east wall, hinged at its south jamb and opening into the room.

Every moving part is placed with clearances (2-5 mm) to whatever it can touch, and free bodies are
authored 1-2 mm above their support, so the scene starts with no interpenetration at all and the
settle pass in `sim.py` has nothing to undo.
"""

from __future__ import annotations

from pathlib import Path

_SCENE = """\
<mujoco model="roomcall_demo">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <size memory="16M"/>

  <visual>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5" specular="0.1 0.1 0.1"/>
    <quality shadowsize="2048"/>
  </visual>

  <asset>
    <material name="floor"     rgba="0.60 0.47 0.34 1" specular="0.15" shininess="0.3"/>
    <material name="wall"      rgba="0.91 0.89 0.85 1" specular="0.05"/>
    <material name="wood"      rgba="0.48 0.32 0.20 1" specular="0.2" shininess="0.4"/>
    <material name="chair"     rgba="0.30 0.42 0.55 1" specular="0.2"/>
    <material name="cabinet"   rgba="0.95 0.95 0.93 1" specular="0.2"/>
    <material name="front"     rgba="0.80 0.86 0.84 1" specular="0.25"/>
    <material name="metal"     rgba="0.62 0.64 0.66 1" specular="0.8" shininess="0.8"/>
    <material name="door"      rgba="0.70 0.55 0.40 1" specular="0.2"/>
    <material name="ball"      rgba="0.86 0.20 0.16 1" specular="0.5" shininess="0.6"/>
    <material name="mug"       rgba="0.20 0.45 0.80 1" specular="0.6" shininess="0.7"/>
    <material name="cardboard" rgba="0.74 0.57 0.37 1" specular="0.05"/>
  </asset>

  <default>
    <!-- Wood on wood / rubber feet on a wooden floor: sliding friction ~0.6-0.8. -->
    <geom friction="0.7 0.005 0.0001" condim="3"/>
    <joint armature="0.01"/>
  </default>

  <worldbody>
    <light name="sun" directional="true" pos="0 0 6" dir="-0.3 -0.2 -1" diffuse="0.6 0.6 0.6"/>
    <light name="fill" pos="0 0 2.4" dir="0 0 -1" diffuse="0.3 0.3 0.3"/>

    <geom name="floor" type="plane" size="2.55 2.05 0.1" material="floor"/>

    <!-- Walls: 10 cm thick, 2.6 m high. The east wall leaves a 0.9 x 2.05 m door opening. -->
    <geom name="wall_n" type="box" pos="0 2.0 1.3" size="2.55 0.05 1.3" material="wall"/>
    <geom name="wall_s" type="box" pos="0 -2.0 1.3" size="2.55 0.05 1.3" material="wall"/>
    <geom name="wall_w" type="box" pos="-2.5 0 1.3" size="0.05 2.05 1.3" material="wall"/>
    <geom name="wall_e_south" type="box" pos="2.5 -0.875 1.3" size="0.05 1.175 1.3"
          material="wall"/>
    <geom name="wall_e_north" type="box" pos="2.5 1.625 1.3" size="0.05 0.425 1.3"
          material="wall"/>
    <geom name="wall_e_lintel" type="box" pos="2.5 0.75 2.325" size="0.05 0.45 0.275"
          material="wall"/>

    <!-- Static table: top surface at z = 0.76. -->
    <body name="table" pos="-0.6 0.2 0">
      <geom name="table_top" type="box" pos="0 0 0.74" size="0.6 0.4 0.02" material="wood"/>
      <geom name="table_leg_1" type="box" pos="0.55 0.35 0.36" size="0.025 0.025 0.36"
            material="wood"/>
      <geom name="table_leg_2" type="box" pos="-0.55 0.35 0.36" size="0.025 0.025 0.36"
            material="wood"/>
      <geom name="table_leg_3" type="box" pos="0.55 -0.35 0.36" size="0.025 0.025 0.36"
            material="wood"/>
      <geom name="table_leg_4" type="box" pos="-0.55 -0.35 0.36" size="0.025 0.025 0.36"
            material="wood"/>
    </body>

    <!-- Two chairs built from boxes, ~6 kg each (density of a light hardwood/plywood mix). -->
    <body name="chair_1" pos="-0.6 -0.6 0.002">
      <freejoint name="chair_1"/>
      <geom name="chair_1_seat" type="box" pos="0 0 0.45" size="0.21 0.21 0.02"
            density="450" material="chair"/>
      <geom name="chair_1_back" type="box" pos="0 -0.19 0.72" size="0.21 0.02 0.25"
            density="450" material="chair"/>
      <geom name="chair_1_leg_1" type="box" pos="0.18 0.18 0.215" size="0.02 0.02 0.215"
            density="450" material="chair"/>
      <geom name="chair_1_leg_2" type="box" pos="-0.18 0.18 0.215" size="0.02 0.02 0.215"
            density="450" material="chair"/>
      <geom name="chair_1_leg_3" type="box" pos="0.18 -0.18 0.215" size="0.02 0.02 0.215"
            density="450" material="chair"/>
      <geom name="chair_1_leg_4" type="box" pos="-0.18 -0.18 0.215" size="0.02 0.02 0.215"
            density="450" material="chair"/>
    </body>
    <body name="chair_2" pos="-0.6 1.0 0.002" euler="0 0 180">
      <freejoint name="chair_2"/>
      <geom name="chair_2_seat" type="box" pos="0 0 0.45" size="0.21 0.21 0.02"
            density="450" material="chair"/>
      <geom name="chair_2_back" type="box" pos="0 -0.19 0.72" size="0.21 0.02 0.25"
            density="450" material="chair"/>
      <geom name="chair_2_leg_1" type="box" pos="0.18 0.18 0.215" size="0.02 0.02 0.215"
            density="450" material="chair"/>
      <geom name="chair_2_leg_2" type="box" pos="-0.18 0.18 0.215" size="0.02 0.02 0.215"
            density="450" material="chair"/>
      <geom name="chair_2_leg_3" type="box" pos="0.18 -0.18 0.215" size="0.02 0.02 0.215"
            density="450" material="chair"/>
      <geom name="chair_2_leg_4" type="box" pos="-0.18 -0.18 0.215" size="0.02 0.02 0.215"
            density="450" material="chair"/>
    </body>

    <!-- A size-5 football: 0.45 kg. condim 6 adds rolling resistance so it does not roll
         forever, which MuJoCo's default condim 3 would let it do. -->
    <body name="ball" pos="1.0 -1.0 0.112">
      <freejoint name="ball"/>
      <geom name="ball" type="sphere" size="0.11" mass="0.45" condim="6"
            friction="0.8 0.01 0.002" material="ball"/>
    </body>

    <!-- A mug-sized box on the table: 0.3 kg, ceramic on wood. -->
    <body name="mug" pos="-0.35 0.3 0.811">
      <freejoint name="mug"/>
      <geom name="mug" type="box" size="0.04 0.04 0.05" mass="0.3" friction="0.6 0.005 0.0001"
            material="mug"/>
    </body>

    <!-- A part-filled cardboard box on the floor: 1.5 kg, cardboard on wood. -->
    <body name="cardboard_box" pos="-1.8 -1.4 0.151">
      <freejoint name="cardboard_box"/>
      <geom name="cardboard_box" type="box" size="0.22 0.16 0.15" mass="1.5"
            friction="0.5 0.005 0.0001" material="cardboard"/>
    </body>

    <!-- Static cabinet against the north wall, front facing -y. Carcass 0.8 x 0.45 x 0.9 m with a
         divider: hinged door above it, sliding drawer below. -->
    <body name="cabinet" pos="0.4 1.72 0">
      <geom name="cabinet_plinth" type="box" pos="0 0 0.04" size="0.4 0.225 0.04"
            material="cabinet"/>
      <geom name="cabinet_top" type="box" pos="0 0 0.88" size="0.4 0.225 0.02"
            material="cabinet"/>
      <geom name="cabinet_left" type="box" pos="-0.39 0 0.47" size="0.01 0.225 0.39"
            material="cabinet"/>
      <geom name="cabinet_right" type="box" pos="0.39 0 0.47" size="0.01 0.225 0.39"
            material="cabinet"/>
      <geom name="cabinet_back" type="box" pos="0 0.215 0.47" size="0.38 0.01 0.39"
            material="cabinet"/>
      <geom name="cabinet_divider" type="box" pos="0 -0.005 0.34" size="0.38 0.21 0.01"
            material="cabinet"/>

      <!-- Door: hinge at the front-left corner, opens outwards (negative angle). -->
      <body name="cabinet_door" pos="-0.38 -0.225 0">
        <joint name="cabinet_door_hinge" type="hinge" axis="0 0 1" range="-100 0"
               damping="0.6" frictionloss="0.15"/>
        <geom name="cabinet_door_panel" type="box" pos="0.38 0.01 0.605" size="0.377 0.01 0.25"
              mass="2.5" material="front"/>
        <geom name="cabinet_door_knob" type="box" pos="0.70 -0.025 0.605"
              size="0.012 0.015 0.05" mass="0.05" material="metal"/>
      </body>

      <!-- Drawer: slides out along -y up to 35 cm. -->
      <body name="cabinet_drawer" pos="0 0 0">
        <joint name="cabinet_drawer_slide" type="slide" axis="0 -1 0" range="0 0.35"
               damping="6" frictionloss="0.8"/>
        <geom name="cabinet_drawer_front" type="box" pos="0 -0.215 0.205" size="0.375 0.01 0.12"
              mass="1.2" material="front"/>
        <geom name="cabinet_drawer_bottom" type="box" pos="0 -0.01 0.10" size="0.36 0.19 0.005"
              mass="0.8" material="cabinet"/>
        <geom name="cabinet_drawer_side_l" type="box" pos="-0.355 -0.01 0.165"
              size="0.005 0.19 0.06" mass="0.3" material="cabinet"/>
        <geom name="cabinet_drawer_side_r" type="box" pos="0.355 -0.01 0.165"
              size="0.005 0.19 0.06" mass="0.3" material="cabinet"/>
        <geom name="cabinet_drawer_rear" type="box" pos="0 0.175 0.165" size="0.35 0.005 0.06"
              mass="0.3" material="cabinet"/>
        <geom name="cabinet_drawer_handle" type="box" pos="0 -0.24 0.24"
              size="0.08 0.015 0.012" mass="0.05" material="metal"/>
      </body>
    </body>

    <!-- Room door in the east wall: hinge on the inner face of the south jamb, opens into the
         room (positive angle). 20 kg, with enough friction loss to stay where it is left. -->
    <body name="room_door" pos="2.45 0.3 0">
      <joint name="room_door_hinge" type="hinge" axis="0 0 1" range="0 100"
             damping="4" frictionloss="1.0"/>
      <geom name="room_door_slab" type="box" pos="0.02 0.45 1.025" size="0.02 0.43 1.01"
            mass="20" material="door"/>
      <geom name="room_door_handle" type="box" pos="-0.03 0.80 1.0" size="0.03 0.05 0.012"
            mass="0.2" material="metal"/>
    </body>
  </worldbody>
</mujoco>
"""


def write_demo_scene(out_dir: Path) -> Path:
    """Write the demo room's scene.xml into `out_dir` (created if needed) and return its path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "scene.xml"
    path.write_text(_SCENE, encoding="utf-8")
    return path
