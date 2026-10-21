"""The one shared physics simulation behind a room call.

Everyone in a room sees the same MuJoCo state: the server owns a `RoomSim`, peers send grab /
move / release, and the server steps it in real time and broadcasts `snapshot()`. Nothing here
knows about sockets or threads beyond its own lock; it is a scene plus the rules for letting many
hands pull on it at once without anyone being able to break it.

Three things shape the design:

* **A grab is a spring, not a teleport.** Writing a held object's pose directly would let it pass
  through walls and other objects and would mean nothing for a door on a hinge or a drawer on a
  slide. Instead the grabbed point (stored in the body's own frame, so it travels with the body)
  is pulled towards the hand's target by a critically damped force applied at that point with
  `mj_applyFT`. That one rule drags a chair, swings a door about its hinge and slides a drawer
  along its rail, because MuJoCo maps the Cartesian force through the body's Jacobian onto
  whatever joints it actually has. A free body also gets its weight cancelled, so it floats while
  held instead of sagging below the hand, and mild angular damping so it does not spin forever.
* **Nobody can launch anything to infinity.** The force is clamped to a multiple of the body's
  weight and the target to the scene's bounds, so the worst a peer can do is throw something hard
  at a wall.
* **NaN never leaves this module.** A scene that goes unstable (non-finite state, MuJoCo's own
  bad-qpos/qvel/qacc warnings or its own silent reset, or a body moving faster than anything in a
  room can) is put back to its settled starting state, rather than broadcasting garbage to every
  peer.
* **Losing one object is not losing the room.** An exported room's floor is a finite box and its
  walls have doorways, so a chair can be carried out and dropped into nothing. Left alone it falls
  until its speed reads as a blow-up and the whole room resets around it; instead that one body is
  put back where it settled, as soon as it is clearly gone.

The model may be shared between rooms (the server compiles it once), so nothing here ever writes
to it: the zero-gravity relax pass cancels gravity with per-body forces on this room's own MjData
instead of switching `opt.gravity` off on the shared model.

A room can also move to a NEW VERSION of its scene while people are in it (a better scan swapped in
live): `RoomSim(new_model, settled=...)` starts from that version's settled state without settling
again, and `adopt(old)` carries what people did in the old version across by body name — where
they moved things, which doors they opened, what they are holding. See `adopt`.
"""

from __future__ import annotations

import math
import operator
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from litereality_agent.room_ops.roomcall.model_info import (
    movable_body_ids,
    rig_joints,
    servo_held_joints,
)

# Grab spring, in units of acceleration per metre of error (1/s^2): kp = 400 pulls a 10 cm error
# at 40 m/s^2 and settles in ~0.2 s. kd = 2*sqrt(kp) makes it critically damped for the subtree
# mass; a hinge's effective mass at the handle is smaller than the subtree mass, which only makes
# it over-damped, never oscillatory.
GRAB_KP = 400.0
# No grab can push harder than this many times the held body's weight (25 g ≈ 245 m/s^2: enough
# to throw a ball across the room, not enough to tunnel it through a wall in one step).
MAX_ACCEL_G = 25.0
# Angular damping on held free bodies (1/s): a held chair turns when pulled off-centre, but stops
# turning when the hand stops. It acts through the body's inertia tensor about the grabbed point,
# so it is the same rate for a ball held at its top and a pencil held at its tip.
GRAB_ANGULAR_DAMPING = 10.0
# Anything faster than this (m/s or rad/s) in a room is a simulation that has blown up.
MAX_SPEED = 1000.0
# How far outside the scene's drawn extent a grab target may be (metres).
TARGET_MARGIN = 1.0
# How far beyond the grab-target box (plus its own reach, so nothing held is ever caught) a free
# body's origin may go sideways or downwards before it counts as having left the room (metres).
ESCAPE_MARGIN = 1.0
# Nothing a hand can carry weighs more (kg). A backstop for scenes whose building-sized bodies are
# not recognised as static: pulling at one would shift every wall.
MAX_GRAB_MASS = 1.0e4
# After a version swap, contacts deeper than this (m) are relaxed apart before anyone sees them: a
# chair carried to where the new version's (bigger) table now stands would otherwise be fired out.
ADOPT_PENETRATION = 0.001
# How long that relax runs (simulated seconds). The experiments separated a 30 %-grown chair pushed
# into a table in 0.3 s; it is a tenth of what the initial settle gets, so a swap stays a blink.
ADOPT_RELAX_SECONDS = 0.3

_FREE = int(mujoco.mjtJoint.mjJNT_FREE)
_BALL = int(mujoco.mjtJoint.mjJNT_BALL)
_STATE = mujoco.mjtState.mjSTATE_INTEGRATION

_WARNINGS = (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
             mujoco.mjtWarning.mjWARN_BADQACC)


def load_model(scene_xml: Path) -> mujoco.MjModel:
    """Compile a scene file. `from_xml_path` resolves meshes and textures relative to the file."""
    return mujoco.MjModel.from_xml_path(str(Path(scene_xml)))


@dataclass
class _Grab:
    body: int
    local: np.ndarray       # anchor in the body's frame
    target: np.ndarray      # world-frame target for the anchor


def _vec3(value: Sequence[float]) -> np.ndarray | None:
    """A finite 3-vector, or None for anything else (wrong length, strings, NaN, inf)."""
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        return None
    return arr


class RoomSim:
    """One room's MuJoCo state plus everyone's grabs. Every public method takes `self._lock`, so
    the server may call `grab`/`move`/`release` from its event loop while `step_realtime` runs in
    a worker thread."""

    def __init__(self, model: mujoco.MjModel, *, settle_seconds: float = 1.0,
                 settled: np.ndarray | None = None) -> None:
        """`settled`, from another `RoomSim(model).settled_state()` on the same model, skips the
        settle: settling is deterministic, so every room of one scene version can start from the
        state one settle produced instead of each spending a second of CPU on it. A state that
        does not fit this model is ignored and the scene is settled as usual."""
        self.model = model
        self.data = mujoco.MjData(model)
        self.movable: list[int] = sorted(movable_body_ids(model))
        self._lock = threading.Lock()
        self._grabs: dict[str, _Grab] = {}
        self._debt = 0.0                     # wall time owed to the simulation, seconds
        self.unstable_resets = 0             # how often instability forced a reset
        self.respawns = 0                    # how often a body that left the room was put back
        self.timestep = float(model.opt.timestep)

        # step1/step2 lets grab forces see the kinematics of the state they are applied to.
        # MuJoCo's split step integrates RK4 as Euler, so an RK4 scene uses a plain mj_step with
        # forces from the previous step's kinematics instead (2 ms stale, harmless).
        self._split = int(model.opt.integrator) != int(mujoco.mjtIntegrator.mjINT_RK4)
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        self._zero3 = np.zeros(3)

        gravity_on = not (int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_GRAVITY))
        self._gravity = np.array(model.opt.gravity, dtype=float) if gravity_on else np.zeros(3)
        g0 = float(np.linalg.norm(self._gravity)) or 9.81
        self._max_accel = MAX_ACCEL_G * g0
        dt = self.timestep if self.timestep > 0 else 0.002
        # Keep the explicit spring well inside its stability limit for coarse timesteps.
        self._kp = min(GRAB_KP, (0.2 / dt) ** 2)
        self._kd = 2.0 * math.sqrt(self._kp)

        self._driver = {b: self._driving_body(b) for b in self.movable}
        free = int(mujoco.mjtJoint.mjJNT_FREE)
        self._free_drivers = {
            d for d in set(self._driver.values())
            if int(model.body_jntnum[d]) > 0
            and int(model.jnt_type[int(model.body_jntadr[d])]) == free
        }
        self._reach = {b: self._body_reach(b) for b in self.movable}
        self._grabbable = {b for b in self.movable
                           if float(model.body_subtreemass[self._driver[b]]) <= MAX_GRAB_MASS}

        if settled is None or not self._seed(settled):
            self._settle(settle_seconds)
        self._bounds = self._scene_bounds()
        self._initial = self._get_state()
        self._settled_qpos = self.data.qpos.copy()
        self._settled_mocap = (self.data.mocap_pos.copy(), self.data.mocap_quat.copy())
        # Per free body, the box its origin must stay in (see `_respawn_escaped`).
        self._free_ids = np.array(sorted(self._free_drivers), dtype=int)
        margin = np.array([ESCAPE_MARGIN + self._reach[b] for b in self._free_ids]).reshape(-1, 1)
        self._escape_lo = self._bounds[0][None, :] - margin
        self._escape_hi = self._bounds[1][None, :] + margin

    # ---- public API ---------------------------------------------------------------------------

    def grab(self, owner: str, body: int, point_world: Sequence[float]) -> bool:
        """Start (or replace) `owner`'s grab of `body` at the world point they clicked. Returns
        False, holding nothing new, for a bad/static body id, a body too heavy for a hand
        (`MAX_GRAB_MASS`, counting whatever a grab of it really moves) or a non-finite point."""
        if isinstance(body, bool):
            return False
        try:
            b = operator.index(body)
        except TypeError:
            return False
        point = _vec3(point_world)
        with self._lock:
            if b not in self._grabbable or point is None:
                return False
            d = self.data
            rot = d.xmat[b].reshape(3, 3)
            local = rot.T @ (point - d.xpos[b])
            # A point far off the body would give a huge lever arm; keep it on the body's extent.
            reach = self._reach[b]
            norm = float(np.linalg.norm(local))
            if norm > reach:
                local *= reach / norm
            anchor = d.xpos[b] + rot @ local
            self._grabs[str(owner)] = _Grab(b, local, anchor.copy())
            return True

    def move(self, owner: str, target_world: Sequence[float]) -> None:
        """Move `owner`'s hand. Ignored if they hold nothing or the target is not finite."""
        target = _vec3(target_world)
        if target is None:
            return
        with self._lock:
            grab = self._grabs.get(str(owner))
            if grab is not None:
                grab.target = np.clip(target, self._bounds[0], self._bounds[1])

    def release(self, owner: str) -> None:
        """Let go. The body keeps its velocity, which is what makes a throw a throw."""
        with self._lock:
            self._grabs.pop(str(owner), None)

    def step_realtime(self, wall_dt: float, *, max_steps: int = 400) -> int:
        """Advance the simulation by `wall_dt` seconds of wall time, applying every grab before
        each `mj_step`. Fractions of a timestep carry over to the next call so the simulation
        keeps real time at any call rate. A long stall (a slow machine, a paused debugger) is
        clamped to `max_steps` and the rest forgotten, rather than caught up in a burst.
        Returns the number of steps taken."""
        try:
            wall_dt = float(wall_dt)
        except (TypeError, ValueError):
            return 0
        if not math.isfinite(wall_dt) or wall_dt <= 0 or self.timestep <= 0 or max_steps <= 0:
            return 0
        with self._lock:
            dt = self.timestep
            self._debt += min(wall_dt, max_steps * dt)
            steps = min(int(self._debt / dt + 1e-9), max_steps)
            self._debt -= steps * dt
            if steps >= max_steps:
                self._debt = 0.0
            m, d = self.model, self.data
            for _ in range(steps):
                t0 = d.time
                try:
                    if self._split:
                        mujoco.mj_step1(m, d)
                        self._apply_grabs()
                        mujoco.mj_step2(m, d)
                    else:
                        self._apply_grabs()
                        mujoco.mj_step(m, d)
                except mujoco.FatalError:
                    self._recover()
                    break
                # On a bad qacc MuJoCo resets the data itself (mj_resetData: authored pose, time 0)
                # and carries on. Time running backwards is that reset seen directly; the warning
                # counter is what still shows it when the step started at time 0.
                if d.time <= t0 or not self._healthy():
                    self._recover()
                    break
                if self._free_ids.size:
                    self._respawn_escaped()
            # mj_step leaves xpos/xquat at the state BEFORE its integration; bring them up to
            # date so snapshot() and the next grab() see where the bodies really are.
            mujoco.mj_kinematics(m, d)
            return steps

    def snapshot(self) -> dict:
        """The state to broadcast: world poses of movable bodies and every active grab, rounded
        to 5 decimals (10 µm) so a frame stays small. Plain Python types, JSON-serialisable."""
        with self._lock:
            d = self.data
            pos = np.round(d.xpos, 5)
            quat = np.round(d.xquat, 5)
            bodies = [[b, *pos[b].tolist(), *quat[b].tolist()] for b in self.movable]
            grabs = []
            for owner, g in self._grabs.items():
                anchor = d.xpos[g.body] + d.xmat[g.body].reshape(3, 3) @ g.local
                grabs.append([owner, g.body, *np.round(anchor, 5).tolist(),
                              *np.round(g.target, 5).tolist()])
            return {"time": round(float(d.time), 5), "bodies": bodies, "grabs": grabs}

    def reset(self) -> None:
        """Put the room back exactly as it was after settling, and drop every grab."""
        with self._lock:
            self._restore()

    def settled_state(self) -> np.ndarray:
        """The settled starting state (what `reset` returns to), for seeding other rooms of the
        same model with `RoomSim(model, settled=...)`."""
        return self._initial.copy()

    # ---- moving to a new version of the scene -----------------------------------------------

    def adopt(self, old: RoomSim) -> dict:
        """Carry `old`'s live room into this one, a new version of the same scene, BY NAME.

        Body ids are an accident of each version's XML (insert one body and every later id shifts),
        but the exporter names bodies after the scan's object handles — Chair0, Table1 — in every
        stage, so a name is what stays the same object across versions:

        - a free body carries its DISPLACEMENT since settling, P_new = (P_old · S_old⁻¹) · S_new
          with S each version's own settled pose. Copying the old pose would be wrong whenever the
          body's frame origin moved between versions (a TRELLIS mesh centred differently from the
          RoomPlan box it replaces) — the experiment measured 50 mm off for a 50 mm origin shift,
          0.000 mm with the displacement — and an untouched body lands exactly on the new
          version's own settled pose;
        - a hinge, slide or ball joint carries its value, clamped to the new range;
        - servo-held and shake-rig joints are the building, and stay as the new version has them;
        - a body new in this version (or whose joints no longer match) starts where it settled;
        - a grab is kept if its body exists and is grabbable here: the same world anchor, re-
          expressed in the new body's frame and clamped to its reach, and the same target clipped
          to the new bounds. Anyone holding something that became static or vanished lets go.

        Everything starts at rest. If the carried poses interpenetrate (a carried chair where the
        new, bigger table now stands) the room is relaxed apart for `ADOPT_RELAX_SECONDS` with the
        grabs suspended, as the initial settle does. A carry that produces anything non-finite
        falls back to this version's settled state.

        This version's settled state stays its own: `reset` and respawns go to the NEW version's
        room, not to a carried copy of the old one. Returns what happened, by name:
        {"carried", "reset", "gone", "grabs_kept", "grabs_dropped", "relaxed"}.
        """
        if old is self:
            return {"carried": [], "reset": [], "gone": [], "grabs_kept": [],
                    "grabs_dropped": [], "relaxed": False}
        with old._lock, self._lock:
            carried, reset, gone = self._carry_bodies(old)
            kept, dropped = self._carry_grabs(old)
            relaxed = False
            if self._finite():
                relaxed = self._separate()
            if not self._finite():
                # Should never happen with finite inputs; never let it reach a snapshot.
                dropped = sorted({*dropped, *kept})
                kept = []
                reset = sorted({*reset, *carried})
                carried = []
                self._restore()
            return {"carried": carried, "reset": reset, "gone": gone, "grabs_kept": kept,
                    "grabs_dropped": dropped, "relaxed": relaxed}

    def _carry_bodies(self, old: RoomSim) -> tuple[list[str], list[str], list[str]]:
        """Write `old`'s joint state into this room's MjData by body name (see `adopt`). Caller
        holds both locks."""
        om, od, nm, nd = old.model, old.data, self.model, self.data
        old_ids = _named_bodies(om)
        old_held = servo_held_joints(om) | rig_joints(om)
        new_held = servo_held_joints(nm) | rig_joints(nm)
        old_moving = set(old.movable)
        carried: list[str] = []
        reset: list[str] = []
        matched: set[int] = set()
        for b in self.movable:
            joints = _own_joints(nm, b, new_held)
            mocap = int(nm.body_mocapid[b])
            if not joints and mocap < 0:
                continue                     # welded under a moving body: follows its parent
            name = mujoco.mj_id2name(nm, mujoco.mjtObj.mjOBJ_BODY, b)
            ob = old_ids.get(name) if name else None
            if ob is None or ob not in old_moving:
                reset.append(name or f"body{b}")
                continue
            matched.add(ob)
            moved = False
            if mocap >= 0 and int(om.body_mocapid[ob]) >= 0:
                moved = self._carry_mocap(old, int(om.body_mocapid[ob]), mocap)
            pairs = _match_joints(om, _own_joints(om, ob, old_held), nm, joints)
            for oj, nj in pairs:
                self._carry_joint(old, oj, nj)
            if moved or pairs:
                carried.append(name)
            else:
                reset.append(name)
        gone = sorted(
            mujoco.mj_id2name(om, mujoco.mjtObj.mjOBJ_BODY, ob) or f"body{ob}"
            for ob in old.movable
            if ob not in matched and (_own_joints(om, ob, old_held) or int(om.body_mocapid[ob]) >= 0))
        m, d = nm, nd
        d.qvel[:] = 0.0
        d.qacc_warmstart[:] = 0.0
        d.qfrc_applied[:] = 0.0
        d.xfrc_applied[:] = 0.0
        if m.na:
            d.act[:] = 0.0
        d.time = float(od.time)          # one room, one clock: time does not jump back at a swap
        mujoco.mj_forward(m, d)
        self._debt = 0.0
        return sorted(carried), sorted(reset), gone

    def _carry_joint(self, old: RoomSim, oj: int, nj: int) -> None:
        om, nm = old.model, self.model
        oa, na = int(om.jnt_qposadr[oj]), int(nm.jnt_qposadr[nj])
        kind = int(nm.jnt_type[nj])
        q_old = old.data.qpos
        q_new = self.data.qpos
        if kind == _FREE:
            moved = _compose(*_compose(q_old[oa:oa + 3], q_old[oa + 3:oa + 7],
                                       *_inverse(old._settled_qpos[oa:oa + 3],
                                                 old._settled_qpos[oa + 3:oa + 7])),
                             self._settled_qpos[na:na + 3], self._settled_qpos[na + 3:na + 7])
            if np.all(np.isfinite(moved[0])) and np.all(np.isfinite(moved[1])):
                q_new[na:na + 3] = moved[0]
                q_new[na + 3:na + 7] = _unit(moved[1])
        elif kind == _BALL:
            quat = np.asarray(q_old[oa:oa + 4], dtype=float)
            if np.all(np.isfinite(quat)) and np.linalg.norm(quat) > 0:
                q_new[na:na + 4] = _unit(quat)
        else:
            value = float(q_old[oa])
            if math.isfinite(value):
                if int(nm.jnt_limited[nj]):
                    lo, hi = (float(v) for v in nm.jnt_range[nj])
                    value = min(max(value, lo), hi)
                q_new[na] = value

    def _carry_mocap(self, old: RoomSim, omid: int, nmid: int) -> bool:
        """A mocap body has no joint; its pose carries by the same displacement rule."""
        pos, quat = _compose(*_compose(old.data.mocap_pos[omid], old.data.mocap_quat[omid],
                                       *_inverse(old._settled_mocap[0][omid],
                                                 old._settled_mocap[1][omid])),
                             self._settled_mocap[0][nmid], self._settled_mocap[1][nmid])
        if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(quat))):
            return False
        self.data.mocap_pos[nmid] = pos
        self.data.mocap_quat[nmid] = _unit(quat)
        return True

    def _carry_grabs(self, old: RoomSim) -> tuple[list[str], list[str]]:
        """Re-attach every grab whose body is still grabbable, by name (see `adopt`). Runs after
        `_carry_bodies`, so the new bodies are already where the old ones were."""
        om, od, nd = old.model, old.data, self.data
        new_ids = _named_bodies(self.model)
        kept, dropped = [], []
        self._grabs.clear()
        for owner, grab in old._grabs.items():
            name = mujoco.mj_id2name(om, mujoco.mjtObj.mjOBJ_BODY, grab.body)
            b = new_ids.get(name) if name else None
            if b is None or b not in self._grabbable:
                dropped.append(owner)
                continue
            world = od.xpos[grab.body] + od.xmat[grab.body].reshape(3, 3) @ grab.local
            local = nd.xmat[b].reshape(3, 3).T @ (world - nd.xpos[b])
            reach = self._reach[b]
            norm = float(np.linalg.norm(local))
            if not math.isfinite(norm):
                dropped.append(owner)
                continue
            if norm > reach:
                local *= reach / norm
            target = np.clip(grab.target, self._bounds[0], self._bounds[1])
            self._grabs[owner] = _Grab(b, local, target)
            kept.append(owner)
        return sorted(kept), sorted(dropped)

    def _separate(self) -> bool:
        """Relax carried interpenetrations apart (grabs suspended: `_relax` never applies them),
        then bring everything to rest. True if anything was deep enough to need it."""
        m, d = self.model, self.data
        if not any(d.contact[i].dist < -ADOPT_PENETRATION for i in range(d.ncon)):
            return False
        clock = float(d.time)
        before = d.qpos.copy()
        try:
            ok = self._relax(ADOPT_RELAX_SECONDS)
        except Exception:  # noqa: BLE001 — a failed relax leaves the carried poses as they were
            ok = False
        if not ok or not self._finite():
            d.qpos[:] = before
        d.qvel[:] = 0.0
        d.qacc_warmstart[:] = 0.0
        d.qfrc_applied[:] = 0.0
        d.xfrc_applied[:] = 0.0
        d.time = clock
        mujoco.mj_forward(m, d)
        self._clear_warnings()
        return True

    # ---- grab physics -------------------------------------------------------------------------

    def _apply_grabs(self) -> None:
        """Write every grab's force into a cleared `qfrc_applied` (called before each step)."""
        m, d = self.model, self.data
        d.qfrc_applied[:] = 0.0
        if not self._grabs:
            return
        for grab in self._grabs.values():
            b = grab.body
            driver = self._driver[b]
            if int(m.body_mocapid[driver]) >= 0:
                if driver == b:        # mocap bodies ignore forces; place them directly
                    mid = int(m.body_mocapid[b])
                    rot = d.xmat[b].reshape(3, 3)
                    d.mocap_pos[mid] = grab.target - rot @ grab.local
                continue
            anchor = d.xpos[b] + d.xmat[b].reshape(3, 3) @ grab.local
            mujoco.mj_jac(m, d, self._jacp, self._jacr, anchor, b)
            vel = self._jacp @ d.qvel
            mass = float(m.body_subtreemass[driver])
            if mass <= 0:
                continue
            force = mass * (self._kp * (grab.target - anchor) - self._kd * vel)
            limit = self._max_accel * mass
            size = float(np.linalg.norm(force))
            if size > limit:
                force *= limit / size
            mujoco.mj_applyFT(m, d, force, self._zero3, anchor, b, d.qfrc_applied)
            if driver in self._free_drivers:
                # Cancel its weight at the centre of mass (no torque), and damp its spin.
                com = d.subtree_com[driver].copy()
                lift = -mass * self._gravity
                omega = self._jacr @ d.qvel
                torque = -GRAB_ANGULAR_DAMPING * (self._spin_inertia(driver, anchor - com) @ omega)
                mujoco.mj_applyFT(m, d, lift, torque, com, driver, d.qfrc_applied)

    def _driving_body(self, b: int) -> int:
        """The body whose joint a grab of `b` really moves: `b` itself if it has a joint (or is
        mocap), else its nearest jointed ancestor (a chair seat welded under a free chair)."""
        m = self.model
        while b > 0 and int(m.body_jntnum[b]) == 0 and int(m.body_mocapid[b]) < 0:
            b = int(m.body_parentid[b])
        return b

    def _spin_inertia(self, b: int, offset: np.ndarray) -> np.ndarray:
        """World-frame inertia tensor of a held body about the grabbed point: its principal
        inertia (scaled up to the mass of anything attached below it) plus the parallel-axis term
        for the anchor's offset from the centre of mass. Per-axis, not a scalar, because a scalar
        big enough for a pencil's long axes is unstable on its thin one."""
        m, d = self.model, self.data
        own = float(m.body_mass[b])
        mass = float(m.body_subtreemass[b])
        scale = mass / own if own > 0 else 1.0
        rot = d.ximat[b].reshape(3, 3)
        inertia = rot @ np.diag(m.body_inertia[b] * scale) @ rot.T
        return inertia + mass * (float(offset @ offset) * np.eye(3) - np.outer(offset, offset))

    def _body_reach(self, b: int) -> float:
        """How far from its frame origin a body's own geometry extends (for clamping anchors)."""
        m = self.model
        start, count = int(m.body_geomadr[b]), int(m.body_geomnum[b])
        if start < 0 or count == 0:
            return 0.5
        reach = max(float(np.linalg.norm(m.geom_pos[g])) + float(m.geom_rbound[g])
                    for g in range(start, start + count))
        return max(reach, 0.05) + 0.05

    def _scene_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """The box grab targets are clipped to: everything drawn, plus a margin. Planes are
        infinite to MuJoCo (rbound 0), so they do not count."""
        m, d = self.model, self.data
        plane = int(mujoco.mjtGeom.mjGEOM_PLANE)
        keep = [g for g in range(m.ngeom)
                if int(m.geom_type[g]) != plane and float(m.geom_rbound[g]) > 0]
        if not keep:
            return np.full(3, -1e3), np.full(3, 1e3)
        pos = d.geom_xpos[keep]
        rad = m.geom_rbound[keep][:, None]
        lo = (pos - rad).min(axis=0) - TARGET_MARGIN
        hi = (pos + rad).max(axis=0) + TARGET_MARGIN
        if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
            return np.full(3, -1e3), np.full(3, 1e3)
        return lo, hi

    # ---- settling, state and recovery ---------------------------------------------------------

    def _settle(self, seconds: float) -> None:
        """Bring the authored scene to rest before anyone sees it, the way `mujoco_shake` does:
        lift free bodies out of walls/fixtures they were authored inside, relax remaining
        overlaps with gravity cancelled and velocity bled off (so they separate instead of
        firing apart), then let gravity act for `seconds`. Any failure falls back to the
        authored pose at rest: a bad scene must still open."""
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        mujoco.mj_forward(m, d)
        authored = d.qpos.copy()
        if not self._finite():
            # Even the authored pose is unusable; qpos0 at rest is all that is left.
            mujoco.mj_resetData(m, d)
            self._finish_settle()
            return
        try:
            seconds = float(seconds)
            if math.isfinite(seconds) and seconds > 0 and self.timestep > 0:
                self._lift_free_bodies()
                start = d.qpos.copy()
                warn0 = self._warning_counts()
                ok = self._relax(seconds) and self._fall(seconds)
                if not ok or not self._finite() or self._warning_counts() != warn0:
                    d.qpos[:] = start if np.all(np.isfinite(start)) else authored
        except Exception:  # never crash on a bad scene
            d.qpos[:] = authored
        self._finish_settle()

    def _finish_settle(self) -> None:
        m, d = self.model, self.data
        d.qvel[:] = 0.0
        d.qacc_warmstart[:] = 0.0
        d.qfrc_applied[:] = 0.0
        d.xfrc_applied[:] = 0.0
        if m.na:
            d.act[:] = 0.0
        d.time = 0.0
        mujoco.mj_forward(m, d)
        self._clear_warnings()   # a failed settle's warnings must not read as a live blow-up

    def _lift_free_bodies(self) -> None:
        """The exporter's geometric fix for free bodies straddling walls or standing inside
        fixtures. Optional: a hand-written scene without wall boxes is simply left alone."""
        try:
            from litereality_agent.room_ops.export.mujoco_shake import _lift_out_of_walls
        except ImportError:  # pragma: no cover - the exporter is part of this package
            return
        before = self.data.qpos.copy()
        try:
            _lift_out_of_walls(self.model, self.data)
        except Exception:
            self.data.qpos[:] = before
        if not self._finite():
            self.data.qpos[:] = before
        mujoco.mj_forward(self.model, self.data)

    def _relax(self, seconds: float) -> bool:
        """Zero-gravity, heavily damped stepping, only if something starts interpenetrating."""
        m, d = self.model, self.data
        mujoco.mj_forward(m, d)
        if not any(d.contact[i].dist < -0.001 for i in range(d.ncon)):
            return True
        weight = (m.body_mass * (1.0 - m.body_gravcomp))[:, None] * self._gravity[None, :]
        steps = max(1, int(min(1.2, seconds) / self.timestep))
        d.xfrc_applied[:] = 0.0
        d.xfrc_applied[:, :3] = -weight
        try:
            for i in range(steps):
                mujoco.mj_step(m, d)
                d.qvel[:] *= 0.55
                if i % 50 == 0 and not self._finite():
                    return False
        finally:
            d.xfrc_applied[:] = 0.0
        d.qvel[:] = 0.0
        return self._finite()

    def _fall(self, seconds: float) -> bool:
        """Normal stepping under gravity so supported things come to rest on their supports."""
        m, d = self.model, self.data
        d.qvel[:] = 0.0
        steps = int(seconds / self.timestep)
        chunk = 50
        for done in range(0, steps, chunk):
            mujoco.mj_step(m, d, nstep=min(chunk, steps - done))
            if not self._finite():
                return False
        return True

    def _seed(self, settled) -> bool:
        """Start from a settled state another RoomSim of this model produced. False (nothing
        changed) if it does not fit this model or is not finite."""
        try:
            state = np.array(settled, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return False
        if state.size != mujoco.mj_stateSize(self.model, _STATE) or not np.all(np.isfinite(state)):
            return False
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_setState(self.model, self.data, state, _STATE)
        self._finish_settle()
        return self._finite()

    def _get_state(self) -> np.ndarray:
        state = np.empty(mujoco.mj_stateSize(self.model, _STATE))
        mujoco.mj_getState(self.model, self.data, state, _STATE)
        return state

    def _restore(self) -> None:
        """Back to the settled state; drops grabs. Caller holds the lock."""
        m, d = self.model, self.data
        mujoco.mj_setState(m, d, self._initial, _STATE)
        d.qfrc_applied[:] = 0.0
        d.xfrc_applied[:] = 0.0
        mujoco.mj_forward(m, d)
        self._grabs.clear()
        self._debt = 0.0
        self._clear_warnings()   # mj_setState leaves them; see `_healthy`

    def _recover(self) -> None:
        self.unstable_resets += 1
        self._restore()

    def _respawn_escaped(self) -> None:
        """Put back, alone, any free body whose origin has left the room: sideways past the
        grab-target box, or below it. Above does not count, because what goes up comes back down
        into the room. Its free joint returns to where it settled, at rest, and whoever held it
        lets go; nothing else in the room is touched. Caller holds the lock."""
        pos = self.data.xpos[self._free_ids]
        out = ((pos[:, :2] < self._escape_lo[:, :2]) | (pos[:, :2] > self._escape_hi[:, :2])).any(1)
        out |= pos[:, 2] < self._escape_lo[:, 2]
        if not out.any():
            return
        m, d = self.model, self.data
        for b in self._free_ids[out].tolist():
            joint = int(m.body_jntadr[b])
            qpos, dof = int(m.jnt_qposadr[joint]), int(m.jnt_dofadr[joint])
            d.qpos[qpos:qpos + 7] = self._settled_qpos[qpos:qpos + 7]
            d.qvel[dof:dof + 6] = 0.0
            d.qacc_warmstart[dof:dof + 6] = 0.0
            for owner in [o for o, g in self._grabs.items() if self._driver[g.body] == b]:
                del self._grabs[owner]
            self.respawns += 1

    def _finite(self) -> bool:
        d = self.data
        return bool(np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel)))

    def _healthy(self) -> bool:
        """Finite, no bad-state warning since the last settle/restore (which zero the counters,
        so a second MuJoCo-internal reset — which zeroes and then increments them — still shows),
        and nothing moving faster than a room allows."""
        d = self.data
        if not self._finite() or any(self._warning_counts()):
            return False
        return not (d.qvel.size and float(np.max(np.abs(d.qvel))) > MAX_SPEED)

    def _warning_counts(self) -> tuple[int, ...]:
        return tuple(int(self.data.warning[int(w)].number) for w in _WARNINGS)

    def _clear_warnings(self) -> None:
        self.data.warning.number[:] = 0


# ---- helpers for carrying state between versions -----------------------------------------------

def _named_bodies(model: mujoco.MjModel) -> dict[str, int]:
    names = {}
    for b in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
        if name:
            names[name] = b
    return names


def _own_joints(model: mujoco.MjModel, b: int, held: set[int]) -> list[int]:
    start, count = int(model.body_jntadr[b]), int(model.body_jntnum[b])
    return [j for j in range(start, start + count) if j not in held] if count > 0 else []


def _match_joints(om: mujoco.MjModel, old: list[int], nm: mujoco.MjModel,
                  new: list[int]) -> list[tuple[int, int]]:
    """Pair one body's joints across versions: by joint name where the new one is named and the
    old body has that name, else by position (an unnamed `<freejoint/>` is the usual case), and
    only joints of the same type."""
    old_names = {mujoco.mj_id2name(om, mujoco.mjtObj.mjOBJ_JOINT, j): j for j in old}
    old_names.pop(None, None)
    pairs, used = [], set()
    for k, nj in enumerate(new):
        name = mujoco.mj_id2name(nm, mujoco.mjtObj.mjOBJ_JOINT, nj)
        oj = old_names.get(name) if name else None
        if oj is None and k < len(old):
            oj = old[k]
        if oj is None or oj in used or int(om.jnt_type[oj]) != int(nm.jnt_type[nj]):
            continue
        used.add(oj)
        pairs.append((oj, nj))
    return pairs


def _compose(pa, qa, pb, qb) -> tuple[np.ndarray, np.ndarray]:
    """Pose product (pa, qa) · (pb, qb), MuJoCo quaternions (w, x, y, z)."""
    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, np.asarray(qa, dtype=float), np.asarray(qb, dtype=float))
    rotated = np.zeros(3)
    mujoco.mju_rotVecQuat(rotated, np.asarray(pb, dtype=float), np.asarray(qa, dtype=float))
    return np.asarray(pa, dtype=float) + rotated, quat


def _inverse(p, q) -> tuple[np.ndarray, np.ndarray]:
    inv = np.zeros(4)
    mujoco.mju_negQuat(inv, _unit(np.asarray(q, dtype=float)))
    moved = np.zeros(3)
    mujoco.mju_rotVecQuat(moved, -np.asarray(p, dtype=float), inv)
    return moved, inv


def _unit(q: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(q))
    return np.asarray(q, dtype=float) / norm if norm > 0 and math.isfinite(norm) \
        else np.array([1.0, 0.0, 0.0, 0.0])
