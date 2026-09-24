# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``FIATLUX-TestLightbulbMechanism-Teleop-v0`` -- teleop bench for the bayonet bulb mechanism.

Purpose: drive the Replace task's **bulb attach/lock mechanism** (``mdp.bulb_attachment``,
issue #77) by hand and record what it does, so its behaviour can be debugged from bags instead
of live repro. Same treatment as :mod:`carry_teleop_env_cfg` (walking: free base + SONIC legs,
pelvis-follow XR camera), on :class:`ReplaceEnvCfg`'s room scene, with two debug-bench choices:

* ``disable_randomization()`` -- deterministic canonical spawns, so a session is repeatable and
  two bags differ by operator input, not by reset noise;
* the INSERT task's tabletop bench replaces the ceiling mount (``apply_tabletop_preset``):
  table + socket fixture on it + fresh bulb at hand height, robot standing at the bench --
  the exact bulb/socket placement proven in the Insert teleop sessions. (The preset drops
  the ladder; sessions test the MECHANISM, not ladder logistics.)

Hands stay the Replace scene's native **Inspire** (trigger = grasp). Record with
``--record bag``: the teleop recorder mirrors ``recording.py``'s bayonet lock telemetry
(per-bulb phase/theta + the per-env sampled lock parameters) into the bag whenever the task
wires ``mdp.bulb_attachment`` -- which this one does.

Run:
    PYTHONPATH=source/fiatlux_task:source/fiatlux_teleop \\
    uv run --extra teleop python scripts/teleop/sonic_teleop.py \\
        --task FIATLUX-TestLightbulbMechanism-Teleop-v0 --input keyboard --record bag
"""

from fiatlux_task.robots.g1 import (
    G1_ARM_JOINTS,
    G1_EE_BODY,
    G1_HAND_GRASP,
    G1_HAND_JOINTS,
    G1_HAND_OPEN,
    G1_LEFT_ARM_JOINTS,
    G1_LEFT_EE_BODY,
    G1_LEFT_HAND_GRASP,
    G1_LEFT_HAND_JOINTS,
    G1_LEFT_HAND_OPEN,
)
from fiatlux_task.tasks.manager_based.fiatlux_task.replace_env_cfg import ReplaceEnvCfg

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DeviceBase, DevicesCfg
from isaaclab.devices.openxr import XrAnchorRotationMode, XrCfg
from isaaclab.devices.openxr.openxr_device import OpenXRDeviceCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
)
from isaaclab.utils import configclass

from .xr_controller_retargeters import (
    ControllerGripperRetargeterCfg,
    Se3RelControllerRetargeterCfg,
)


@configclass
class TestLightbulbMechanismActionsCfg:
    """Bimanual absolute-EE-pose IK + binary Inspire grip (replaces the RL whole-body action).
    SONIC drives the legs+waist in the driver; these cover the arms + hands."""

    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=G1_ARM_JOINTS,
        body_name=G1_EE_BODY,
        controller=DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls",
            ik_params={"lambda_val": 0.05},   # match Insert/Carry-Teleop: relaxed rest pose
        ),
        scale=1.0,
    )
    hand_action = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=G1_HAND_JOINTS,
        open_command_expr=G1_HAND_OPEN,
        close_command_expr=G1_HAND_GRASP,
    )
    left_arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=G1_LEFT_ARM_JOINTS,
        body_name=G1_LEFT_EE_BODY,
        controller=DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls",
            ik_params={"lambda_val": 0.05},
        ),
        scale=1.0,
    )
    left_hand_action = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=G1_LEFT_HAND_JOINTS,
        open_command_expr=G1_LEFT_HAND_OPEN,
        close_command_expr=G1_LEFT_HAND_GRASP,
    )


@configclass
class TestLightbulbMechanismEnvCfg(ReplaceEnvCfg):
    """``FIATLUX-Replace-v0`` with a teleop action interface, benched for bayonet debugging."""

    # Debug bench: reachable ladder relative to the fixture (ReplaceEnvCfg's own aid).
    couple_ladder_to_fixture: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()

        # Deterministic canonical spawns: a debugging session must be repeatable.
        self.disable_randomization()

        # REACHABLE BENCH = the INSERT task's own tabletop layout, verbatim. The Replace preset
        # mounts the fixture at ceiling height (z ~= 3 m, unreachable; SONIC cannot climb), so we
        # re-apply Insert's proven manipulation bench instead: packing table, socket fixture
        # standing ON it, fresh bulb at hand height beside it, robot at the bench's +y side.
        from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import apply_tabletop_preset
        _ladder = self.scene.ladder   # the Insert preset drops the ladder, but Replace's obs/reward
        _old_bulb = self.scene.old_bulb  # ...and (on current main) the old bulb: Insert has none,
        apply_tabletop_preset(self.scene)  # so the preset nulls it -- restore both after.
        self.scene.old_bulb = _old_bulb
        self.scene.ladder = _ladder   # terms reference it -- keep it, but PARK it: its preset spot
        # (near the old ceiling fixture) is exactly where the table now stands.
        _lp = self.scene.ladder.init_state.pos
        self.scene.ladder.init_state.pos = (-2.6, 2.6, _lp[2])
        # Two Replace-context fixups the Insert preset does not know about:
        #  - the Replace preset ceiling-mounted the socket UPSIDE-DOWN; the bench is upright;
        self.scene.socket.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        #  - the old bulb must spawn SEATED in the (now upright, relocated) socket. Seat and plug
        #    offsets are equal (assets.py), so the seated root pose IS the socket root pose.
        # Work pieces near the robot's (+y) edge of the table, not the middle.
        _sp = self.scene.socket.init_state.pos
        self.scene.socket.init_state.pos = (_sp[0], _sp[1] + 0.08, _sp[2])
        _bp = self.scene.fresh_bulb.init_state.pos
        # -x = the operator's right (robot faces -y): fresh bulb offset further from the socket.
        self.scene.fresh_bulb.init_state.pos = (_bp[0] - 0.14, _bp[1] + 0.02, _bp[2])
        self.scene.old_bulb.init_state.pos = self.scene.socket.init_state.pos
        self.scene.old_bulb.init_state.rot = (1.0, 0.0, 0.0, 0.0)

        # Re-sync the hand-contact filters now that old_bulb is back. apply_tabletop_preset ran its
        # own _sync_bulb_contact_filters while old_bulb was still nulled (Insert has none), so BOTH
        # hand sensors ended up filtered to the fresh bulb ALONE -- and the operator removes the OLD
        # bulb. The sensor reports zero force for an unfiltered body, so without this every take
        # reads 0 N on both hands even while the bulb is being crushed (the 2026-09-01 bench bags:
        # peak contact 0.0 N, where S03 read 3.4 kN). Re-run it here, after the scene is final.
        from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import _sync_bulb_contact_filters

        _sync_bulb_contact_filters(self.scene)

        # SINK THE BENCH to SONIC working height. Insert BOLTS the robot at full standing height,
        # but here SONIC balances it at its own stance (pelvis ~0.74 m, knees bent) -- the robot
        # stands a head shorter, so Insert's 0.99 m tabletop lands at CHEST height and the hands
        # can't work the surface (operator screenshot, 2026-08-21). Lower the whole assembly --
        # table, socket, both bulbs -- by one delta so the surface sits at ~0.66 m (waist height,
        # hands comfortably above it, matching the Insert-session ergonomics). The table legs clip
        # ~28 cm into the floor: cosmetic, and irrelevant to a mechanism bench.
        _SINK = 0.33
        for _e in (self.scene.table, self.scene.socket, self.scene.fresh_bulb, self.scene.old_bulb):
            _p = _e.init_state.pos
            _e.init_state.pos = (_p[0], _p[1], _p[2] - _SINK)

        # teleop action interface (arms + hands); legs+waist are SONIC's, driven in sonic_teleop.py.
        self.actions = TestLightbulbMechanismActionsCfg()

        # Hand select (FIATLUX_TELEOP_HAND, default inspire). The subtask twins get this from
        # apply_subtask_teleop; the bench predates that machinery. swap_robot_variant carries the
        # binary grips' open/close presets across since d6e3862, so one call is the whole swap --
        # and the swapped-in G1_DEX3_CFG keeps self-collisions enabled.
        import os as _os
        if _os.environ.get("FIATLUX_TELEOP_HAND", "inspire").lower() == "dex3":
            from fiatlux_task.robots.g1 import swap_robot_variant
            swap_robot_variant(self, "dex3")

        # operator-paced: no automatic terminations (they would auto-reset mid-session).
        for _t in ("time_out", "success", "fell_below", "fell_over", "ladder_tipped",
                   "fresh_bulb_dropped", "old_bulb_dropped"):
            if getattr(self.terminations, _t, None) is not None:
                setattr(self.terminations, _t, None)

        # FOLLOW CAMERA: first-person pelvis anchor (same rationale/values as Carry-Teleop) so the
        # view tracks the robot as SONIC walks it to the fixture.
        rp = self.scene.robot.init_state.pos
        self.xr = XrCfg(
            anchor_pos=(0.0, 0.0, -0.9),
            anchor_rot=(0.0, 0.0, 0.0, 1.0),
            anchor_prim_path="/World/envs/env_0/Robot/pelvis",
            fixed_anchor_height=True,
            anchor_rotation_mode=XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED,
        )

        # controller_rel teleop device, root pose baked to the robot spawn (see Carry-Teleop).
        self.teleop_devices = DevicesCfg(
            devices={
                "controller_rel": OpenXRDeviceCfg(
                    retargeters=[
                        Se3RelControllerRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT,
                            root_pos=(rp[0], rp[1], rp[2] - 0.05),
                            sim_device=self.sim.device,
                        ),
                        ControllerGripperRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT, sim_device=self.sim.device
                        ),
                        Se3RelControllerRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_LEFT,
                            root_pos=(rp[0], rp[1], rp[2] - 0.05),
                            sim_device=self.sim.device,
                        ),
                        ControllerGripperRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_LEFT, sim_device=self.sim.device
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
            }
        )


# The bench's Inspire->DEX3 swap grew into the task-agnostic helper used for every
# free-base task; re-exported here so existing imports keep working.
from fiatlux_teleop.hand_swap import apply_dex3_hands  # noqa: F401,E402
