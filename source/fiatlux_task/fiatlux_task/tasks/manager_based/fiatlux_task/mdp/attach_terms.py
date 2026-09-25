# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rigid-attach ("weld") an object a subtask starts already holding to a robot body link.

A start state built by ``nav_terms.compose_carried_pose`` alone is not physically held:
nothing but incidental finger-mesh contact resists gravity, so a multi-kilogram payload swings
free of a one-point contact and falls within a few physics steps.

The fix is Isaac Sim's own standard runtime-attach mechanism: a ``UsdPhysics.FixedJoint``
between the two bodies, the same primitive ``isaacsim.robot_setup.assembler.RobotAssembler``
uses to rigidly mount a tool onto a robot. :func:`weld_to_body` is that, scoped to a
manager-based env's ``mode="reset"`` event and vectorized over ``env_ids``.

Scope: built for subtasks that never let go mid-episode (their own success gate requires the
grip held the whole time, e.g. S05's "old bulb still held" conjunct) -- there the weld is
unconditional and correct for the full episode. It is deliberately NOT wired into S06 (or the
mate-tier S03/S11): those subtasks' entire job is the act of releasing, which a
permanent weld cannot demonstrate, and a *correct* release trigger has to come from something
the policy actually controls (a commanded finger-open crossing some threshold, most likely),
not the ``object_released`` contact-force sensor already used for their success gates --
that sensor reads real finger-mesh contact force, which a welded object has no physical need to
keep exerting, so it would report "released" from the very first physics step. Wiring a release
half of this mechanism is future work, not a small addition to this one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    import torch

    from isaaclab.envs import ManagerBasedEnv


def weld_to_body(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg,
    body_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Freeze ``object_cfg``'s current pose relative to ``robot_cfg``'s ``body_name`` link.

    Reads whatever relative pose the two bodies are ALREADY at (set earlier the same reset by
    ``compose_carried_pose`` writing the object's ``init_state``) and locks it with a
    ``UsdPhysics.FixedJoint`` -- this moves nothing itself, it only stops gravity from moving
    it. Idempotent per env: removes a joint this function created on an earlier reset before
    creating the new one, so repeated resets in one process don't stack duplicate joints on the
    same env slot.

    Reads the relative pose from ``.data.body_pos_w``/``.data.body_quat_w`` (the physics tensor
    state, already reset by the time this runs -- ``EventCfg``'s own docstring: declaration
    order is execution order), not raw USD prim transforms: those go through Fabric/PhysX
    buffers this codebase never assumes are synced to the USD stage on a given frame (nothing
    else here reads a pose that way either).
    """
    import omni.usd
    from pxr import Gf, Sdf, UsdPhysics

    from isaaclab.utils.math import quat_apply_inverse, quat_inv, quat_mul

    stage = omni.usd.get_context().get_stage()

    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    body_idx = robot.body_names.index(body_name)
    # Resolved per-env prim paths off the PhysX views, not string-templated from
    # ``cfg.prim_path``: by the time an event runs, ``{ENV_REGEX_NS}`` has already been resolved
    # into the view's own regex, so that placeholder is gone and a naive ``.replace`` is a
    # silent no-op that leaves a literal ``env_.*`` in the path (caught by hand: PhysX rejected
    # the resulting joint target outright).
    body0_paths = robot.root_physx_view.link_paths
    body1_paths = obj.root_physx_view.prim_paths

    pos0 = robot.data.body_pos_w[env_ids, body_idx]
    quat0 = robot.data.body_quat_w[env_ids, body_idx]
    pos1 = obj.data.root_pos_w[env_ids]
    quat1 = obj.data.root_quat_w[env_ids]
    # Body0's pose expressed in body1's local frame -- the joint's frame1, with frame0 left at
    # body0's own origin (identity) -- so the joint locks in exactly the relative transform the
    # two bodies already have, not "teleport them together".
    local_pos1 = quat_apply_inverse(quat1, pos0 - pos1)
    local_rot1 = quat_mul(quat_inv(quat1), quat0)

    for row, i in enumerate(env_ids.tolist()):
        env_root = env.scene.env_prim_paths[i]
        body0_path = body0_paths[i][body_idx]
        body1_path = body1_paths[i]
        joint_path = f"{env_root}/CarryAttachJoint"

        if stage.GetPrimAtPath(joint_path).IsValid():
            stage.RemovePrim(joint_path)

        joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
        joint.GetBody0Rel().SetTargets([Sdf.Path(body0_path)])
        joint.GetBody1Rel().SetTargets([Sdf.Path(body1_path)])
        p = local_pos1[row].tolist()
        q = local_rot1[row].tolist()  # (w, x, y, z)
        joint.GetLocalPos1Attr().Set(Gf.Vec3f(*p))
        joint.GetLocalRot1Attr().Set(Gf.Quatf(q[0], Gf.Vec3f(*q[1:])))
