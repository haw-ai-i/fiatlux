"""The OBSERVATION half of the LadderMan <-> fiatlux contract: what the teacher expects to be
fed, term by term, with every width derived rather than copied.

Why this is a table and not a docstring. The rollout policy is ``Seq(encoder_priv, actor)`` --
``encoder_priv`` takes ``[priv, object]`` and ``actor`` takes ``[command, policy, priv_feature]``
-- so ALL FOUR groups are needed at inference, including the 1631-wide ``priv`` that reads like
a critic-only input. Feed a group at the right width but in the wrong term order and nothing
raises: the policy just acts on noise. So each group below is spelled out as the ordered terms
Hydra instantiates, each width is computed from the robot and the tracking config, and
``verify()`` checks the totals against the VecNorm running statistics stored in the checkpoint
itself. The checkpoint is the only authority that cannot drift.

Sources, all read rather than assumed:
  cfg/task/base/ladder.yaml                          the four observation groups, in order
  envs/mdp/commands/hdmi/observations.py             the command/object/diff_* terms
  envs/mdp/observations/common.py                    the history, action and torque terms
  envs/mdp/observations/priv_body.py                 body_pos_b / body_vel_b / body_height

Usage:
    python scripts/climb/ladderman_obs.py --checkpoint /path/to/checkpoint_2700.pt
"""

from __future__ import annotations

# ---------------------------------------------------------------------------------------------
# Shapes the widths are built from. Each is resolved from a config pattern against the G1, so a
# changed robot or tracking list changes these and the assertion in verify() fires.
# ---------------------------------------------------------------------------------------------

# command.future_steps = [1, 2, 8, 16, 32]
NUM_FUTURE_STEPS = 5

# command.tracking_keypoint_names resolved against g1_29dof:
#   .*_hip_(pitch|yaw)_link 4 | .*_knee_link 2 | .*_ankle_roll_link 2 | pelvis 1
#   torso_link 1 | .*_shoulder_pitch_link 2 | .*_elbow_link 2 | .*_wrist_yaw_link 2
NUM_TRACKING_BODIES = 16

# command.tracking_joint_names resolved against g1_29dof:
#   waist_.*_joint 3 | .*_hip_.*_joint 6 | .*_knee_joint 2 | .*_ankle_.*_joint 4
#   .*_shoulder_.*_joint 6 | .*_elbow_joint 2
NUM_TRACKING_JOINTS = 23

# The robot has 29 joints; the policy actuates 23 of them (the six wrists are held at default,
# see ladderman_contract.ACTION_SCALING). `joint_pos_history` defaults to joint_names=".*" and so
# carries all 29, while `applied_action` carries only what the action manager drives.
NUM_JOINTS = 29
NUM_ACTUATED = 23

# priv's three history terms use history_steps [0..8]; policy's use [0] / [0,1,2,3,4,8].
PRIV_HISTORY_STEPS = 9
POLICY_JOINT_HISTORY_STEPS = 6

_B, _J, _F = NUM_TRACKING_BODIES, NUM_TRACKING_JOINTS, NUM_FUTURE_STEPS

# ---------------------------------------------------------------------------------------------
# The groups, in the order Hydra instantiates them. (term, width, why)
# ---------------------------------------------------------------------------------------------

COMMAND = [
    ("ref_body_pos_future_local", _F * _B * 3, "future body positions in the MOTION root's yaw frame, z zeroed"),
    ("ref_joint_pos_future",      _F * _J,     "future reference joint angles"),
    ("ref_motion_phase",          1,           "t / motion_len -- the only term we must drive ourselves"),
]

POLICY = [
    ("root_ang_vel_history",      1 * 3,                          "history_steps [0], noise 0.05"),
    ("projected_gravity_history", 1 * 3,                          "history_steps [0], noise 0.05"),
    ("joint_pos_history",         POLICY_JOINT_HISTORY_STEPS * NUM_JOINTS,
                                  "steps [0,1,2,3,4,8]; MEAN of the last two physics substeps, minus the action offset"),
    ("prev_actions",              3 * NUM_ACTUATED,               "steps=3 of the action buffer"),
]

OBJECT = [
    ("ladder_pos_b",     2, "rung 0's xy in the robot's YAW-ONLY root frame -- not per-rung"),
    ("ladder_heading_b", 2, "[cos, sin] of wrap_to_pi(rung 0 yaw - robot yaw)"),
]

PRIV = [
    ("root_ang_vel_history",           PRIV_HISTORY_STEPS * 3,       "steps [0..8], no noise"),
    ("projected_gravity_history",      PRIV_HISTORY_STEPS * 3,       "steps [0..8], no noise"),
    ("joint_pos_history",              PRIV_HISTORY_STEPS * NUM_JOINTS, "steps [0..8], all 29 joints"),
    ("ref_root_pos_future_b",          _F * 3,                       "future reference root position, robot root frame"),
    ("ref_root_ori_future_b",          _F * 2 * 3,                   "FIRST TWO ROWS of the rotation matrix (6D rotation), not a quaternion"),
    ("diff_body_pos_future_local",     _F * _B * 3,                  "reference minus robot, body positions"),
    ("diff_body_ori_future_local",     _F * _B * 2 * 3,              "same 6D rotation convention -- the single largest term"),
    ("diff_body_lin_vel_future_local", _F * _B * 3,                  "reference minus robot, body linear velocity"),
    ("diff_body_ang_vel_future_local", _F * _B * 3,                  "reference minus robot, body angular velocity"),
    ("root_linvel_b",                  3,                            "root linear velocity, body frame"),
    ("body_pos_b",                     2 * 3,                        "body_names '.*ankle_roll_link' -> both ankles"),
    ("body_vel_b",                     2 * 3,                        "both ankles"),
    ("body_height",                    4,                            "both ankles + pelvis + torso_link"),
    ("applied_action",                 NUM_ACTUATED,                 "what the action manager actually applied"),
    ("applied_torque",                 NUM_JOINTS,                   "joint_names defaults to '.*', so all 29 -- NOT the 23 actuated"),
]

# A fifth input, and the one easiest to miss: the teacher is built with `residual_action=True`,
# so `build_actor` appends ``RefJointPos()``, whose forward is literally ``ref_jpos + action``.
# The network's 23 outputs are therefore an offset on the REFERENCE pose expressed in action
# units, not a standalone action. It is not a VecNorm group -- the checkpoint's running stats
# cover only the four above -- because it bypasses the trunk and is added to the head's output.
REF_JPOS = [
    ("ref_joint_pos_action_policy", NUM_ACTUATED,
     "(ref_joint_pos - LadderMan default) / action_scaling, over the 23 actuated joints"),
]

GROUPS = {"command": COMMAND, "policy": POLICY, "object": OBJECT, "priv": PRIV,
          "ref_joint_pos_": REF_JPOS}

# Only these four carry VecNorm running statistics in the checkpoint.
VECNORM_GROUPS = ("command", "policy", "object", "priv")


def width(group: str) -> int:
    return sum(w for _, w, _ in GROUPS[group])


def offsets(group: str) -> list[tuple[str, int, int]]:
    """(term, start, stop) slice of each term inside its group's flat vector."""
    out, at = [], 0
    for name, w, _ in GROUPS[group]:
        out.append((name, at, at + w))
        at += w
    return out


def verify(checkpoint_path: str) -> dict[str, tuple[int, int]]:
    """Check every group's derived width against the checkpoint's own VecNorm statistics.

    VecNorm keeps a running sum/sumsq per observation element, so its shapes ARE the widths the
    trained network was fed. Raises on any mismatch.
    """
    import torch

    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    stats = ck["vecnorm"]["_extra_state"]
    result, bad = {}, []
    for group in VECNORM_GROUPS:
        key = f"{group}_sum"
        if key not in stats:
            bad.append(f"{group}: checkpoint has no '{key}'")
            continue
        got, want = int(stats[key].shape[0]), width(group)
        result[group] = (want, got)
        if got != want:
            bad.append(f"{group}: derived {want}, checkpoint {got} (off by {want - got:+d})")
    if bad:
        raise SystemExit("observation contract does NOT match the checkpoint:\n  " + "\n  ".join(bad))
    return result


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", default=None, help="a teacher .pt to check the widths against")
    a = ap.parse_args()

    for group in GROUPS:
        print(f"\n{group}  ({width(group)})")
        for name, start, stop in offsets(group):
            why = next(w for n, _, w in GROUPS[group] if n == name)
            print(f"  [{start:5d}:{stop:5d}] {stop-start:5d}  {name:32s} {why}")

    if a.checkpoint:
        print()
        for group, (want, got) in verify(a.checkpoint).items():
            print(f"[OK] {group:8s} derived {want:5d} == checkpoint {got:5d}")
        print("[OK] all four groups match the trained artifact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
