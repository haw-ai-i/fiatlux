"""The LadderMan <-> fiatlux contract, read off both sources rather than assumed.

Everything here is a place where the two projects agree on a NAME but not on a NUMBER, which is
the class of bug that produces a robot falling over with no error anywhere. Each constant below
cites where it came from so it can be re-checked when either side moves.

The action contract (LadderMan `envs/mdp/action.py::JointPosition.__call__`):

    pos_target = default_joint_pos + offset            # offset is zeros unless randomised
    pos_target[:, joint_ids] += applied_action * action_scaling
    asset.set_joint_position_target(pos_target)

and only the 23 named joints move -- the six wrists are held at default and are not actuated.

CORRECTION (read off `learning/ppo/ppo_teacher.py` + `common.py`, which an earlier version of
this file got backwards): the teacher is built with ``residual_action=True``, so `build_actor`
appends ``Mod(RefJointPos(), [REF_JPOS_KEY, "loc"], ["loc"])`` and ``RefJointPos.forward`` is
``ref_jpos + action``. The network's 23 outputs are an offset on the REFERENCE pose, in action
units -- not on the default pose. Substituting:

    applied_action = (ref_joint_pos - default) / action_scaling + net_out
    pos_target     = default + applied_action * action_scaling
                   = ref_joint_pos + net_out * action_scaling

So the absolute target is the REFERENCE joint angle plus a scaled residual, and the default pose
cancels out of the final target entirely. It still matters for `joint_pos_history`, which is
measured against it.

fiatlux S02 (`subtask_env_cfg.py`):

    mdp.JointPositionActionCfg(joint_names=[".*"], scale=0.5, use_default_offset=True)

    pos_target = default_joint_pos + 0.5 * action      # all 29 joints

Equating the two absolute targets gives the mapping in `to_fiatlux_action`. Note the two robots
do NOT share a default pose: LadderMan's G1 stands in a deep crouch with the arms forward, ours
stands nearly straight, and the difference reaches 27 deg at the knee. Dropping that term sends
every leg joint 15-27 deg off with no error raised, so it is carried explicitly.

Isaac Lab only clamps an action when `cfg.clip` is set (`joint_actions.py::process_actions`), and
S02 sets none, so the compensated action may exceed +-1 without being silently truncated.
"""

from __future__ import annotations

# LadderMan cfg/task/base/ladder.yaml -> action.action_scaling. 23 entries; the six wrist joints
# are deliberately absent ("wrist joints are not actuated by the policy").
ACTION_SCALING: dict[str, float] = {
    "hip_yaw": 0.55, "hip_roll": 0.35, "hip_pitch": 0.55,
    "knee": 0.35, "ankle_pitch": 0.44, "ankle_roll": 0.44,
    "waist_roll": 0.44, "waist_pitch": 0.44, "waist_yaw": 0.55,
    "shoulder_pitch": 0.44, "shoulder_roll": 0.44, "shoulder_yaw": 0.44,
    "elbow": 0.44,
}
UNACTUATED = ("wrist_roll", "wrist_pitch", "wrist_yaw")

# LadderMan active_adaptation/assets/g1.py -> init_state.joint_pos
LM_DEFAULT_JOINT_POS: dict[str, float] = {
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
}
# fiatlux source/fiatlux_task/fiatlux_task/robots/g1.py -> init_state.joint_pos
FIATLUX_DEFAULT_JOINT_POS: dict[str, float] = {
    ".*_hip_pitch_joint": -0.05,
    ".*_knee_joint": 0.2,
    ".*_ankle_pitch_joint": -0.15,
}

FIATLUX_ACTION_SCALE = 0.5          # subtask_env_cfg.py JointPositionActionCfg(scale=0.5)
LADDERMAN_STEP_DT = 0.02            # cfg sim.step_dt -- 50 Hz policy
LADDERMAN_PHYSICS_DT = 0.005        # cfg sim.isaac_physics_dt -- 200 Hz

# cfg/task/base/ladder.yaml -> observation groups, in the order the networks consume them.
#   encoder_priv: [priv, object] -> priv_feature
#   actor:        [command, policy, priv_feature] -> action
# (active_adaptation/learning/ppo/ppo_teacher.py module docstring)
ACTOR_GROUPS = ("command", "policy", "priv_feature")
ENCODER_GROUPS = ("priv", "object")


def _match(name: str, table: dict[str, float]) -> float | None:
    """Resolve a joint name against a table whose keys may be substrings or `.*` patterns."""
    import re
    for pat, val in table.items():
        if pat.startswith(".*") or pat.endswith(".*") or "(" in pat:
            if re.fullmatch(pat.replace(".*", ".*"), name):
                return val
        elif pat == name:
            return val
    return None


def scaling_for(joint_name: str) -> float | None:
    """LadderMan's per-joint action scale, or None if the joint is not actuated by the policy."""
    if any(w in joint_name for w in UNACTUATED):
        return None
    for key, val in ACTION_SCALING.items():
        if key in joint_name:
            return val
    return None


def default_delta(joint_name: str) -> float:
    """LadderMan's default joint angle minus ours, in radians.

    This is the term that is easy to forget and impossible to see: both sides call the quantity
    `default_joint_pos`, and both are correct for their own robot.
    """
    lm = _match(joint_name, LM_DEFAULT_JOINT_POS) or 0.0
    fx = _match(joint_name, FIATLUX_DEFAULT_JOINT_POS) or 0.0
    return lm - fx


def to_fiatlux_action(lm_action, joint_names):
    """Map a LadderMan action vector onto S02's 29-dim action space.

    `lm_action` is indexed by LadderMan's actuated-joint order; `joint_names` is S02's full joint
    name list in its own order. Returns a list the length of `joint_names`.
    """
    out = []
    for name in joint_names:
        scale = scaling_for(name)
        if scale is None:                       # wrist: held at default on both sides
            out.append(0.0)
            continue
        a = lm_action[name] if hasattr(lm_action, "get") else 0.0
        out.append((default_delta(name) + scale * a) / FIATLUX_ACTION_SCALE)
    return out
