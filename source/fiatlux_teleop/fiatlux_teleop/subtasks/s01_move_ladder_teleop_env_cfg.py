"""``FIATLUX-S01-MoveLadder-Teleop-v0`` -- walk to the ladder, take it, and stand it up, teleoperated.

S01MoveLadderEnvCfg with the shared teleop interface applied (arm-IK + binary grip in place of the RL
whole-body joint action, the pelvis-anchored XR follow camera and ``controller_rel`` device,
and the failure terminations cleared so the session is operator-paced). Legs and waist stay
SONIC's, driven by ``scripts/teleop/sonic_teleop.py``.

The longest subtask to teleoperate: one episode covers what the retired approach / grab / carry
/ place split scored as four, and the ladder is a free body positioned in the grip rather than
attached, so the grips have to close before it is walked anywhere.

Per-task teleop tweaks (camera height, arm spawn, staged object poses) belong HERE, after the
shared recipe -- keep :func:`~fiatlux_teleop.subtask_teleop.apply_subtask_teleop` generic.
"""

from fiatlux_task.tasks.manager_based.fiatlux_task.subtasks.s01_move_ladder_env_cfg import S01MoveLadderEnvCfg

from isaaclab.utils import configclass

from ..subtask_teleop import apply_subtask_teleop


@configclass
class S01MoveLadderTeleopEnvCfg(S01MoveLadderEnvCfg):
    """S01MoveLadderEnvCfg with the teleop action/XR interface."""

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_subtask_teleop(self)
