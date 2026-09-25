"""``FIATLUX-S11-ScrewInBulb-Teleop-v0`` -- seat the fresh bulb in the fixture, teleoperated.

S11ScrewInBulbEnvCfg with the shared teleop interface applied (arm-IK + binary grip in place of the RL
whole-body joint action, the pelvis-anchored XR follow camera and ``controller_rel`` device,
and terminations cleared so the session is operator-paced). Legs and waist stay SONIC's,
driven by ``scripts/teleop/sonic_teleop.py``.

Per-task teleop tweaks (camera height, arm spawn, staged object poses) belong HERE, after the
shared recipe -- keep :func:`~fiatlux_teleop.subtask_teleop.apply_subtask_teleop` generic.
"""

from fiatlux_task.tasks.manager_based.fiatlux_task.subtasks.s11_screw_in_bulb_env_cfg import S11ScrewInBulbEnvCfg

from isaaclab.utils import configclass

from ..subtask_teleop import apply_subtask_teleop


@configclass
class S11ScrewInBulbTeleopEnvCfg(S11ScrewInBulbEnvCfg):
    """S11ScrewInBulbEnvCfg with the teleop action/XR interface."""

    def __post_init__(self) -> None:
        super().__post_init__()
        apply_subtask_teleop(self)
