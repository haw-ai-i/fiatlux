# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``fiatlux_teleop`` -- VR teleoperation extension for the Fiatlux benchmark.

Teleop is tooling *on top of* the benchmark, kept as its own package so the core ``fiatlux_task``
benchmark imports and runs without teleop's heavy deps (OpenXR / CloudXR / the SONIC onnxruntime
stack). The dependency arrow points teleop -> benchmark, never the reverse: the env cfgs here
subclass the benchmark envs (``fiatlux_task.tasks…``) and swap their RL whole-body action for an
arm-IK + binary-grip interface driven by ``scripts/teleop/sonic_teleop.py``.

``import fiatlux_teleop`` registers the teleop gym tasks below (lazy string entry points -- the cfg
modules load only when a teleop task is actually made, so importing this package is cheap). The
teleop scripts import both ``fiatlux_task`` (benchmark tasks) and ``fiatlux_teleop`` (these).
"""

import gymnasium as gym

# Teleop variant of Insert (arm differential-IK + binary grip); drive with scripts/teleop/sonic_teleop.py
# (whole-body: SONIC legs + arm teleop, --input vr|keyboard).
gym.register(
    id="FIATLUX-Insert-Teleop-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.insert_teleop_env_cfg:G1BulbInsertTeleopEnvCfg"},
)

# Teleop variant of Carry (ladder-positioning): bimanual arm IK + Dex3 grip, SONIC legs
# (scripts/teleop/sonic_teleop.py).
gym.register(
    id="FIATLUX-Carry-Teleop-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.carry_teleop_env_cfg:CarryTeleopEnvCfg"},
)

# Walk-through gallery of every ladder design (open floor, all designs in a grid), same teleop
# machinery as Carry-Teleop.
gym.register(
    id="FIATLUX-LadderGallery-Teleop-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.ladder_gallery_teleop_env_cfg:LadderGalleryTeleopEnvCfg"},
)

# --- teleop twins of the 12 benchmark subtasks -------------------------------------------
# Each has its own thin cfg file under `subtasks/` (matching how the benchmark writes its
# subtasks: explicit file per task + shared behaviour in a common module -- here
# `subtask_teleop.apply_subtask_teleop`). Registration is table-driven so ids and entry
# points cannot drift apart.
from .subtask_teleop import SUBTASKS  # noqa: E402

for _tid, _mod, _cls in SUBTASKS:
    _teleop_mod = _mod.replace("_env_cfg", "_teleop_env_cfg")
    _teleop_cls = _cls.replace("EnvCfg", "TeleopEnvCfg")
    gym.register(
        id=f"{_tid}-Teleop-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": f"{__name__}.subtasks.{_teleop_mod}:{_teleop_cls}"},
    )
