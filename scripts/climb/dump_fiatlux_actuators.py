"""Dump the LIVE fiatlux robot's resolved per-joint actuator numbers to JSON.

The point is independence. ``ladderman_g1_fiatlux.py`` transcribes fiatlux's actuator model into
LadderMan's config by hand, and checking a hand transcription against a hand-written table of the
same numbers catches nothing -- the same typo sits on both sides. This reads what PhysX actually
resolved in a real fiatlux env, so comparing it against the trainer's own ``asset_meta.json``
compares two independently produced artifacts.

    python scripts/climb/dump_fiatlux_actuators.py --headless --out /tmp/fx_actuators.json
"""

import argparse
import json
import pathlib

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser(description="Dump resolved fiatlux actuator numbers.")
ap.add_argument("--task", default="FIATLUX-S02-ClimbLadder-v0")
ap.add_argument("--out", default="/tmp/fx_actuators.json")
ap.add_argument("--layout_seed", type=int, default=0)
AppLauncher.add_app_launcher_args(ap)
a = ap.parse_args()
a.enable_cameras = True
app = AppLauncher(a).app

import fiatlux_task.tasks  # noqa: E402,F401
import gymnasium as gym  # noqa: E402
from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import set_layout_seed  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

set_layout_seed(a.layout_seed)
env = gym.make(a.task, cfg=parse_env_cfg(a.task, num_envs=1)).unwrapped
env.reset()

rob = env.scene["robot"]
d = rob.data
out = {}
for i, name in enumerate(rob.joint_names):
    out[name] = {
        "stiffness": round(float(d.joint_stiffness[0, i]), 6),
        "damping": round(float(d.joint_damping[0, i]), 6),
        "armature": round(float(d.joint_armature[0, i]), 9),
        "effort_limit": round(float(d.joint_effort_limits[0, i]), 4),
    }

pathlib.Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True))
print(f"[DUMP] {len(out)} joints -> {a.out}")
for n in (
    "left_hip_pitch_joint",
    "left_knee_joint",
    "waist_pitch_joint",
    "left_ankle_pitch_joint",
    "left_shoulder_pitch_joint",
):
    if n in out:
        v = out[n]
        print(
            f"  {n:28s} kp {v['stiffness']:7.2f}  kd {v['damping']:6.3f}  "
            f"armature {v['armature']:.9f}  effort {v['effort_limit']:6.1f}"
        )

app.close()
