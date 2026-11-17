"""Drive a LadderMan teacher through a fiatlux climb subtask and record the bag.

A sibling of ``scripts/record_run.py``, not a modification of it: the teacher is not a
``make_policy`` spec (it needs four observation groups the benchmark does not build, its own
reference clip and its own phase clock), and wiring it in would mean editing both ``policy.py``
and the spec list that has to stay in sync with it. Everything else -- the env, the recorder, the
bag format -- is the benchmark's own.

Step order matters and mirrors LadderMan's: refresh the history buffers from the CURRENT state,
compute the groups, act, then record the action and advance the reference clock. That ordering is
what makes ``prev_actions`` and ``applied_action`` hold the PREVIOUS action at observation time,
which is what they held in training.

Examples
--------
    python scripts/climb/record_climb.py --headless --checkpoint .../checkpoint_2700.pt \
        --motion /home/yujin-chen/psi0_ft/climb/topout --out /tmp/climb-run --max_steps 1200
"""

import argparse
import pathlib
import sys

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser(description="Record a LadderMan teacher on a climb subtask.")
ap.add_argument("--task", default="FIATLUX-S02-ClimbLadder-v0")
ap.add_argument("--checkpoint", required=True, help="a LadderMan teacher .pt")
ap.add_argument("--motion", required=True, help="the motion dir the teacher was trained on")
ap.add_argument("--out", default=None, help="write a bag here (omit to only print the gate trace)")
ap.add_argument("--episodes", type=int, default=1)
ap.add_argument("--max_steps", type=int, default=6000, help="S02 is 120 s at 50 Hz = 6000")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--format", default="hdf5")
ap.add_argument("--print_every", type=int, default=100)
ap.add_argument("--match_torque", action="store_true",
                help="convert targets so OUR actuators reproduce LadderMan's torque")
ap.add_argument("--residual_scale", type=float, default=1.0,
                help="scale the NETWORK's 23 outputs. 0 = pure open-loop replay of the reference "
                     "through the same action path, which isolates the conversion from the policy.")
AppLauncher.add_app_launcher_args(ap)
a = ap.parse_args()
a.enable_cameras = True
app = AppLauncher(a).app

sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))

import fiatlux_task.tasks  # noqa: E402,F401
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from fiatlux_task.recording import TrajectoryRecorder  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from ladderman_policy import LadderManPolicy  # noqa: E402
from ladderman_runtime import LadderManObs  # noqa: E402

cfg = parse_env_cfg(a.task, num_envs=1)
env = gym.make(a.task, cfg=cfg)
base = env.unwrapped

obs_builder = LadderManObs(base, a.motion)
net = LadderManPolicy(a.checkpoint, device=base.device)
print(f"[POLICY] {a.checkpoint}")
print(f"[MOTION] {obs_builder.motion_len} steps at 50 Hz = {obs_builder.motion_len / 50:.2f} s")
print(f"[FRAME ] rung0 {[round(v, 3) for v in obs_builder.rung0_w[0].tolist()]}, "
      f"climb yaw {float(obs_builder.climb_yaw[0]) * 57.2958:+.1f} deg")

recorder = (
    TrajectoryRecorder(base, policy_spec=f"ladderman:{pathlib.Path(a.checkpoint).name}",
                       seed=a.seed, checkpoint=a.checkpoint, out_dir=a.out, fmt=a.format)
    if a.out else None
)

obs, _ = env.reset(seed=a.seed)
obs_builder.reset()

gate_hits = 0
best = {}
steps = 0
try:
    with torch.inference_mode():
        while steps < a.max_steps:
            obs_builder.update()                       # buffers see the CURRENT state
            groups = obs_builder.groups()
            a_lm = net.act(groups)
            if a.residual_scale != 1.0:
                a_lm = groups["ref_joint_pos_"] + (a_lm - groups["ref_joint_pos_"]) * a.residual_scale
            action = net.to_fiatlux(a_lm, obs_builder, match_torque=a.match_torque)
            obs_builder.step(a_lm)                     # record action, advance the clock

            obs, reward, terminated, truncated, extras = env.step(action)
            if recorder is not None:
                recorder.record_step(obs, action, reward, terminated, truncated)
            steps += 1

            # the gate conjuncts, read live so a run that never scores still says how close
            pel = float(base.scene["robot"].data.root_pos_w[0, 2])
            best["pelvis_max"] = max(best.get("pelvis_max", -9e9), pel)
            if steps % a.print_every == 0:
                print(f"[STEP  ] {steps:5d}  phase {float(obs_builder.t[0]) / obs_builder.motion_len:5.2f}"
                      f"  pelvis {pel:6.3f} m  action |max| {float(action.abs().max()):6.2f}")

            done = terminated | truncated
            if bool(done.any()):
                ids = done.nonzero(as_tuple=False).squeeze(-1)
                why = []
                tm = getattr(base, "termination_manager", None)
                if tm is not None:
                    for term in tm.active_terms:
                        try:
                            if bool(tm.get_term(term)[0]):
                                why.append(term)
                        except Exception:
                            pass
                print(f"[RESET ] step {steps}: {len(ids)} env(s) done "
                      f"(terminated={bool(terminated.any())}, truncated={bool(truncated.any())})"
                      f" reason={why or 'unknown'}")
                obs_builder.reset(ids)
                gate_hits += 1
                if gate_hits >= a.episodes:
                    break
finally:
    print(f"[RESULT] {steps} steps; highest pelvis {best.get('pelvis_max', float('nan')):.3f} m")
    if recorder is not None:
        info = recorder.write(a.out, fmt=a.format)
        print(f"[RESULT] wrote bag {info['bag']} ({info['episodes']} episodes)")

env.close()
app.close()
