# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record a Fiatlux run: a video of the rollout and/or a trajectory "bag".

A single rollout produces both artifacts so they describe the *same* run:

- ``video/`` : an MP4 of the run plus a poster PNG, captured by an RTX sensor camera
  (``fiatlux_task.viz``) posed by ``--cam``: fixed ``third_person`` / ``closeup``
  viewpoints, a 360-degree ``orbit`` of the scene, or ``ego`` -- the robot's own
  head-mounted ``ego_camera`` sensor (any task whose scene attaches one), unposed
  since it already moves with the robot.
- ``run.h5`` + ``meta.json`` : the experiment bag -- every per-step signal needed to
  score the run offline (see ``scripts/score.py``). ``--format npz`` for a flat fallback.

The policy is anything ``make_policy`` accepts (``zero`` / ``random`` / a TorchScript
``.pt`` / ``rsl_rl[:<ckpt>]`` / ``sonic_stand`` / ``groot``) -- the recorder is
policy-agnostic.

Examples:
    python scripts/record_run.py --task FIATLUX-Insert-v0 --policy random \
        --episodes 2 --record both --out logs/runs/random0
    python scripts/record_run.py --task FIATLUX-Insert-v0 --policy logs/.../policy.pt \
        --record bag --episodes 50 --seed 0 --out logs/runs/policyA
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record a Fiatlux run (video and/or bag).")
parser.add_argument("--task", type=str, required=True, help="Task / env id.")
parser.add_argument(
    "--policy",
    type=str,
    default="zero",
    help="Policy spec: zero | random | <path>.pt | rsl_rl[:<ckpt>].",
)
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path for rsl_rl policies.")
parser.add_argument(
    "--record",
    type=str,
    default="both",
    choices=["video", "bag", "both"],
    help="Which artifacts to produce.",
)
parser.add_argument(
    "--no_randomize",
    action="store_true",
    default=False,
    help="Deterministic canonical spawns: strip the task's reset-time randomization terms.",
)
parser.add_argument("--episodes", type=int, default=1, help="Episodes to record.")
parser.add_argument("--num_envs", type=int, default=1, help="Parallel envs (bag covers all).")
parser.add_argument("--seed", type=int, default=0, help="Run seed.")
parser.add_argument("--out", type=str, required=True, help="Output directory.")
parser.add_argument("--format", type=str, default="hdf5", choices=["hdf5", "npz"], help="Bag file format.")
parser.add_argument(
    "--cam",
    type=str,
    default="third_person",
    choices=["third_person", "closeup", "orbit", "fixture", "ego"],
    help="Camera pose for the video: fixed presets, a 360-degree scene orbit, a low orbit looking UP "
    "at the mounted fixture, or the robot's own ego_camera sensor.",
)
parser.add_argument("--video_length", type=int, default=600, help="Video length (env steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Use USD I/O.")
parser.add_argument(
    "--instruction",
    type=str,
    default=None,
    help="Language instruction for VLA policies (groot); default: the task's canonical sentence.",
)
parser.add_argument(
    "--robot",
    type=str,
    default="inspire",
    choices=["inspire", "dex3"],
    help="G1 hand variant. dex3 matches GR00T's REAL_G1 embodiment.",
)
# Benchmark telemetry flags (--wandb, --wandb_project, ...); mirrors fiatlux_task.telemetry.
parser.add_argument("--wandb", action="store_true", default=False, help="Stream the score breakdown to wandb.")
parser.add_argument("--wandb_project", type=str, default="fiatlux", help="wandb project name.")
parser.add_argument("--wandb_entity", type=str, default=None, help="wandb entity (team/user).")
parser.add_argument("--wandb_run_name", type=str, default=None, help="wandb run name.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

want_video = args_cli.record in ("video", "both")
want_bag = args_cli.record in ("bag", "both")
# Cameras are required to render video frames -- and to build the env at all. EVERY FIATLUX task
# calls `add_ego_camera`, and Isaac Lab raises at startup for a camera spawned without the flag.
# Gating this on `want_video` meant `--record bag` died before the first step on every task, with
# an error about rendering that reads like a video problem rather than a missing flag.
args_cli.enable_cameras = True
# Always headless (video still renders via enable_cameras). AppLauncher's --headless is
# store_true default False -- never None -- so the old None-guard was dead code and a
# displayless machine wedged in GUI mode; use --livestream for interactive viewing.
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

import os
import random
import sys
import traceback

import fiatlux_task.tasks  # noqa: F401
import gymnasium as gym
import torch
from fiatlux_task.policy import make_policy
from fiatlux_task.recording import TrajectoryRecorder
from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import set_layout_seed
from fiatlux_task.telemetry import ScoreLogger
from fiatlux_task.viz import VideoRecorder, fixture_orbit, make_video_camera_cfg, orbit_pose

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def _cam_pose_fn(kind: str, env_cfg):
    """Camera pose per captured frame ``(i, n_frames) -> (eye, lookat)``.

    ``third_person`` frames the task cfg's own viewer eye/lookat; ``orbit`` turntables
    around the cfg's orbit fields (the same framing verify_scene --record uses), so every
    task -- bench-scale Insert or room-scale Replace -- records its own layout. ``closeup``
    stays the Insert bench's fixed close view. ``fixture`` orbits below the mount looking up,
    the only framing here that shows an overhead fixture (every other one points at the
    floor and the bench).
    """
    if kind == "third_person":
        eye, lookat = tuple(env_cfg.viewer.eye), tuple(env_cfg.viewer.lookat)
        return lambda i, n: (eye, lookat)
    if kind == "closeup":
        return lambda i, n: ((0.9, 0.8, 1.4), (0.45, 0.0, 1.15))
    if kind == "fixture":
        orbit = fixture_orbit(env_cfg)
        return lambda i, n: orbit_pose(i, n, **orbit)
    center = getattr(env_cfg, "orbit_center", (0.45, 0.0, 1.1))
    radius = getattr(env_cfg, "orbit_radius", 2.6)
    height = getattr(env_cfg, "orbit_height", 2.0)
    return lambda i, n: orbit_pose(i, n, center=center, radius=radius, height=height)


def main():
    # The room layout is drawn at cfg-build time, so its seed is declared before
    # parse_env_cfg (same determinism contract as eval.py).
    set_layout_seed(args_cli.seed)
    random.seed(args_cli.seed)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    if args_cli.robot != "inspire":
        from fiatlux_task.robots.g1 import swap_robot_variant

        swap_robot_variant(env_cfg, args_cli.robot)
    if args_cli.no_randomize:
        env_cfg.disable_randomization()
    if want_video and args_cli.cam == "ego":
        if getattr(env_cfg.scene, "ego_camera", None) is None:
            raise ValueError(f"--cam ego needs an ego_camera sensor; {args_cli.task}'s scene does not attach one.")
    elif want_video:
        # RTX sensor camera for the video (fiatlux_task.viz), posed per frame from --cam.
        env_cfg.scene.video_cam = make_video_camera_cfg()

    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped

    video = None
    # ego_camera is body-attached and moves with the robot; nothing to pose per frame.
    pose_fn = None if args_cli.cam == "ego" else _cam_pose_fn(args_cli.cam, env_cfg)
    if want_video:
        video_path = os.path.join(args_cli.out, "video", "run.mp4")
        cam_name = "ego_camera" if args_cli.cam == "ego" else "video_cam"
        video = VideoRecorder(base_env, base_env.scene[cam_name], video_path)
        print(f"[INFO] recording video to {video_path}")

    policy = make_policy(args_cli.policy, base_env, checkpoint=args_cli.checkpoint, instruction=args_cli.instruction)
    recorder = (
        TrajectoryRecorder(
            base_env,
            policy_spec=args_cli.policy,
            seed=args_cli.seed,
            checkpoint=args_cli.checkpoint,
            out_dir=args_cli.out,
            fmt=args_cli.format,
        )
        if want_bag
        else None
    )
    # All metric definitions live in fiatlux_task.telemetry; this loop feeds it raw
    # step artifacts (sink-less when --wandb is off; aggregation still runs).
    score_logger = ScoreLogger.from_args(args_cli, extra_config={"record": args_cli.record})

    obs, _ = env.reset(seed=args_cli.seed)
    try:
        with torch.inference_mode():
            while score_logger.episodes_done < args_cli.episodes:
                actions = policy(obs)
                obs, reward, terminated, truncated, extras = env.step(actions)
                if video is not None and len(video) < args_cli.video_length:
                    pose = pose_fn(len(video), args_cli.video_length) if pose_fn is not None else None
                    video.capture(pose)
                if video is not None and recorder is None and len(video) >= args_cli.video_length:
                    break
                if recorder is not None:
                    recorder.record_step(obs, actions, reward, terminated, truncated)
                score_logger.step(base_env, extras, terminated | truncated)
    finally:
        # A streaming (out_dir-backed) recorder holds an open HDF5 file from the moment it is
        # constructed. Without this, a mid-run exception (env.step, the policy, ...) skips
        # write()'s close() and leaves that file open -- exactly the run worth keeping most.
        if recorder is not None:
            info = recorder.write(args_cli.out, fmt=args_cli.format)
            print(f"[INFO] wrote bag {info['bag']} ({info['episodes']} episodes)")
            print(f"[INFO] wrote metadata {info['meta']}")

    if video is not None:
        video_file = video.write()
        print(f"[INFO] wrote video {video_file}")
        score_logger.video(video_file, caption=f"{args_cli.task} / {args_cli.policy}")
    score_logger.close()

    env.close()


if __name__ == "__main__":
    # Kit runs non-daemon threads, so an uncaught exception leaves the process alive
    # spinning at ~100% CPU instead of dying -- a crash then looks indistinguishable from
    # a very slow run (a bad reward term once burned three hours that way). Closing the app
    # in a `finally` is NOT enough: with the env left un-closed, simulation_app.close()
    # itself blocks, so the interpreter never reaches the traceback. Print it first, then
    # hard-exit past the hung threads. The success path closes normally.
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    simulation_app.close()
