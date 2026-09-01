# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay the recorded grasp that would not let go (issue #92).

``diagnose_stuck_bulb.py`` builds a grasp by parking the bulb at the palm and closing the fingers.
That rig kept measuring its own placement: the bulb's radius is about 39 mm and the parked centre
sits within 2 mm of the palm body origin, so the bulb probably encloses the palm geometry and the
huge contact forces may be depenetration of an overlap the rig itself created.

This does not place the bulb at all. Yujin's hand placed it on 2026-08-21, and the bag recorded
where. The replay drives the robot's 43 joints to their recorded values and puts the bulb at its
recorded pose, so the geometry under test is a grasp a person actually made.

The bag has no robot root pose, so bag-world and sim-world do not share an origin. Both do record
the right wrist (``eef_pose``), and the joints are the same in both, so the rigid transform that
maps the recorded wrist onto the simulated one maps the whole recorded world onto the simulated
one. That transform is applied to the bulb.

What the outcome means:

* bulb hangs at the hand -> the pathology is real, and this is the geometry that produces it.
* bulb falls -> the sticking needs something the bag does not carry, and the synthetic rig's
  forces were its own artifact.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay a recorded grasp and watch the bulb (issue #92).")
parser.add_argument("--bag", type=str, default="/home/pasha/bags_92/run.h5")
parser.add_argument("--meta", type=str, default="/home/pasha/bags_92/meta.json")
parser.add_argument("--task", type=str, default="FIATLUX-Remove-v0")
parser.add_argument("--variant", type=str, default="dex3", choices=["dex3", "inspire"])
parser.add_argument("--start", type=int, default=1520, help="Bag step to seed from (hand still closed).")
parser.add_argument("--steps", type=int, default=58, help="Bag steps to replay forward.")
parser.add_argument("--free-bulb", action="store_true", help="Seed the bulb once, then let physics own it.")
parser.add_argument(
    "--bare",
    action="store_true",
    help="Drop the scene furniture and pin the robot's root. The bag came from a different task, so "
    "its table, socket and crate sit elsewhere -- the recorded bulb pose lands INSIDE this task's "
    "table and is ejected at 2 m/s before the hand is ever involved. Only the hand matters here.",
)
parser.add_argument("--render", type=str, default=None, help="Write a PNG of the seeded grasp to this path.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything else follows."""

import contextlib  # noqa: E402
import json  # noqa: E402

import fiatlux_task.tasks  # noqa: F401, E402  -- registers the FIATLUX Gym environments

with contextlib.suppress(ImportError):  # the teleop benches live in a package that is not always installed
    import fiatlux_teleop  # noqa: F401
import gymnasium as gym  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from fiatlux_task.robots.g1 import (  # noqa: E402
    G1_DEX3_HAND_GRASP,
    G1_DEX3_LEFT_HAND_GRASP,
    swap_robot_variant,
)
from fiatlux_task.tasks.manager_based.fiatlux_task.mdp import observations as _obs  # noqa: E402
from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import set_layout_seed  # noqa: E402
from fiatlux_task.viz import make_video_camera_cfg  # noqa: E402

from isaaclab.utils.math import quat_apply, quat_inv, quat_mul  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def _repoint_binary_hand_commands(cfg, variant: str) -> None:
    """Rekey a ``BinaryJointPositionActionCfg``'s open/close dicts onto the swapped hand.

    ``swap_robot_variant`` remaps a term's ``joint_names`` but not the ``open_command_expr`` and
    ``close_command_expr`` dicts beside them, which stay keyed by the ORIGINAL hand's joint names.
    The action term then resolves those keys against the new robot and raises. This affects any
    task with a binary hand action, not just this replay -- the teleop benches are where it bites.

    Worked around here rather than fixed in ``robots/g1.py``, because this branch is diagnostic
    tooling and that is task code. It deserves its own issue.
    """
    if variant != "dex3":
        return
    for name in dir(cfg.actions):
        term = getattr(cfg.actions, name, None)
        if term is None or not hasattr(term, "open_command_expr"):
            continue
        joints = list(term.joint_names)
        grasp = G1_DEX3_LEFT_HAND_GRASP if any(j.startswith("left_") for j in joints) else G1_DEX3_HAND_GRASP
        term.open_command_expr = {j: 0.0 for j in joints}
        term.close_command_expr = {j: float(grasp.get(j, 0.0)) for j in joints}
        print(f"ACTION  repointed {name} onto {len(joints)} {variant} joints", flush=True)


def main() -> int:

    with open(args_cli.meta) as meta_file:
        meta = json.load(meta_file)
    bag_joint_names = meta["joint_names"]
    with h5py.File(args_cli.bag, "r") as f:
        demo = f["data"][list(f["data"].keys())[0]]
        joint_pos = np.asarray(demo["joint_pos"])
        eef_pose = np.asarray(demo["eef_pose"])
        bulb_pos = np.asarray(demo["old_bulb_pos"])
        bulb_quat = np.asarray(demo["old_bulb_quat"])

    set_layout_seed(0)
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    if args_cli.variant != "inspire":
        swap_robot_variant(cfg, args_cli.variant)
        _repoint_binary_hand_commands(cfg, args_cli.variant)
    if args_cli.bare:
        # MOVE the furniture aside rather than deleting it. Event and observation terms reference
        # these entities by name -- `reset_socket` and `privileged/socket_pose` both take the
        # socket -- so removing one makes the env fail to build. 20 m away is out of the way.
        for entity in ("table", "bin", "ladder", "socket", "pendant", "fixture"):
            item = getattr(cfg.scene, entity, None)
            if item is not None and getattr(item, "init_state", None) is not None:
                pos = item.init_state.pos
                item.init_state.pos = (pos[0] + 20.0, pos[1] + 20.0, pos[2])
        # A free base under zero actions sags, and the bag->sim transform IS the root difference
        # between the two robots -- a sagging root tilts it, and with it the direction of "up".
        cfg.scene.robot.spawn.articulation_props.fix_root_link = True
    if args_cli.render:
        cfg.scene.video_cam = make_video_camera_cfg()
    env = gym.make(args_cli.task, cfg=cfg).unwrapped
    env.reset()

    robot = env.scene["robot"]
    # The bulb entity is named differently across branches. #76 Step 1 renamed the scene's single
    # `bulb` to `fresh_bulb`, and the teleop bench this bag came from predates that rename, so a
    # replay has to accept either.
    names = [n for n in ("old_bulb", "fresh_bulb", "bulb") if n in env.scene.rigid_objects]
    if not names:
        raise SystemExit(f"no bulb in the scene; rigid objects are {list(env.scene.rigid_objects)}")
    bulb = env.scene[names[0]]
    print(f"BULB    using scene entity {names[0]!r}", flush=True)
    dev = env.device

    # Bag column -> sim joint index. A name the sim does not have is skipped and reported, because
    # a silently dropped joint would replay a different pose than the one recorded.
    sim_names = list(robot.joint_names)
    pairs = [(col, sim_names.index(n)) for col, n in enumerate(bag_joint_names) if n in sim_names]
    missing = [n for n in bag_joint_names if n not in sim_names]
    print(
        f"JOINTS mapped {len(pairs)}/{len(bag_joint_names)}" + (f", missing {missing}" if missing else ""), flush=True
    )
    cols = torch.tensor([c for c, _ in pairs], dtype=torch.long)
    ids = torch.tensor([i for _, i in pairs], dtype=torch.long, device=dev)

    # Clamp to what the bag actually holds. --steps past the end raised IndexError mid-replay and
    # threw away the run, which is a poor trade for a diagnostic that takes minutes to reach.
    available = int(joint_pos.shape[0]) - args_cli.start
    if args_cli.steps > available:
        print(f"STEPS   {args_cli.steps} requested, bag has {available} left from {args_cli.start}", flush=True)
        args_cli.steps = available

    ee_ids, _ = robot.find_bodies(meta["ee_body"])
    ee_id = ee_ids[0]
    action = torch.zeros(env.action_space.shape, device=dev)
    zeros6 = torch.zeros((1, 6), device=dev)

    def set_joints(step: int) -> None:
        q = torch.tensor(joint_pos[step][cols.numpy()], dtype=torch.float32, device=dev).unsqueeze(0)
        robot.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=ids)
        robot.set_joint_position_target(q, joint_ids=ids)

    # Seed the recorded pose, then settle so the articulation actually holds it.
    set_joints(args_cli.start)
    env.sim.step(render=False)
    robot.update(env.physics_dt)

    # bag world -> sim world, from the wrist both frames record.
    bag_p = torch.tensor(eef_pose[args_cli.start][:3], dtype=torch.float32, device=dev).unsqueeze(0)
    bag_q = torch.tensor(eef_pose[args_cli.start][3:], dtype=torch.float32, device=dev).unsqueeze(0)
    sim_p = robot.data.body_state_w[:, ee_id, :3]
    sim_q = robot.data.body_state_w[:, ee_id, 3:7]
    t_quat = quat_mul(sim_q, quat_inv(bag_q))
    t_pos = sim_p - quat_apply(t_quat, bag_p)

    def to_sim(p_bag, q_bag):
        p = torch.tensor(p_bag, dtype=torch.float32, device=dev).unsqueeze(0)
        q = torch.tensor(q_bag, dtype=torch.float32, device=dev).unsqueeze(0)
        return quat_apply(t_quat, p) + t_pos, quat_mul(t_quat, q)

    p0, q0 = to_sim(bulb_pos[args_cli.start], bulb_quat[args_cli.start])
    bulb.write_root_pose_to_sim(torch.cat([p0, q0], dim=-1))
    bulb.write_root_velocity_to_sim(zeros6)

    if args_cli.render:
        # Look at the seeded grasp from close range, from three sides. One picture settles whether
        # the bulb is buried in the palm, which no amount of force arithmetic has managed to.
        import imageio.v2 as imageio

        cam = env.scene["video_cam"]
        target = tuple(float(v) for v in p0[0])
        for tag, offset in (
            ("side", (0.30, -0.02, 0.04)),
            ("front", (0.02, -0.30, 0.04)),
            ("top", (0.02, -0.04, 0.30)),
        ):
            eye = tuple(t + o for t, o in zip(target, offset))
            cam.set_world_poses_from_view(
                torch.tensor(eye, dtype=torch.float32, device=dev).expand(env.num_envs, 3),
                torch.tensor(target, dtype=torch.float32, device=dev).expand(env.num_envs, 3),
            )
            for _ in range(3):
                env.sim.render()
            cam.update(0.0)
            frame = cam.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
            path = args_cli.render.rsplit(".", 1)[0] + f"_{tag}.png"
            imageio.imwrite(path, frame)
            print(f"RENDER  wrote {path}", flush=True)

    sensors = [n for n in ("hand_contact", "left_hand_contact") if n in env.scene.sensors]
    weight = 0.035 * 9.81
    seeded_z = float(p0[0, 2])
    print(f"SEEDED  bag step {args_cli.start}, bulb at sim z={seeded_z:.4f} m", flush=True)
    print("  step   sim z    recorded z   dz(sim-rec)   |v| mm/s   force N", flush=True)

    for k in range(args_cli.steps):
        step = args_cli.start + k
        set_joints(step)
        if not args_cli.free_bulb:
            pk, qk = to_sim(bulb_pos[step], bulb_quat[step])
            bulb.write_root_pose_to_sim(torch.cat([pk, qk], dim=-1))
            bulb.write_root_velocity_to_sim(zeros6)
        env.step(action)
        if k % 8 == 0 or k == args_cli.steps - 1:
            rec, _ = to_sim(bulb_pos[step], bulb_quat[step])
            sim_z = float(bulb.data.root_pos_w[0, 2])
            speed = float(torch.norm(bulb.data.root_lin_vel_w[0])) * 1000.0
            force = max(
                float(torch.norm(_obs.object_contact_forces(env.scene.sensors[s]), dim=-1).max()) for s in sensors
            )
            print(
                f"  {step:5d}  {sim_z:.4f}   {float(rec[0, 2]):.4f}     "
                f"{(sim_z - float(rec[0, 2])) * 1000:+7.1f} mm  {speed:8.1f}  {force:8.2f}",
                flush=True,
            )

    final_z = float(bulb.data.root_pos_w[0, 2])
    dropped = (seeded_z - final_z) * 1000.0
    force = max(float(torch.norm(_obs.object_contact_forces(env.scene.sensors[s]), dim=-1).max()) for s in sensors)
    print(
        f"\nVERDICT bulb {'FELL' if dropped > 50.0 else 'STAYED'} -- moved {dropped:+.1f} mm down "
        f"over {args_cli.steps} steps, final contact {force:.2f} N = {force / weight:.0f}x its weight",
        flush=True,
    )
    env.close()
    return 0


if __name__ == "__main__":
    import os
    import sys

    exit_code = 1
    try:
        exit_code = main()
    except BaseException:
        import traceback

        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        if exit_code:
            os._exit(exit_code)
        simulation_app.close()
