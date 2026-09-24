# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contact force against bulb-to-palm distance (issue #92).

Two modes. By default the fingers stay in their reset pose, which is OPEN, so this characterises
the contact between the bulb and a stationary open hand. With ``--grasp`` the fingers close on the
bulb first and OPEN to release it, which is the scenario #92 actually reports: a hand that opens
and does not drop the bulb. Read the ``curl`` column in that mode -- it is the measured finger
angle, and a bulb that "stayed" with the fingers still curled proves nothing.


``diagnose_stuck_bulb.py`` reports 181-223x the bulb's weight in a hand whose fingers are open. It
also parks the bulb at the palm body's ORIGIN, and the bulb's radius is about 39 mm, so the bulb
probably encloses the palm geometry: those forces may be the solver depenetrating an overlap the
script created rather than anything about the hand.

This decides it. Hold the bulb at a series of distances from the palm and read the hand-bulb
contact force at each.

* Force decays sharply as the offset grows -> the high readings are interpenetration. The scripted
  stuck-bulb result is an artifact of its own placement, and the recording is the only evidence.
* Force stays at tens of newtons with the bulb well clear of the hand -> the contact pathology is
  real, and it belongs in the bulb or hand collision setup.

The bulb is held at each offset rather than dropped. The question is how hard the solver pushes at
a given separation, which is what holding it measures.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Sweep hand-bulb contact force against distance (issue #92).")
parser.add_argument("--task", type=str, default="FIATLUX-Replace-v0")
parser.add_argument("--variant", type=str, default="dex3", choices=["dex3", "inspire"])
parser.add_argument("--hand", type=str, default="right", choices=["left", "right"])
parser.add_argument(
    "--offsets",
    type=float,
    nargs="+",
    default=[0.0, 0.01, 0.02, 0.04, 0.06, 0.10],
    help="Metres from the palm body origin. 0 is where diagnose_stuck_bulb.py parks it.",
)
parser.add_argument("--settle", type=int, default=20, help="Steps to hold at each offset before reading.")
parser.add_argument(
    "--quiet",
    type=int,
    default=15,
    help="Steps to run after restoring the scene, before force is read. The restore snaps the arm "
    "back to its captured pose, and that transient would otherwise be counted as held force.",
)
parser.add_argument("--friction", type=float, default=None, help="Override the bulb's static+dynamic friction.")
parser.add_argument("--contact-offset", type=float, default=None, help="Override the bulb's contact_offset (m).")
parser.add_argument("--rest-offset", type=float, default=None, help="Override the bulb's rest_offset (m).")
parser.add_argument("--solver-iters", type=int, default=None, help="Override the bulb's position iteration count.")
parser.add_argument(
    "--mass",
    type=float,
    default=None,
    help="Override the bulb's mass (kg). The default 0.035 kg against a ~35 kg articulation is a "
    "mass ratio near 1000:1, which PhysX handles poorly. Every force here is reported against the "
    "OVERRIDDEN weight, so the ratios stay comparable across masses.",
)
parser.add_argument(
    "--release",
    type=int,
    default=90,
    help="Steps to watch after letting go at each offset. This is the half that is not confounded: "
    "holding the bulb pose-writes it, and pose-writing a body in contact is itself the #77 "
    "pathology. Nothing writes the bulb during these steps.",
)
parser.add_argument(
    "--placement",
    choices=["seat", "ray"],
    default="seat",
    help="Where the bulb starts. 'seat' uses the task's own calibrated palm seat "
    "(mdp.nav_terms.settle_carried_payload_live) and sweeps the offset along the palm NORMAL, "
    "lifting the bulb off the surface it rests on. 'ray' is the original: offset along a "
    "horizontal ray from the robot root, anchored on the palm BODY ORIGIN -- which issue #105 "
    "measured as several cm off the visible mesh, so the bulb starts 1-2 cm inside the collider "
    "and the solver throws it out. Kept only to reproduce the superseded numbers.",
)
parser.add_argument(
    "--grasp",
    action="store_true",
    help="Close the fingers on the bulb before letting go, and OPEN them to release. Without this "
    "the fingers stay in their reset pose, which is open, so the sweep measures a bulb held near a "
    "stationary open hand -- useful, but not the reported bug. #92 is specifically that OPENING "
    "the fingers does not drop the bulb, and only this flag tests that.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything else follows."""

import fiatlux_task.tasks  # noqa: F401, E402  -- registers the FIATLUX Gym environments
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from fiatlux_task.robots.g1 import (  # noqa: E402
    G1_DEX3_HAND_GRASP,
    G1_DEX3_LEFT_HAND_GRASP,
    G1_DEX3_PALM_BODIES,
    G1_HAND_GRASP,
    G1_LEFT_HAND_GRASP,
    G1_PALM_BODIES,
    G1_PALM_LOCAL_AXES,
    swap_robot_variant,
)
from fiatlux_task.tasks.manager_based.fiatlux_task.mdp import observations as _obs  # noqa: E402
from fiatlux_task.tasks.manager_based.fiatlux_task.mdp.nav_terms import settle_carried_payload_live  # noqa: E402
from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import BULB_MASS_KG, set_layout_seed  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_inv, quat_mul  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

_LEFT, _RIGHT = 0, 1


def grasp_targets(variant: str, hand: str) -> dict[str, float]:
    """The closed-hand joint preset for this variant and side.

    Kept identical to ``diagnose_stuck_bulb.py``. The Dex3 presets are MIRRORED, not shared: right
    fingers curl toward + and left toward -, so applying the right-hand preset to the left hand
    opens it instead of closing it.
    """
    if variant == "dex3":
        return dict(G1_DEX3_LEFT_HAND_GRASP if hand == "left" else G1_DEX3_HAND_GRASP)
    if hand == "left":
        return dict(G1_LEFT_HAND_GRASP)
    return dict(G1_HAND_GRASP)


def main() -> int:
    set_layout_seed(0)
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    if args_cli.variant != "inspire":
        swap_robot_variant(cfg, args_cli.variant)
    # Move the furniture aside rather than deleting it -- event and observation terms reference
    # these entities by name. Pin the root: a free base under zero actions sags, and a moving palm
    # would change the offset being swept.
    for entity in ("table", "bin", "ladder", "socket", "pendant", "fixture"):
        item = getattr(cfg.scene, entity, None)
        if item is not None and getattr(item, "init_state", None) is not None:
            pos = item.init_state.pos
            item.init_state.pos = (pos[0] + 20.0, pos[1] + 20.0, pos[2])
    cfg.scene.robot.spawn.articulation_props.fix_root_link = True

    # Physics overrides, so each candidate cause in #92 can be tested against the same measurement
    # rather than argued about. The bulb's authored material is staticFriction 1.2 / dynamicFriction
    # 1.0 (LightBulb_collision.usda), which #77 task 6 named as a candidate and never tested.
    bulb_cfg = cfg.scene.old_bulb if getattr(cfg.scene, "old_bulb", None) is not None else cfg.scene.fresh_bulb
    tweaks = []
    if args_cli.friction is not None:
        bulb_cfg.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=args_cli.friction, dynamic_friction=args_cli.friction, restitution=0.0
        )
        bulb_cfg.spawn.physics_material_prim_path = "/PhysicsMaterials/HighFriction"
        tweaks.append(f"friction={args_cli.friction}")
    if args_cli.contact_offset is not None:
        bulb_cfg.spawn.collision_props.contact_offset = args_cli.contact_offset
        tweaks.append(f"contact_offset={args_cli.contact_offset}")
    if args_cli.rest_offset is not None:
        bulb_cfg.spawn.collision_props.rest_offset = args_cli.rest_offset
        tweaks.append(f"rest_offset={args_cli.rest_offset}")
    if args_cli.solver_iters is not None:
        bulb_cfg.spawn.rigid_props.solver_position_iteration_count = args_cli.solver_iters
        tweaks.append(f"solver_iters={args_cli.solver_iters}")
    if args_cli.mass is not None:
        bulb_cfg.spawn.mass_props.mass = args_cli.mass
        tweaks.append(f"mass={args_cli.mass}")

    env = gym.make(args_cli.task, cfg=cfg).unwrapped
    env.reset()

    robot = env.scene["robot"]
    bulb_name = "old_bulb" if "old_bulb" in env.scene.rigid_objects else "fresh_bulb"
    bulb = env.scene[bulb_name]
    bulb_cfg_entity = SceneEntityCfg(bulb_name)
    palms = G1_DEX3_PALM_BODIES if args_cli.variant == "dex3" else G1_PALM_BODIES
    palm_idx = robot.find_bodies(palms[_LEFT if args_cli.hand == "left" else _RIGHT])[0][0]
    sensors = [n for n in ("hand_contact", "left_hand_contact") if n in env.scene.sensors]

    action = torch.zeros(env.action_space.shape, device=env.device)
    zeros6 = torch.zeros((1, 6), device=env.device)
    # Against the OVERRIDDEN mass, not the default. Every ratio below and the STUCK threshold are
    # multiples of the bulb's own weight, so holding this at 0.343 N while --mass changes the bulb
    # would make a heavier bulb look better purely by arithmetic.
    bulb_mass = args_cli.mass if args_cli.mass is not None else BULB_MASS_KG
    weight = bulb_mass * 9.81

    # Offset AWAY FROM THE ROBOT, not along world up. Offsetting up from the palm walks the bulb
    # into the forearm, so a reading at 100 mm was contact with the arm rather than clearance:
    # the same commanded position read 0.00 N in one run and 236 N in another.
    if args_cli.placement == "seat":
        # Use the task's OWN calibrated seat rather than a placement invented here. Issue #105
        # measured what the invented one costs: the palm body origin sits several cm off the
        # visible mesh, so a bulb anchored to it starts 1-2 cm inside the collider and the solver
        # throws it out at 3-6 m/s -- which is what the 65-2366 N "grip" readings in #105, and the
        # 100-285 N readings this script used to report, actually were.
        #
        # Calling the event term rather than copying its arithmetic. It is gated on
        # episode_length_buf == 1, the step at which the arm has been simulated into its target
        # pose, so the counter is set to that value here for the same reason.
        if args_cli.hand != "right":
            raise SystemExit(
                "--placement seat is right-hand only: G1_FINGER_BASE_BODIES_BY_VARIANT and "
                "G1_PALM_BODY_BY_VARIANT both name right-hand bodies. Use --hand right, or "
                "--placement ray to sweep the left hand with the superseded placement."
            )
        env.episode_length_buf[:] = 1
        settle_carried_payload_live(
            env, torch.arange(env.num_envs, device=env.device), bulb_cfg_entity, args_cli.variant
        )
        env.sim.step(render=False)
        robot.update(env.physics_dt)
        bulb.update(env.physics_dt)
        # Offsets lift the bulb off the surface it rests on, along the palm's outward normal.
        (_seat_n_idx, _seat_n_sign), _, _ = G1_PALM_LOCAL_AXES[args_cli.variant]
        away = matrix_from_quat(robot.data.body_quat_w[0:1, palm_idx])[0, :, _seat_n_idx] * _seat_n_sign
        # Store the seat IN THE PALM FRAME, not as a fixed world point. The arm is commanded to
        # hold still but does not hold perfectly, and a seat pinned to world coordinates leaves
        # the bulb hanging in space as soon as the palm drifts -- which reads as 0.00 N at every
        # offset, the bulb having never been near the hand at all.
        palm_pos0 = robot.data.body_pos_w[0:1, palm_idx]
        palm_quat0 = robot.data.body_quat_w[0:1, palm_idx]
        seat_pos = quat_apply(quat_inv(palm_quat0), bulb.data.root_pos_w[0:1] - palm_pos0)
        seat_quat = quat_mul(quat_inv(palm_quat0), bulb.data.root_quat_w[0:1])
    else:
        # Offset AWAY FROM THE ROBOT, not along world up. Offsetting up from the palm walks the
        # bulb into the forearm, so a reading at 100 mm was contact with the arm rather than
        # clearance: the same commanded position read 0.00 N in one run and 236 N in another.
        root = robot.data.root_pos_w[0]
        palm0 = robot.data.body_pos_w[0, palm_idx, :]
        away = palm0 - root
        away[2] = 0.0  # horizontal: straight up or down still meets the arm or the floor
        away = away / torch.norm(away).clamp(min=1e-6)
        seat_pos = seat_quat = None
        _seat_n_idx, _seat_n_sign = 0, 1.0

    def _target_pose(offset: float):
        """Where the bulb is held at this offset, and at what orientation.

        Seat mode lifts the calibrated seat along the palm normal and keeps the seat's own
        orientation -- the bulb lies across the palm, which is the pose the hand closes on. Ray
        mode reproduces the superseded placement.
        """
        palm_p = robot.data.body_pos_w[:, palm_idx, :]
        if seat_pos is None:
            return palm_p + offset * away, init_bulb_quat
        # Rebuild from the LIVE palm each step, so the seat rides the hand.
        palm_q = robot.data.body_quat_w[:, palm_idx]
        normal = matrix_from_quat(palm_q)[:, :, _seat_n_idx] * _seat_n_sign
        return palm_p + quat_apply(palm_q, seat_pos) + offset * normal, quat_mul(palm_q, seat_quat)

    # Hold the arm where it starts. Only the root is pinned, so under zero actions the arm sags,
    # and an arm that drifts between offsets changes the geometry being swept.
    arm_target = robot.data.joint_pos.clone()

    # Captured once and restored before EVERY offset. Without the restore the six offsets are one
    # continuous simulation: each ends with a 90-step release where the bulb is free, often ejected
    # at over 3 m/s and sometimes lodging at hundreds of newtons, and the next offset is then
    # measured on the scene that release just disturbed. The contamination accumulates down the
    # sweep -- at 0.035 kg the spread in held force grew from 183 N at the first offset to 948 N at
    # the last -- so every comparison between configurations was confounded with how violently the
    # previous offset happened to end.
    init_joint_pos = robot.data.joint_pos.clone()
    init_joint_vel = robot.data.joint_vel.clone()
    # The bulb's attitude too. A bulb that tumbled during the previous release re-enters the next
    # offset at whatever orientation it stopped in, and the contact geometry depends on it.
    init_bulb_quat = bulb.data.root_quat_w.clone()

    # Finger control. Without --grasp the fingers stay wherever the reset pose left them, which is
    # open, so the sweep characterises contact with a stationary open hand. #92 is about OPENING
    # the fingers failing to drop the bulb, and that needs the hand closed first.
    finger_ids: list[int] = []
    closed_cmd = opened_cmd = None
    if args_cli.grasp:
        targets = grasp_targets(args_cli.variant, args_cli.hand)
        finger_ids = [robot.find_joints(name)[0][0] for name in targets]
        closed_cmd = torch.tensor([list(targets.values())], device=env.device)
        opened_cmd = torch.zeros_like(closed_cmd)

    def drive(fingers) -> None:
        """Hold the arm still, and command the fingers if --grasp is on.

        The arm target covers EVERY joint, so the finger command has to come after it or it is
        overwritten.
        """
        robot.set_joint_position_target(arm_target)
        if fingers is not None:
            robot.set_joint_position_target(fingers, joint_ids=finger_ids)

    print(
        f"SETUP variant={args_cli.variant} hand={args_cli.hand} bulb weight={weight:.3f} N"
        + (f"  OVERRIDES {', '.join(tweaks)}" if tweaks else "  (stock physics)"),
        flush=True,
    )
    print(f"      offset direction (world, away from the root): {[round(float(v), 3) for v in away]}", flush=True)
    print("  offset   held force   freed force    moved  dir   to palm      speed      outcome", flush=True)

    def peak_force() -> float:
        return max(float(torch.norm(_obs.object_contact_forces(env.scene.sensors[s]), dim=-1).max()) for s in sensors)

    results = []
    for offset in args_cli.offsets:
        # Restore, so this offset is independent of the ones before it.
        robot.write_joint_state_to_sim(init_joint_pos, init_joint_vel)
        robot.set_joint_position_target(arm_target)
        bulb.write_root_velocity_to_sim(zeros6)
        for _ in range(args_cli.quiet):
            drive(closed_cmd)
            target, quat = _target_pose(offset)
            bulb.write_root_pose_to_sim(torch.cat([target, quat], dim=-1))
            bulb.write_root_velocity_to_sim(zeros6)
            env.step(action)

        held = 0.0
        for _ in range(args_cli.settle):
            # Hold the bulb at the offset every step. This half is confounded -- see the caveat at
            # the end -- and exists to say whether the bulb is in contact at this separation.
            drive(closed_cmd)
            target, quat = _target_pose(offset)
            bulb.write_root_pose_to_sim(torch.cat([target, quat], dim=-1))
            bulb.write_root_velocity_to_sim(zeros6)
            env.step(action)
            held = max(held, peak_force())

        # Now let go. Nothing writes the bulb from here, so what happens is physics alone. A bulb
        # that is merely touching an open hand falls. A bulb the solver is fighting does not.
        start_z = float(bulb.data.root_pos_w[0, 2])
        freed = 0.0
        for _ in range(args_cli.release):
            drive(opened_cmd)
            env.step(action)
            freed = max(freed, peak_force())
        fell_mm = (start_z - float(bulb.data.root_pos_w[0, 2])) * 1000.0
        speed = float(torch.norm(bulb.data.root_lin_vel_w[0])) * 1000.0
        # Distance to the palm at the end. "Stayed" near the palm is the hand holding it; "stayed"
        # far from the palm is the bulb caught on something else, and those are different bugs.
        palm_now = robot.data.body_pos_w[0, palm_idx, :]
        dist_mm = float(torch.norm(bulb.data.root_pos_w[0] - palm_now)) * 1000.0

        # Classify by whether the bulb LEFT THE HAND, not by which way it went. An earlier version
        # tested downward motion only, so a bulb ejected upward -- which is a release, however
        # ugly -- was labelled "stayed". That hid the friction=0 result, whose released force is
        # 0.18-1.53 N, essentially no contact at all.
        #
        # Two facts decide it: is the bulb still near the hand, and is it still being pushed. The
        # hand's contact reaches about 40-60 mm from the palm origin, so 150 mm is clear of it.
        results.append((offset, held, freed, fell_mm, dist_mm))
        gone = dist_mm > 150.0
        pushed = freed > 5.0 * weight
        if gone and not pushed:
            outcome = "released"
        elif not gone and pushed:
            outcome = "STUCK"
        elif gone and pushed:
            outcome = "left, still pushed"
        else:
            outcome = "near hand, no force"
        went = "down" if fell_mm > 5.0 else ("up" if fell_mm < -5.0 else "--")
        # The MEASURED finger angle, not the commanded one. "The bulb did not fall" has a trivial
        # explanation -- the fingers never opened -- and nothing else in this row rules it out.
        curl = f"  curl {float(robot.data.joint_pos[0, finger_ids].abs().max()):.3f}" if finger_ids else ""
        print(
            f"  {offset * 1000:5.0f} mm  {held:9.2f} N  {freed:9.2f} N  {fell_mm:+8.1f} mm "
            f"{went:4s} {dist_mm:7.1f} mm  {speed:8.1f} mm/s  {outcome}{curl}",
            flush=True,
        )

    # Report the SHAPE of the curve, not its endpoints. An earlier version compared the first and
    # last readings and printed "DECAYS" whenever the last one was small -- but the largest offset
    # is where the bulb is out of reach of the hand entirely, so its reading is zero for a reason
    # that has nothing to do with how the solver behaves in contact. That comparison called a
    # curve holding 100-330 N through 60 mm a decay.
    touching = [(off, held) for off, held, _, _, _ in results if held > 1.0]
    print("\nVERDICT", flush=True)
    if not touching:
        print("  NO CONTACT at any offset. Nothing was measured; move the bulb closer.", flush=True)
    else:
        widest, force_at_widest = touching[-1]
        print(
            f"  contact at {len(touching)}/{len(results)} offsets, out to {widest * 1000:.0f} mm, "
            f"where it reads {force_at_widest:.2f} N = {force_at_widest / weight:.0f}x the bulb's weight",
            flush=True,
        )
        gentlest = min(f for _, f in touching)
        if gentlest < 5.0 * weight:
            print(
                f"  There IS a gentle regime: the lightest contact reads {gentlest:.2f} N, within a "
                "few times the bulb's weight, so contact force tracks separation as it should.",
                flush=True,
            )
        else:
            print(
                f"  There is NO gentle regime while held: even the lightest contact reads "
                f"{gentlest:.2f} N = {gentlest / weight:.0f}x the bulb's weight.",
                flush=True,
            )

    # The released column is the one that answers the issue, because nothing writes the bulb during
    # it. Held readings are confounded -- see below -- but a bulb that will not fall when nothing
    # is touching it cannot be blamed on the measurement.
    in_contact = [(off, freed, fell, dist) for off, held, freed, fell, dist in results if held > 1.0]
    # Stuck means near the hand AND still being pushed. Not merely 'did not move downward'.
    stayed = [
        (off, freed, fell, dist) for off, freed, fell, dist in in_contact if dist <= 150.0 and freed > 5.0 * weight
    ]
    print("\n  ON RELEASE, at the offsets that were in contact:", flush=True)
    if not in_contact:
        print("    nothing was in contact, so there was nothing to release.", flush=True)
    elif not stayed:
        print(
            f"    all {len(in_contact)} left the hand once released. The hand does not hold a bulb it is\n"
            "    gripping, so the large held readings above are the pose-write confound and not a\n"
            "    property of the hand. The scripted stuck-bulb result is an artifact of placement.",
            flush=True,
        )
    else:
        worst_off, worst_force, worst_fell, worst_dist = max(stayed, key=lambda r: r[1])
        print(
            f"    {len(stayed)}/{len(in_contact)} stayed STUCK -- near the hand and still pushed. "
            f"Worst at {worst_off * 1000:.0f} mm: "
            f"moved {worst_fell:+.1f} mm with {worst_force:.2f} N = {worst_force / weight:.0f}x its\n"
            f"    weight, and nothing writing its pose. It ended {worst_dist:.0f} mm from the palm.",
            flush=True,
        )
        near_palm = [r for r in stayed if r[3] < 120.0]
        print(
            f"    {len(near_palm)}/{len(stayed)} of those ended within 120 mm of the palm, so the "
            "hand is what holds them."
            if near_palm
            else "    None ended near the palm, so whatever holds them is not the hand.",
            flush=True,
        )

    print(
        "\n  CAVEAT the HELD column pose-writes the bulb every step, and pose-writing a body already\n"
        "  in contact is itself the pathology #77 measured at 1067 N on the socket. Read the RELEASED\n"
        "  column for anything about the hand.",
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
