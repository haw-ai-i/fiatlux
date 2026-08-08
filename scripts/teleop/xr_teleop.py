# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""XR (Meta Quest / Pico via CloudXR) hand-tracking teleoperation for fiatlux tasks (issue #51).

Renders the fiatlux scene to an XR headset through NVIDIA CloudXR and drives the robot's right-hand
end-effector (absolute IK) + grip from **hand tracking** -- the wrist pose maps to the EE target and
the thumb-index pinch maps to open/close. This is Isaac Lab's *native* OpenXR teleop path
(``isaaclab.devices.OpenXRDevice`` + retargeters), which renders the sim itself; CloudXR only streams
it. The target env must define a ``handtracking`` device in its ``teleop_devices`` (see
``insert_teleop_env_cfg.py``).

This is a thin, fiatlux-aware adaptation of Isaac Lab's
``scripts/environments/teleoperation/teleop_se3_agent.py`` -- the only substantive addition is
importing ``fiatlux_task.tasks`` so the ``FIATLUX-*`` envs are registered.

Prerequisites (see ``vr_teleop/vr_teleop_setup.md``):
  * The CloudXR runtime is running (e.g. ``python -m isaacteleop.cloudxr --host-client`` with
    ``NV_CXR_ENDPOINT_IP`` set for remote/Tailscale) and its env is sourced so ``XR_RUNTIME_JSON``
    points at CloudXR.

Run::

    conda activate env_isaaclab
    source ~/.cloudxr/run/cloudxr.env            # sets XR_RUNTIME_JSON -> CloudXR
    export PYTHONPATH=$PWD/source/fiatlux_task
    python scripts/xr_teleop.py --task FIATLUX-Insert-Teleop-v0 --teleop_device handtracking

Then, in the Isaac Sim UI, open the **AR** panel (Output Plugin: OpenXR, Runtime: System OpenXR
Runtime) and click **Start AR**; connect the headset's CloudXR web client. Use the headset UI
**Play/Stop/Reset** (or keyboard ``R``) to gate teleoperation.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="XR hand-tracking teleoperation for fiatlux environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="handtracking",
    help="Teleop device key defined under the env's 'teleop_devices' (handtracking | keyboard | spacemouse).",
)
parser.add_argument("--task", type=str, default="FIATLUX-Insert-Teleop-v0", help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
parser.add_argument(
    "--hand",
    type=str,
    default="inspire",
    choices=["inspire", "dex3"],
    help="G1 hand variant: inspire (default 5-finger) or dex3 (Unitree 3-finger).",
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio (only needed for dex/Pink-IK retargeters; not required for the Insert task).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher so IsaacLab's build is used, not Isaac Sim's.
    import pinocchio  # noqa: F401
_dev = args_cli.teleop_device.lower()
if _dev == "handtracking" or _dev.startswith("controller"):
    # All headset devices (hand tracking, controller, controller_rel) need the XR experience
    # (stereo render + OpenXR); auto-enable it so the AR panel appears and cameras are stripped.
    app_launcher_args["xr"] = True

app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

"""Rest everything follows."""

import logging

import gymnasium as gym
import torch

from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.envs import ManagerBasedRLEnvCfg

from isaaclab_tasks.utils import parse_env_cfg

import fiatlux_task.tasks  # noqa: F401  registers FIATLUX-* Gym envs
import fiatlux_teleop  # noqa: F401,E402  -- registers the FIATLUX-*-Teleop gym ids

logger = logging.getLogger(__name__)


def _strip_camera_configs(env_cfg):
    """Remove ``CameraCfg`` scene sensors and their observation terms for XR rendering.

    Isaac Lab's own ``remove_camera_configs`` deletes the policy obs term using the *scene camera*
    name (assuming the obs term is named the same as the camera). Our env names them differently
    (scene ``wrist_camera`` vs obs ``wrist_rgb``), so that helper raises ``AttributeError``. This
    version deletes the obs term by its *actual* name (matched via the term's ``SceneEntityCfg``
    param pointing at the removed camera).
    """
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.sensors import CameraCfg

    cam_names = []
    for attr_name in list(dir(env_cfg.scene)):
        if isinstance(getattr(env_cfg.scene, attr_name), CameraCfg):
            cam_names.append(attr_name)
            delattr(env_cfg.scene, attr_name)
            logger.info(f"XR: removed camera sensor '{attr_name}'")

    policy = getattr(env_cfg.observations, "policy", None)
    if policy is not None:
        for obs_name in list(dir(policy)):
            obsterm = getattr(policy, obs_name)
            params = getattr(obsterm, "params", None)
            if params and any(isinstance(v, SceneEntityCfg) and v.name in cam_names for v in params.values()):
                delattr(policy, obs_name)
                logger.info(f"XR: removed camera observation term '{obs_name}'")
    return env_cfg


def main() -> None:
    """Create the env, wire the teleop device + callbacks, and run the sim loop."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    if args_cli.hand.lower() == "dex3":
        # swap the G1 to Dex3 hands (robot + observations + grip actions + Dex3 grasp presets)
        from fiatlux_teleop.insert_teleop_env_cfg import apply_dex3_hands

        apply_dex3_hands(env_cfg)
        print("Hand variant: Dex3 (Unitree 3-finger).")
    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise ValueError(
            "Teleoperation is only supported for ManagerBasedRLEnv environments. "
            f"Received environment config type: {type(env_cfg).__name__}"
        )

    # operator-paced: no auto time-out reset (the Insert-Teleop env already disables all terminations)
    env_cfg.terminations.time_out = None

    if args_cli.xr:
        # XR renders stereo views; drop camera sensors (perf) and use DLSS for the stereo render.
        env_cfg = _strip_camera_configs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # teleoperation flow flags
    should_reset = False
    # For hand tracking, wait for the headset UI "Play" (START) before applying commands.
    teleoperation_active = not args_cli.xr

    def reset_env() -> None:
        nonlocal should_reset
        should_reset = True
        print("Reset triggered - environment will reset on next step")

    def start_teleop() -> None:
        nonlocal teleoperation_active
        teleoperation_active = True
        print("Teleoperation activated")

    def stop_teleop() -> None:
        nonlocal teleoperation_active
        teleoperation_active = False
        print("Teleoperation deactivated")

    callbacks = {"R": reset_env, "START": start_teleop, "STOP": stop_teleop, "RESET": reset_env}

    # build the teleop device from the env config (handtracking OpenXRDevice + retargeters)
    if not (hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices):
        logger.error(
            f"No teleop device '{args_cli.teleop_device}' in env config for task '{args_cli.task}'. "
            "Add it to the env's teleop_devices."
        )
        env.close()
        simulation_app.close()
        return
    teleop_interface = create_teleop_device(args_cli.teleop_device, env_cfg.teleop_devices.devices, callbacks)
    print(f"Using teleop device: {teleop_interface}")

    def _reset_teleop() -> None:
        # Full reset. OpenXRDevice.reset() does NOT reset its retargeters, so the accumulated EE
        # target (Se3RelControllerRetargeter._pos) and its motion references survive a reset. After
        # env.reset() puts the arm at its default pose, a stale target makes the IK drive the arm on
        # its own ("moves when the controller isn't moving / doesn't fully reset"). Reset both.
        teleop_interface.reset()
        for _rt in getattr(teleop_interface, "_retargeters", None) or []:
            if hasattr(_rt, "reset"):
                _rt.reset()

    env.reset()
    _reset_teleop()
    print(
        "Teleop ready. In the Isaac Sim UI: open the AR panel (OpenXR / System OpenXR Runtime) and "
        "click 'Start AR', then connect the headset. Press 'R' to reset."
    )

    # Passive insertion monitor -- PHYSICS does the inserting. The human lowers the bulb into the
    # socket themselves: the socket is a concave triangle-mesh collider (a real hole) and the bulb's
    # convex decomposition is shrink-wrapped so the screw base clears the bore. This block NEVER moves
    # the bulb (no auto-snap / no weld) -- it only detects and reports when the bulb is seated.
    # "Seated" = the bulb root is (a) centered on the socket axis and (b) at the seated height (base
    # dropped into the bore, not perched on the rim), and (c) settled (low speed). Measured headless
    # (the slightly-smaller 0.006 bulb): a centered drop settles at dz ~= +0.004 relative to the socket
    # root, dxy ~= 0.000; a base resting on the flat socket top (not in the hole) sits higher, so the
    # upper dz bound rejects it.
    _SEAT_DZ_LO = -0.030  # bulb root vs socket root: lower bound (allow a touch deeper)
    _SEAT_DZ_HI = 0.028   # ... upper bound -- above this the base is perched on the rim, not in the bore
    _SEAT_DXY = 0.020     # centered in the bore (m)
    _SEAT_VMAX = 0.08     # settled, not mid-flight (m/s)

    _seat_state = {"seated": False, "dbg": 0}
    try:
        _bulb = env.scene["bulb"]
        _socket = env.scene["socket"]
        _seat_ok = True
    except Exception as _e:  # noqa: BLE001
        logger.warning(f"Insertion monitor disabled (bulb/socket not found): {_e}")
        _seat_ok = False

    def _update_bulb_seat() -> None:
        """Report-only. Never moves the bulb -- the human inserts it via physics."""
        if not _seat_ok:
            return
        sroot = _socket.data.root_pos_w[0]
        broot = _bulb.data.root_pos_w[0]
        dz = float(broot[2] - sroot[2])
        dxy = float(torch.norm(broot[:2] - sroot[:2]))
        spd = float(torch.norm(_bulb.data.root_lin_vel_w[0]))
        inserted = (_SEAT_DZ_LO < dz < _SEAT_DZ_HI) and (dxy < _SEAT_DXY) and (spd < _SEAT_VMAX)
        if inserted and not _seat_state["seated"]:
            _seat_state["seated"] = True
            print("Bulb inserted into socket.", flush=True)
        elif _seat_state["seated"] and (dz > _SEAT_DZ_HI + 0.03 or dxy > _SEAT_DXY + 0.02):
            _seat_state["seated"] = False  # pulled back out
            print("Bulb removed from socket.", flush=True)
        _seat_state["dbg"] += 1
        if _seat_state["dbg"] % 30 == 0:  # TEMP: watch insertion progress (dz sinks negative as it goes in)
            print(f"[SEAT] dz={dz:+.3f} dxy={dxy:.3f} spd={spd:.3f} seated={int(_seat_state['seated'])}", flush=True)

    while simulation_app.is_running():
        try:
            with torch.inference_mode():
                action = teleop_interface.advance()
                if action is not None and teleoperation_active:
                    actions = action.repeat(env.num_envs, 1)
                    env.step(actions)
                    _update_bulb_seat()
                else:
                    # keep rendering (and streaming to the headset) while inactive / awaiting tracking
                    env.sim.render()

                if should_reset:
                    should_reset = False
                    env.reset()
                    _reset_teleop()
                    _seat_state["seated"] = False  # insertion state clears on reset
                    print("Environment reset complete")
        except KeyboardInterrupt:
            break
        except Exception as e:  # noqa: BLE001
            # Clicking "Start AR" rebuilds the render stage and can transiently invalidate the PhysX
            # tensor view; an env.step() in that window raises ("Simulation view ... invalidated").
            # Don't hard-crash -- keep the app alive and rendering so the headset stream continues;
            # stepping resumes once the view re-validates.
            logger.warning(f"XR: step skipped while sim view recovers: {e}")
            try:
                env.sim.render()
            except Exception:  # noqa: BLE001
                pass

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
