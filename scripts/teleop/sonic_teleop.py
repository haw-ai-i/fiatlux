"""Whole-body loco-manipulation teleop for any FIATLUX-*-Teleop task.

Runs a real teleop gym env (``--task`` -- Insert / Carry / LadderGallery / ...), FREES the base, and
drives legs+waist with the pre-trained NVIDIA SONIC policy so the operator can walk the robot around
the scene and manipulate -- true whole-body teleop, on any task.

Two input modes (``--input``):
  * vr        -- Pico controllers over CloudXR: LEFT stick walks, the env's own bimanual arm teleop
                 (controller pose -> IK, trigger -> grip) drives the arms.
  * keyboard  -- desktop, no headset: arrow keys walk; TAB picks the active arm; W/S A/D Q/E move it,
                 U/O I/K J/L rotate the wrist, G grips -- both arms + wrist rotation, i.e. VR parity.

  * arms  = the env's action manager (differential-IK EE pose + binary grip)
  * legs  = SONIC (balance + walk)
The env is retimed to SONIC's 200 Hz / 50 Hz. VR appends a walk retargeter to the env's teleop device
so ``advance()`` returns ``[arm_action..., vx,vy,wz,stop,lean]``; keyboard builds the same
``[arm_action..., walk]`` from key state. Either way the loop hands the arm part to ``env.step`` and
the walk part to SONIC.
"""

import argparse
import contextlib
import math
import os
import random
import re

from isaaclab.app import AppLauncher

_DEFAULT_POLICY_DIR = os.path.expanduser(
    "~/robotica_project/GR00T-WholeBodyControl/gr00t_wbc/sim2mujoco/resources/robots/g1/policy"
)
_POLICY_DIR = os.environ.get("SONIC_POLICY_DIR", _DEFAULT_POLICY_DIR)

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="FIATLUX-Insert-Teleop-v0")
parser.add_argument(
    "--input",
    choices=["vr", "keyboard"],
    default="vr",
    help="vr = Pico controllers over CloudXR; keyboard = desktop keys (no headset)",
)
parser.add_argument(
    "--layout_seed",
    default="random",
    help="room layout: an integer picks a specific reproducible room; 'random' "
    "(default) draws one; 'none' leaves the layout unseeded. The seed in "
    "use is ALWAYS printed and stored in the demo bag's meta, so any "
    "session -- including a bad draw -- can be reproduced later.",
)
parser.add_argument(
    "--stop-on-success",
    dest="stop_on_success",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="close the take automatically the moment the success gate fires "
    "(default on). Keeps a take to ONE episode: the scene no longer resets "
    "on success, so recording past it would append a second, failed episode "
    "and halve the take's score. --no-stop-on-success to keep rolling.",
)
parser.add_argument(
    "--lock-base",
    dest="lock_base",
    action="store_true",
    help="bolt the pelvis to the world and stop driving the legs. For testing the "
    "MANIPULATION half of an on-ladder task while the spawn settle is losing "
    "height (the robot slides ~0.9 m off its staged tread): a bolted base "
    "holds the staged pose exactly, so the arms can be exercised against the "
    "fixture. NOT for collecting demos -- the legs are inert and the base "
    "cannot fall, so the trajectory is not a real attempt.",
)
parser.add_argument(
    "--record-settle",
    dest="record_settle",
    action="store_true",
    help="also record the ~90 startup settle steps (feet planting + SONIC warm-in) "
    "that run BEFORE 'Teleop ready'. Off by default: those steps are not "
    "operator-driven. On for evidence of spawn-time failures -- a staged "
    "payload is lost during this window, and a take that starts at the main "
    "loop only ever shows it already on the floor. Requires --record.",
)
parser.add_argument(
    "--camera",
    choices=["auto", "follow", "static", "fixture", "bench", "crate"],
    default="auto",
    help="auto (default) = pick per task so the video keeps the robot AND what the task is "
    "about in frame: a side view of the socket on the mate legs (S03/S11), of the bench on "
    "the grasp legs (S07/S08), of the crate on S06, and the chase cam framing the task's own "
    "objects elsewhere (ladder+fixture, ladder+crate on S05, bench+ladder on S09). "
    "crate = one fixed shot of the robot and the disposal crate from the side. "
    "bench = one fixed shot of the robot and the fresh bulb on the bench, from the side of "
    "the robot->bulb line, for the tabletop legs (S07/S08/S09). "
    "fixture = one fixed shot of the socket from the SIDE (off the robot->socket "
    "line, so neither robot nor ladder hides it), level at 1.5 m: the socket sits "
    "in the upper third of frame and the floor beneath it in the lower -- for "
    "evidence of anything that falls out of, or snaps into, the socket. "
    "follow = over-the-shoulder chase cam that orbits with the robot. "
    "static = one fixed wide shot, placed once from the scene layout so the "
    "robot, the whole ladder and the fixture all stay in frame (better for "
    "tasks where the action moves between two fixed places, e.g. S01).",
)
parser.add_argument(
    "--walk_scale",
    type=float,
    default=1.0,
    help="metres/second at full LEFT-stick deflection (default 1.0). The SONIC "
    "command clamp allows 1.0 forward / 0.5 lateral, so the previous 0.5 "
    "reached only half the available speed. Raise for faster traverses, "
    "lower for fine positioning.",
)
parser.add_argument("--teleop_device", default="controller_rel")
parser.add_argument("--hand", default="dex3", choices=["dex3", "inspire"])
parser.add_argument("--walk_onnx", default=f"{_POLICY_DIR}/GR00T-WholeBodyControl-Walk.onnx")
parser.add_argument("--balance_onnx", default=f"{_POLICY_DIR}/GR00T-WholeBodyControl-Balance.onnx")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--record",
    choices=["none", "bag"],
    default="none",
    help="bag: record the session as a robomimic-style HDF5 demo bag (same format "
    "as record_run.py, via TeleopTrajectoryRecorder). Episodes split on [R] "
    "resets and on exit; the benchmark score is embedded in meta.json.",
)
parser.add_argument(
    "--record-start",
    choices=["auto", "toggle"],
    default="auto",
    help="auto: recording runs from the moment teleop starts. toggle: recording "
    "starts OFF and the operator turns it on/off live -- keyboard [C], or the "
    "right controller's B button (the UPPER face button) in VR. The toggle also works in auto "
    "mode (pause/resume); each OFF closes the episode and flushes the bag.",
)
parser.add_argument(
    "--record-format",
    choices=["hdf5", "npz"],
    default="hdf5",
    help="bag format. hdf5 (default): robomimic-style data/demo_<i> groups. "
    "npz: flat numpy archive, no h5py needed to read. The scorer handles both.",
)
parser.add_argument(
    "--record-video",
    action="store_true",
    help="also render an MP4 of the session (follow-camera on the robot) + a poster "
    "PNG into the session folder. Frames are captured only while recording is "
    "ON and streamed to disk (constant memory). Costs sim speed -- off by default.",
)
parser.add_argument(
    "--record-images",
    action="store_true",
    help="also capture the env's own cameras (wrist_camera, ego_camera, ...) as JPEGs "
    "into <session>/images/<camera>/. Decimated by --images-stride; file index = "
    "recorded-step counter, aligning frames 1:1 with bag rows. Costs sim speed -- "
    "off by default.",
)
parser.add_argument(
    "--images-stride",
    type=int,
    default=5,
    help="capture every Nth recorded step for --record-images (default 5 = 10 Hz at the 50 Hz control rate).",
)
parser.add_argument(
    "--out",
    default="",
    help="output directory for the --record bags. Default: the dataset-first tree "
    "<captures>/<task>/<hand>/<kind>/<input>/<YYYY-MM-DD>/<HHMMSS>/, with one "
    "epNN_score<X.XX>/ folder (bag + meta + video) per record-on..off take -- "
    "each take carries its OWN score, so a listing reads as per-demo results. "
    "One dimension per level: hand (dex3=43 vs inspire=53 joint "
    "columns), kind = hdf5|npz(+images), input device -- so <task>/<hand>/"
    "<kind>/ is always a schema-homogeneous training dataset. <captures> = "
    "$FIATLUX_CAPTURES_DIR, else ../teleop-captures beside the repo -- OUTSIDE "
    "the git tree, so sessions never pollute it.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="end the session cleanly after N teleop steps (0 = run until quit). "
    "Gives scripted/timed captures a clean exit -- NOTE: killing the process "
    "instead (SIGINT) is NOT clean, Kit's own signal handler fast-exits and "
    "skips the final bag write; only episodes already closed by [R] survive.",
)
parser.add_argument(
    "--keys",
    default="",
    help="scripted keyboard input for hands-off tests: comma-separated NAME@SECONDS pairs "
    "(teleop time, so 0 = 'Teleop ready'), e.g. \"G@4,R@8,G@12,ESCAPE@15\". Injected into the "
    "same queue as real key presses; keyboard input only.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.record_video or args.record_images:
    args.enable_cameras = True  # RTX camera sensors; frames need the camera pipeline
    if args.record == "none":
        args.record = "bag"  # visual capture documents a demo session; it rides along with a bag
if getattr(args, "headless", False):
    # The FIATLUX scenes always carry camera sensors (wrist/ego); with a GUI they render anyway,
    # but headless needs the camera pipeline explicitly or env creation fails.
    args.enable_cameras = True
args.xr = args.input == "vr"  # CloudXR stereo render only in VR mode (keyboard = desktop GUI)
args.device = "cuda:0"  # env physics on GPU (xr otherwise defaults to cpu)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import collections  # noqa: E402
from dataclasses import dataclass  # noqa: E402

import fiatlux_task  # noqa: F401,E402
import fiatlux_teleop  # noqa: F401,E402  -- registers the FIATLUX-*-Teleop gym ids
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import (  # noqa: E402
    set_layout_seed,
)

from isaaclab.devices.device_base import DeviceBase  # noqa: E402
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg  # noqa: E402
from isaaclab.devices.teleop_device_factory import create_teleop_device  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

# ---------------------------------------------------------------------------
# SONIC contract (29-joint obs / 15-action legs+waist; from NVIDIA's GR00T-WholeBodyControl)
# ---------------------------------------------------------------------------
SONIC_JOINTS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
DEFAULT_15 = np.array([-0.1, 0, 0, 0.3, -0.2, 0, -0.1, 0, 0, 0.3, -0.2, 0, 0, 0, 0], dtype=np.float32)
DEFAULT_29 = np.zeros(29, dtype=np.float32)
DEFAULT_29[:15] = DEFAULT_15
ANG_VEL_SCALE, DOF_POS_SCALE, DOF_VEL_SCALE, ACTION_SCALE = 0.5, 1.0, 0.05, 0.25
CMD_SCALE = np.array([2.0, 2.0, 0.5], dtype=np.float32)
HEIGHT_CMD, LEAN_MAG = 0.74, 0.30
OBS_DIM, HIST_LEN, N_ACT = 86, 6, 15

_ROW = DeviceBase.MotionControllerDataRowIndex.INPUTS.value
_IDX = DeviceBase.MotionControllerInputIndex
_TL = DeviceBase.TrackingTarget.CONTROLLER_LEFT
_TR = DeviceBase.TrackingTarget.CONTROLLER_RIGHT


class WalkRetargeter(RetargeterBase):
    """LEFT stick -> walk; RIGHT stick X -> turn; RIGHT A (lower) -> stop; RIGHT B (upper) -> record toggle;
    LEFT X/Y -> lean. Output [vx,vy,wz,stop,lean,rec,lsqueeze,lboth]; the last two feed the rail-hand
    toggle (left grip squeeze = take the left arm back; both left face buttons = put it on the rail)."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg

    @staticmethod
    def _read(data, target):
        cd = data.get(target) if data else None
        if cd is None or len(cd) <= _ROW:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        inp = cd[_ROW]
        return (
            float(inp[_IDX.THUMBSTICK_X.value]),
            float(inp[_IDX.THUMBSTICK_Y.value]),
            float(inp[_IDX.BUTTON_0.value]),
            float(inp[_IDX.BUTTON_1.value]),
            float(inp[_IDX.SQUEEZE.value]),
        )

    def retarget(self, data):
        lx, ly, lb0, lb1, lsq = self._read(data, _TL)
        rx, ry, rb0, rb1, _rsq = self._read(data, _TR)
        dz = self.cfg.deadzone
        lx = lx if abs(lx) > dz else 0.0  # deadzone: idle thumbstick drift must not walk the robot
        ly = ly if abs(ly) > dz else 0.0
        rx = rx if abs(rx) > dz else 0.0
        ms = self.cfg.movement_scale
        stop = 1.0 if rb0 > 0.5 else 0.0
        lean = (1.0 if lb0 > 0.5 else 0.0) - (1.0 if lb1 > 0.5 else 0.0)
        rec = 1.0 if rb1 > 0.5 else 0.0  # right B (UPPER face button): record toggle (edge-detected downstream)
        lsqueeze = 1.0 if lsq > 0.5 else 0.0
        lboth = 1.0 if (lb0 > 0.5 and lb1 > 0.5) else 0.0  # both left face buttons (lean cancels to 0)
        return torch.tensor(
            [ly * ms, -lx * ms, -rx, stop, lean, rec, lsqueeze, lboth], device=self.cfg.sim_device, dtype=torch.float32
        )

    def get_requirements(self):
        return [RetargeterBase.Requirement.MOTION_CONTROLLER]


@dataclass
class WalkRetargeterCfg(RetargeterCfg):
    movement_scale: float = 1.0  # m/s at full stick; --walk_scale overrides
    deadzone: float = 0.12
    retargeter_type: type = WalkRetargeter


def main():  # noqa: C901  (one long orchestration: env setup + settle/resettle + the teleop loop)
    # --- env: real Insert-Teleop scene, retimed for SONIC, base freed ---
    # Subtask teleop envs read the hand from the environment (the swap has to happen inside the
    # cfg's __post_init__, before its action terms are built). Insert/Carry keep their own
    # post-parse patches below.
    _is_subtask_task = re.search(r"-S\d\d-", args.task) is not None
    # Every env built on the subtask recipe reads its hand from this variable -- the S01..S12
    # twins and any env derived from them. Only the legacy Insert/Carry patches below ignore
    # it, and they are harmless with it set. It used to be set only for "-S<NN>-" ids, so a
    # derived env silently came up with Inspire while the launcher reported dex3.
    os.environ["FIATLUX_TELEOP_HAND"] = args.hand.lower()

    # Room layout seed. Drawn here rather than left to the scene module's unseeded default so
    # the value is KNOWN: an unreproducible bad draw (robot spawned collapsed, ladder tipped at
    # settle) is otherwise impossible to hand to anyone else. Applied before parse_env_cfg --
    # the layout is sampled inside the cfg's __post_init__.
    layout_seed: int | None = None
    if str(args.layout_seed).lower() != "none":
        if str(args.layout_seed).lower() == "random":
            layout_seed = random.randrange(2**31)
        else:
            try:
                layout_seed = int(args.layout_seed)
            except ValueError:
                raise SystemExit(f"--layout_seed must be an integer, 'random', or 'none'; got {args.layout_seed!r}")
        set_layout_seed(layout_seed)
        print(
            f"[sonic] room layout seed {layout_seed}  (reproduce this exact room with --layout_seed {layout_seed})",
            flush=True,
        )

    # Inspire thumb-fix asset (teleop only): the stock thumb frame leans over the palm and buries
    # itself in a held bulb; the wrapper USD (scripts/omniverse/inspire_thumb_frame.py) re-authors
    # it to stand off and grasp cleanly. On by default for Inspire, resolved dynamically so any
    # machine works: local copy, else GCS, else regenerate in-process. Explicit
    # FIATLUX_TELEOP_ROBOT_USD wins; fall back to the stock hand only if all three fail.
    if args.hand == "inspire" and not os.environ.get("FIATLUX_TELEOP_ROBOT_USD"):
        from fiatlux_task.assets import FIATLUX_ASSETS_DIR, G1_USD

        _tf = G1_USD[: -len(".usd")] + "_thumbfix.usd"
        if not os.path.isfile(_tf) and os.path.isfile(G1_USD):
            _gcs = "gs://fiatlux/assets/" + os.path.relpath(_tf, FIATLUX_ASSETS_DIR)
            print(f"[sonic] Inspire thumb-fix asset missing locally; trying GCS: {_gcs}", flush=True)
            try:
                import subprocess

                subprocess.run(["gsutil", "-q", "cp", _gcs, _tf], check=True, timeout=180)
                print("[sonic] fetched the thumb-fix asset from GCS.", flush=True)
            except Exception as _e:  # noqa: BLE001
                print(f"[sonic] GCS fetch unavailable ({_e}); regenerating the thumb-fix asset...", flush=True)
                try:
                    import importlib.util

                    _gp = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)), "..", "omniverse", "inspire_thumb_frame.py"
                    )
                    _spec = importlib.util.spec_from_file_location("inspire_thumb_frame", _gp)
                    _gen = importlib.util.module_from_spec(_spec)
                    _spec.loader.exec_module(_gen)  # module-level `from pxr import ...` is fine post-Kit
                    _gen.generate_thumbfix(G1_USD, _tf)
                    print("[sonic] regenerated the thumb-fix asset locally.", flush=True)
                except Exception as _e2:  # noqa: BLE001
                    print(
                        f"[sonic] WARNING: could not obtain the Inspire thumb-fix asset ({_e2}); using the "
                        "STOCK thumb -- a held bulb will show thumb interpenetration.",
                        flush=True,
                    )
        if os.path.isfile(_tf):
            os.environ["FIATLUX_TELEOP_ROBOT_USD"] = _tf
            print(f"[sonic] Inspire thumb-fix asset active: {_tf}", flush=True)

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise ValueError("expected a ManagerBasedRLEnv task")
    # Insert-Teleop defaults to Inspire (swap to Dex3 on request); Carry-Teleop is Dex3-native
    # (swap to Inspire on request). Each env's patch handles the robot + hand-action repoint.
    #
    # SUBTASKS ARE EXCLUDED. These are substring matches on the task id, and the subtask ids
    # S05-CarryBulbToDisposal / S09-CarryBulbToLadder contain "Carry" -- so `--hand inspire`
    # applied the legacy Carry patch to an already-Inspire-native subtask and swap_robot_variant
    # raised "don't know how to remap joint_names=['R_index_proximal_joint', ...]". The subtask
    # twins pick their own hand inside apply_subtask_teleop via FIATLUX_TELEOP_HAND, set above.
    if not _is_subtask_task:
        if args.hand.lower() == "dex3" and "Insert" in args.task:
            from fiatlux_teleop.insert_teleop_env_cfg import apply_dex3_hands

            apply_dex3_hands(env_cfg)
        elif args.hand.lower() == "inspire" and "Carry" in args.task:
            from fiatlux_teleop.carry_teleop_env_cfg import apply_inspire_hands

            apply_inspire_hands(env_cfg)

    env_cfg.sim.dt = 0.005  # 200 Hz (SONIC's rate)
    env_cfg.decimation = 4  # -> 50 Hz control
    env_cfg.sim.render_interval = 4
    env_cfg.terminations.time_out = None
    # FREE the base so SONIC can balance + walk (the teleop env bolts it down for stationary insert)
    env_cfg.scene.robot.spawn.articulation_props.fix_root_link = bool(args.lock_base)
    if args.lock_base:
        print(
            "[sonic] LOCK-BASE: pelvis bolted to the world, legs not driven. The robot cannot "
            "fall or slide -- use for exercising the arms against the scene, NOT for demos.",
            flush=True,
        )
    # Harden the spawn against the intermittent PhysX launch: cap depenetration velocity (a bad
    # contact can't fling the free base metres up) and drop the random joint-offset reset (it
    # perturbs the free-base start pose out of SONIC's balance basin). Keep the bulb reset.
    env_cfg.scene.robot.spawn.rigid_props.max_depenetration_velocity = 1.0
    for _ev in ("reset_robot_joints", "reset_robot", "randomize_robot_root", "push_robot"):
        if getattr(env_cfg.events, _ev, None) is not None:
            setattr(env_cfg.events, _ev, None)
    # Operator-paced: clear the FAILURE terminations (a "fall" term would auto-reset mid-
    # instability). KEEP `success` -- it is the only place a task's success predicate is
    # evaluated, and `recording.term_flag` records it as the `success_term` column that
    # `scripts/score.py` reads. Clearing it does not disable scoring; term_flag falls back to an
    # all-False vector, so every recorded demo silently scores 0.0 however well it was performed.
    # (subtask_teleop.apply_subtask_teleop keeps it for the same reason; this loop used to undo
    # that a few lines later.)
    # Stash the success gate BEFORE clearing it: the recorder evaluates the same conjuncts itself
    # (see teleop_recording), and once the termination is None its spec is gone from the cfg.
    _succ_cfg = getattr(env_cfg.terminations, "success", None)
    if _succ_cfg is not None:
        env_cfg.teleop_success_spec = dict(_succ_cfg.params or {})
        env_cfg.teleop_success_fn = _succ_cfg.func
    # `success` included: as a TERMINATION it resets the scene the instant the gate fires, which
    # yanks the episode away from the operator mid-take. The recorder evaluates the same gate
    # itself and records `success_term`, so takes still score -- and an episode ends only when the
    # operator ends it.
    for _t in (
        "time_out",
        "success",
        "object_dropped",
        "robot_fell",
        "fall_terminated",
        "bad_orientation",
        "base_contact",
        "illegal_contact",
    ):
        if getattr(env_cfg.terminations, _t, None) is not None:
            setattr(env_cfg.terminations, _t, None)
    # Take the robot's x,y from whatever env is loaded (task-specific placement); keep SONIC's
    # standing joint stance so the balance policy starts in-distribution, and raise the spawn a
    # touch so the feet clear the floor. Generalizes across tasks (Insert, Carry, ...).
    _p = env_cfg.scene.robot.init_state.pos
    env_cfg.scene.robot.init_state.pos = (_p[0], _p[1], max(_p[2], 0.80))
    # SONIC leg stance (both envs need it). The ARM spawn pose is env-dependent because the two envs'
    # IK behaves differently: Insert's redundant IK RELAXES the arm to a natural low rest regardless of
    # spawn (so a bent-elbow spawn settles to ~0.17), but Carry's IK HOLDS whatever it spawns in -- so
    # Carry must spawn directly in the natural pose or it stays tucked at the spawn angle. Keep Insert's
    # spawn exactly as it was so its settled pose is unchanged.
    _legs = {".*_hip_pitch_joint": -0.1, ".*_knee_joint": 0.3, ".*_ankle_pitch_joint": -0.2}
    # Insert is the ONE env whose IK relaxes the 1.57 spawn on its own. Everything else -- Carry,
    # LadderGallery, the S01..S12 subtask twins, and any env derived from them -- holds whatever
    # it spawns in, so all of them take the natural low spawn. Matching the exception rather than
    # listing the rule: a new env used to fall through to Insert's pose by default and come up
    # with its arm tucked at the chest.
    if "Insert" not in args.task:
        # Drop the SHOULDER so the arm hangs low. The elbow drifts up to ~1.1 on its own (redundant IK),
        # so we don't fight it -- a low/back shoulder points the upper arm down so the bent forearm sits
        # low instead of up at the chest. (Per operator: change the joint above the 90-deg elbow.)
        _arm_spawn = {".*_shoulder_pitch_joint": -0.35, ".*_elbow_joint": 0.35}
    else:
        _arm_spawn = {".*_elbow_joint": 1.57}  # Insert etc: IK relaxes this to ~0.17 (unchanged from before)

    # MERGE, don't assign: subtasks stage joints that the task depends on -- the carry/hold ones
    # place the arms where they must be to hold the payload (e.g. S03's LADDER_CARRY_ARM_JOINT_POS,
    # merged in by the subtask itself). A hard assign here threw those away, so the operator
    # started NOT holding the ladder/bulb and the payload dropped onto the robot.
    # Our generic teleop spawn is the BASE; anything the task staged specifically wins.
    # Expand OUR ".*_x_joint" defaults to explicit left_/right_ names first. Isaac Lab rejects
    # a joint matched by two keys ("Multiple matches for 'right_shoulder_pitch_joint':
    # '.*_shoulder_pitch_joint' and 'right_shoulder_pitch_joint'"), which is exactly what happens
    # when a subtask stages ONE arm (S03/S07 name only right-arm joints) and we also carry a
    # both-sides pattern. Explicit names let the task's entry simply replace ours, per joint.
    def _explicit(d):
        out = {}
        for k, v in d.items():
            if k.startswith(".*_"):
                out[f"left_{k[3:]}"] = v
                out[f"right_{k[3:]}"] = v
            else:
                out[k] = v
        return out

    _staged = dict(env_cfg.scene.robot.init_state.joint_pos or {})
    _base_generic = {".*_hip_pitch_joint", ".*_knee_joint", ".*_ankle_pitch_joint"}
    _task_specific = {k: v for k, v in _staged.items() if k not in _base_generic}
    env_cfg.scene.robot.init_state.joint_pos = {
        **_explicit(_legs),
        **_explicit(_arm_spawn),
        **_task_specific,
    }
    # Stiffen the arms so the ready pose (elbow ~90) HOLDS against gravity + the heavy Dex3 hand. The
    # RL-tuned arm gains (~50 at the elbow) are too soft, so the arm droops toward straight before the
    # operator connects. A firmer PD tracks the IK's joint targets, keeping the elbow bent, and also
    # makes teleop feel more precise. (Arms are commanded externally, separate from SONIC's legs.)
    _arms = env_cfg.scene.robot.actuators["arms"]
    _arms.stiffness = {".*_(shoulder|elbow|wrist).*_joint": 200.0}
    _arms.damping = {".*_(shoulder|elbow|wrist).*_joint": 20.0}

    # Rail hand (the settle below): FIATLUX_RAIL_HOLD=1 forces it on, 0 off; unset = AUTO, which is
    # on exactly when the task stages the robot ON the ladder (pelvis >= 1 m above the ladder root:
    # S03/S11 and the descend legs), because that is the stance that topples backward off the
    # tread unattended (ladder-diag: 27-29% idle, 0/30 with the hand on the cap). Floor spawns
    # (climb legs, bench legs) keep the left arm free -- the cap is out of reach there anyway.
    _rail_env = os.environ.get("FIATLUX_RAIL_HOLD", "auto").lower()
    _rail_on = False
    if _rail_env != "0" and getattr(env_cfg.scene, "ladder", None) is not None:
        _on_ladder = float(env_cfg.scene.robot.init_state.pos[2]) - float(env_cfg.scene.ladder.init_state.pos[2]) >= 1.0
        _rail_on = _rail_env == "1" or (_rail_env == "auto" and _on_ladder)
        if _rail_env == "auto":
            print(
                f"[sonic] rail hand: {'ON' if _rail_on else 'off'} (auto: robot staged "
                f"{'on' if _on_ladder else 'off'} the ladder; FIATLUX_RAIL_HOLD=0/1 overrides)",
                flush=True,
            )
    if _rail_on:
        # A sensor for the LEFT hand's force against the ladder only. The env's own
        # left_hand_contact is filtered to the task's bulb, so the brace never shows in it, and
        # the unfiltered net force also counts the hand's own colliders.
        from isaaclab.sensors import ContactSensorCfg  # noqa: E402

        # The WHOLE left arm, not just the hand: a badly converged approach can park the elbow
        # inside the ladder's side rail while the base is pinned, and that has to be seen
        # before the release (VR batch, S11 inspire seed 4: 1.1 m/s kick on release).
        env_cfg.scene.rail_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/(left_shoulder_.*|left_elbow_.*|left_wrist_.*|left_hand_.*|L_.*)",
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Ladder"],
            history_length=1,
        )

    # FIATLUX_SETTLE_PROBE=1: one contact sensor per arm, filtered against every scene object at
    # once, so a hit during staging is attributed to the thing that was hit (socket, bulb, wall,
    # ladder) instead of a single net number. Diagnostic only; off by default.
    _probe_names: list[str] = []
    if os.environ.get("FIATLUX_SETTLE_PROBE", "0") == "1":
        from isaaclab.sensors import ContactSensorCfg as _ProbeSensorCfg  # noqa: E402

        _probe_paths = []
        for _nm in ("socket", "fresh_bulb", "old_bulb", "ladder", "room", "fixture", "table", "bin", "pendant"):
            _obj = getattr(env_cfg.scene, _nm, None)
            if _obj is not None and getattr(_obj, "prim_path", None):
                _probe_names.append(_nm)
                _probe_paths.append(_obj.prim_path)
        for _side, _bodies in (
            ("r", "right_shoulder_.*|right_elbow_.*|right_wrist_.*|right_hand_.*|R_.*"),
            ("l", "left_shoulder_.*|left_elbow_.*|left_wrist_.*|left_hand_.*|L_.*"),
        ):
            setattr(
                env_cfg.scene,
                f"probe_{_side}",
                _ProbeSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/(" + _bodies + ")",
                    filter_prim_paths_expr=list(_probe_paths),
                    history_length=1,
                ),
            )
        print(f"[sonic] settle probe: arm contacts attributed to {_probe_names}", flush=True)
        if args.out:
            os.makedirs(args.out, exist_ok=True)
            with open(os.path.join(args.out, "probe_targets.txt"), "w") as _fh:
                _fh.write("\n".join(_probe_names) + "\n")

    if args.xr:
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    if args.record_video:
        # RTX video camera in the scene (same rig as record_run.py); posed per frame as a follow-cam.
        from fiatlux_task.viz import make_video_camera_cfg

        env_cfg.scene.video_cam = make_video_camera_cfg(960, 544)  # height % 16 == 0: no ffmpeg resize
        # Wider lens for the review footage. The 20 mm default is a 33 deg vertical field, and a
        # robot, a ladder 2 m away and a 2.2 m fixture do not fit that from inside an 8 m room --
        # the chase cam either clipped the fixture or lost the robot. 12 mm on the 20.955 mm
        # aperture is ~82 x 53 deg, which frames all three from ~4 m. Video only; the ego camera
        # and the RL observation cameras are untouched.
        env_cfg.scene.video_cam.spawn.focal_length = 12.0

    env = gym.make(args.task, cfg=env_cfg).unwrapped
    robot = env.scene["robot"]

    # Left arm rest pose: shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw in degrees,
    # captured from an operator take. Empty string keeps the asset's pose. The recorded pose is
    # the ladder-cap approach, meaningful only for the rail-hand feature -- gate the write on
    # _rail_on (computed above, before gym.make): default_joint_pos is shared state that the
    # post-settle restore and mid-session reset re-home both read for EVERY task, rail or not, so
    # writing it unconditionally silently replaces a non-rail task's own left-arm rest pose with
    # this ladder-specific one.
    _ARM_REST_L = os.environ.get("FIATLUX_ARM_REST_LEFT", "-4.8,9.4,0.5,15.6,-4.3,-9.6,4.8")
    if _rail_on and _ARM_REST_L.strip():
        _names = [
            f"left_{_n}_joint"
            for _n in (
                "shoulder_pitch",
                "shoulder_roll",
                "shoulder_yaw",
                "elbow",
                "wrist_roll",
                "wrist_pitch",
                "wrist_yaw",
            )
        ]
        _vals = [float(_x) for _x in _ARM_REST_L.split(",")]
        if len(_vals) == len(_names) and all(_n in robot.joint_names for _n in _names):
            for _n, _v in zip(_names, _vals):
                robot.data.default_joint_pos[:, robot.joint_names.index(_n)] = math.radians(_v)
            print(f"[sonic] left arm rest pose set to {_ARM_REST_L} deg (recorded)", flush=True)
        else:
            print(f"[sonic] ignoring FIATLUX_ARM_REST_LEFT: need {len(_names)} values", flush=True)
    # Say which hand was actually BUILT, not which was requested: the two have disagreed silently
    # before (an env came up Inspire under "hand=dex3"). Joint names are the ground truth --
    # Dex3 fingers are right_hand_*_N_joint, Inspire's are R_*_joint.
    _jn = list(robot.joint_names)
    _built = (
        "dex3"
        if any("right_hand_" in n for n in _jn)
        else "inspire"
        if any(n.startswith("R_") for n in _jn)
        else "no hand joints found"
    )
    print(f"[sonic] robot built with {len(_jn)} joints -- hand: {_built} (requested {args.hand.lower()})", flush=True)
    if _built not in (args.hand.lower(), "no hand joints found"):
        print(f"[sonic] WARNING: hand mismatch -- the env ignored --hand {args.hand}", flush=True)
    dev = env.device

    reset_flag = {"do": False}

    def _score_line(info: dict) -> str:
        """One-line benchmark score for the operator, printed on every flush.

        The score is already computed and embedded in meta.json by the recorder's write();
        without printing it the operator has to open the file to learn whether the take they
        just finished actually counted as a success.
        """
        sc = info.get("score") or {}
        if not sc:
            return "  score: unavailable (see meta.json)"
        n = sc.get("episodes") or 0
        rate = sc.get("success_rate") or 0.0
        return (
            f"  score: success {round(rate * n)}/{n} ({rate:.0%})  "
            f"mean_score={sc.get('mean_score', 0.0):.2f}  "
            f"clean={sc.get('clean_success_rate', 0.0):.0%}  "
            f"broken={sc.get('broken_rate', 0.0):.0%}  "
            f"dropped={sc.get('dropped_rate', 0.0):.0%}  "
            f"peak_force={sc.get('peak_contact_force', 0.0):.1f}N"
        )

    rec_flag = {"toggle": False}  # operator asked to flip recording on/off (keyboard C / VR right B=upper)

    def _reset():
        reset_flag["do"] = True
        print("[sonic] reset requested", flush=True)

    # VR only: append the walk retargeter + unify ALL retargeter devices to the env device (so
    # advance()'s torch.cat over the retargeter outputs doesn't mix cpu/cuda). advance() = [arm..., walk(5)].
    teleop = None
    if args.input == "vr":
        dev_cfg = env_cfg.teleop_devices.devices[args.teleop_device]
        for rt in dev_cfg.retargeters:
            rt.sim_device = str(dev)
        dev_cfg.retargeters = list(dev_cfg.retargeters) + [
            WalkRetargeterCfg(sim_device=str(dev), movement_scale=args.walk_scale)
        ]
        teleop = create_teleop_device(
            args.teleop_device, env_cfg.teleop_devices.devices, {"R": _reset, "RESET": _reset}
        )
        print(f"[sonic] VR teleop device: {args.teleop_device} (+walk)", flush=True)

    # --- SONIC policy + joint maps ---
    for _onnx_path, _flag in ((args.walk_onnx, "--walk_onnx"), (args.balance_onnx, "--balance_onnx")):
        if not os.path.exists(_onnx_path):
            raise FileNotFoundError(
                f"SONIC policy ONNX file not found at '{_onnx_path}' ({_flag}). "
                "Please specify valid paths via --walk_onnx and --balance_onnx, "
                "or set the SONIC_POLICY_DIR environment variable."
            )

    jn = robot.joint_names
    sidx = [jn.index(n) for n in SONIC_JOINTS]
    act_idx = torch.tensor(sidx[:15], device=dev)
    walk_sess = ort.InferenceSession(args.walk_onnx, providers=["CPUExecutionProvider"])
    bal_sess = ort.InferenceSession(args.balance_onnx, providers=["CPUExecutionProvider"])
    in_name = walk_sess.get_inputs()[0].name
    print(f"[sonic] loaded Walk + Balance ONNX (input '{in_name}', 516->15)", flush=True)

    loco_cmd = np.zeros(3, dtype=np.float32)
    rpy_cmd = np.zeros(3, dtype=np.float32)
    obs_hist = collections.deque([np.zeros(OBS_DIM, dtype=np.float32)] * HIST_LEN, maxlen=HIST_LEN)
    last_action = np.zeros(N_ACT, dtype=np.float32)

    def build_obs():
        o = np.zeros(OBS_DIM, dtype=np.float32)
        o[0:3] = loco_cmd * CMD_SCALE
        o[3] = HEIGHT_CMD
        o[4:7] = rpy_cmd
        o[7:10] = robot.data.root_ang_vel_b[0].cpu().numpy() * ANG_VEL_SCALE
        o[10:13] = robot.data.projected_gravity_b[0].cpu().numpy()
        q = robot.data.joint_pos[0, sidx].cpu().numpy()
        qd = robot.data.joint_vel[0, sidx].cpu().numpy()
        o[13:42] = (q - DEFAULT_29) * DOF_POS_SCALE
        o[42:71] = qd * DOF_VEL_SCALE
        o[71:86] = last_action
        return o

    env.reset()

    # Optional demo recording: the benchmark's bag writer, made teleop-safe (task-field guards +
    # operator-driven episode boundaries). Created BEFORE the settle so it exists for the loop,
    # but record_step only runs inside the main loop -- settle/resettle steps are never recorded.
    recorder = None
    video = None
    ego_video = None
    images = None
    recording_on = False
    if args.record == "bag":
        import time as _time

        from fiatlux_teleop.teleop_recording import TeleopTrajectoryRecorder

        if not args.out:
            # <captures>/task/input-mode/date/session-time; the session folder gains a _score<mean>
            # suffix at clean exit (the score exists only once the session is over). Captures live
            # OUTSIDE the git tree (../teleop-captures beside the repo) unless overridden.
            _repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            _captures = os.environ.get("FIATLUX_CAPTURES_DIR", os.path.join(os.path.dirname(_repo), "teleop-captures"))
            # Dataset-first layout: one dimension per level, so a TRAINING DATASET is simply
            # <task>/<hand>/<kind>/** and every session inside it is schema-homogeneous:
            #   hand  -- dex3 bags have 43 joint columns, inspire 53: never mixable;
            #   kind  -- bag format + whether images were captured, so sessions with and
            #            without frames never share a folder;
            #   input -- keyboard vs vr demos separable below that; then date / session.
            # (--record-video does NOT fork the path: the MP4 is review material, not data schema.)
            _kind = args.record_format + ("+images" if args.record_images else "")
            # Hand label from the LIVE robot, not --hand: the flag only swaps hands on tasks that
            # define a swap, so on others the mounted hand can differ from the flag -- and a bag
            # filed under the wrong hand poisons an otherwise schema-homogeneous dataset.
            _jn = env.scene["robot"].joint_names
            _hand = (
                "dex3"
                if any("hand_index_0" in n for n in _jn)
                else "inspire"
                if any("proximal" in n for n in _jn)
                else args.hand
            )
            if _hand != args.hand:
                print(
                    f"[sonic] note: --hand {args.hand} requested but this task mounts {_hand}; "
                    f"filing the session under {_hand}/",
                    flush=True,
                )
            args.out = os.path.join(
                _captures, args.task, _hand, _kind, args.input, _time.strftime("%Y-%m-%d"), _time.strftime("%H%M%S")
            )
        recorder = TeleopTrajectoryRecorder(
            env, policy_spec=f"teleop_{args.input}", seed=layout_seed if layout_seed is not None else 0
        )
        # Per-take capture: every record-on..record-off span writes its OWN epNN/ subfolder --
        # a single-demo bag, a meta.json carrying THAT take's score, and a finalized,
        # immediately-playable video. Each take is shareable the moment it ends, and a bad take
        # is deleted on its own without touching the rest of the session.
        from fiatlux_teleop.teleop_recording import ImageCapture, StreamingVideoRecorder

        ep_idx = 0
        ep_scores = []
        if args.record_images:
            recorder._meta["images"] = {
                "dir": "images",
                "stride": args.images_stride,
                "index": "this take's bag row (f<row>.jpg pairs 1:1 with run.* row <row>)",
            }

        def _seal_take(_ep_dir, _info):
            """Rename a finished take's folder to carry its own score: ep03 -> ep03_score1.00.

            Called only after the bag AND the video are closed, so nothing is still writing into
            the old path. meta.json refers to its bag by bare filename, so the rename is safe.
            The score belongs on the take, not on the session: a listing of a collection session
            then reads as per-demo results rather than one averaged number.
            """
            _sc = (_info.get("score") or {}).get("mean_score")
            if _sc is None:
                return _ep_dir
            _dst = f"{_ep_dir}_score{_sc:.2f}"
            try:
                os.rename(_ep_dir, _dst)
                return _dst
            except OSError as _e:
                print(f"[sonic] could not append score to {_ep_dir} ({_e})", flush=True)
                return _ep_dir

        def _open_take_capture(_ep_dir):
            """Per-take writers, opened at record-ON and closed at record-OFF.

            Two videos, because one angle cannot serve both purposes: `video.mp4` is the
            third-person follow cam (what happened in the room) and `ego.mp4` is the robot's own
            head camera (what the robot could see). The follow cam's offset is fixed in WORLD
            space, so the robot can turn its back on it -- the ego view is the one that always
            shows the manipulation.
            """
            _v = (
                StreamingVideoRecorder(env, env.scene["video_cam"], os.path.join(_ep_dir, "video.mp4"), fps=50)
                if args.record_video
                else None
            )
            _ego = None
            if args.record_video and "ego_camera" in getattr(env.scene, "sensors", {}):
                _ego = StreamingVideoRecorder(env, env.scene["ego_camera"], os.path.join(_ep_dir, "ego.mp4"), fps=50)
            _im = (
                ImageCapture(env, os.path.join(_ep_dir, "images"), stride=args.images_stride)
                if args.record_images
                else None
            )
            return _v, _ego, _im

        recording_on = args.record_start == "auto"
        if recording_on:
            video, ego_video, images = _open_take_capture(os.path.join(args.out, "ep00"))
        _state = "ON from start" if recording_on else "OFF -- press [C] / right ctrl B (upper) to start"
        print(
            f"[sonic] recording demos -> {args.out} ({_state}; "
            "one epNN/ folder -- bag + meta + video -- per record-on..off take)",
            flush=True,
        )

    # true world spawn (to re-home on reset) + the position-hold target.
    spawn_root = robot.data.root_state_w[:, 0:7].clone()
    # The cfg-STAGED pose, kept before the settle overwrites spawn_root below. Reset re-homes to
    # THIS, not to where the settle happened to leave the robot: on the on-ladder subtasks the
    # settle loses height (staged 1.97 m tread -> observed 0.86-1.86 m), so re-homing to the
    # settled pose made every reset inherit the loss and drift lower again on each one.
    staged_root = spawn_root.clone()
    home_xy = spawn_root[0, 0:2].cpu().numpy().copy()
    HOLD_KP, HOLD_VMAX, WALK_TH, WARMUP = 0.8, 0.25, 0.06, 100
    # Hold hysteresis: engage past 0.20 m from home, walk back until within 0.06 m, and never
    # command less than 0.12 m/s while engaged. A single deadband with an unbounded-small
    # command left the robot marching in place at the deadband edge -- SONIC walks at roughly
    # half the commanded speed, so a 0.08 m/s correction can never close the error.
    # HOLD_ANCHOR_V: home keeps following the robot until the base has actually stopped
    # (speed under this), not just until the stick is released -- SONIC glides a few steps
    # decelerating, and anchoring at the release point made the hold march it BACK the way it
    # came after every walk. Released stick = "stop here", not "return to where I let go".
    HOLD_ENGAGE, HOLD_RELEASE, HOLD_VMIN, HOLD_ANCHOR_V = 0.20, 0.06, 0.12, 0.15
    _hold_on = False

    # "hold current pose" arm action (root-frame EE pose + open grip, per arm), so the arm doesn't
    # fling before the controller streams. Order matches the action manager: R_arm(7), R_grip(1),
    # L_arm(7), L_grip(1).
    # Legs that begin with the bulb already in the right hand (the env re-seats it on the live
    # palm through its ``settle_bulb`` event) hold that grip CLOSED at rest: an open palm loses
    # the bulb during the settle, before anyone is in control (#102).
    _right_starts_closed = getattr(getattr(env_cfg, "events", None), "settle_bulb", None) is not None
    _rest_grip = {"right_wrist_yaw_link": -1.0 if _right_starts_closed else 1.0, "left_wrist_yaw_link": 1.0}
    # That payload is held by finger contact alone, so a joint/root STATE write that moves the
    # wrist -- the post-settle arm restore, the [R] re-home -- teleports the hand out from under
    # it (0.13 m in one step, measured) and it is left behind. Carry it along, rigid to the wrist.
    _payload_name = env_cfg.events.settle_bulb.params["payload_cfg"].name if _right_starts_closed else None
    _ee_bid = robot.body_names.index("right_wrist_yaw_link")

    def _carry_payload(write_fn):
        """Run ``write_fn`` (state writes that move the wrist) keeping the in-hand payload with it."""
        if _payload_name is None:
            write_fn()
            env.scene.write_data_to_sim()
            env.sim.forward()
            return
        payload = env.scene[_payload_name]
        ee = robot.data.body_state_w[:, _ee_bid, 0:7].clone()
        p_rel, q_rel = subtract_frame_transforms(
            ee[:, 0:3], ee[:, 3:7], payload.data.root_pos_w.clone(), payload.data.root_quat_w.clone()
        )
        write_fn()
        env.scene.write_data_to_sim()
        env.sim.forward()
        ee = robot.data.body_state_w[:, _ee_bid, 0:7]
        p_new, q_new = combine_frame_transforms(ee[:, 0:3], ee[:, 3:7], p_rel, q_rel)
        payload.write_root_pose_to_sim(torch.cat([p_new, q_new], dim=-1))
        payload.write_root_velocity_to_sim(torch.zeros((env.num_envs, 6), device=dev))
        env.scene.write_data_to_sim()
        env.sim.forward()

    # PRE-GRASP init-state (#125): spawn the bulb already grasped instead of open-palm->close (whose
    # grip-close transient flings the wide bulb at the top). Hold a real grasp end-state through the
    # settle -- right-hand fingers at their settled grasp angles + bulb at its captured
    # wrist-relative seat -- then release: no fling, and it still lets go when opened (no jam).
    # Inspire also pairs with the gentle finger effort in g1.py. FIATLUX_SETTLE_CARRY=0 disables it.
    _settle_mode = os.environ.get("FIATLUX_SETTLE_CARRY", "pregrasp").lower()
    # Per-hand pre-grasp end-state: finger angles + bulb pose relative to the wrist, each baked from
    # a settled genuine grasp (inspire from the #125 dump; dex3 from a held S05 carry take).
    _PREGRASP_INSPIRE = {
        "R_index_proximal_joint": 0.13,
        "R_index_intermediate_joint": 1.46,
        "R_middle_proximal_joint": 0.23,
        "R_middle_intermediate_joint": 1.46,
        "R_ring_proximal_joint": 0.42,
        "R_ring_intermediate_joint": 1.46,
        "R_pinky_proximal_joint": 0.49,
        "R_pinky_intermediate_joint": 1.47,
        "R_thumb_proximal_yaw_joint": 1.16,
        "R_thumb_proximal_pitch_joint": 0.16,
        "R_thumb_intermediate_joint": 0.26,
        "R_thumb_distal_joint": 1.19,
    }
    _PREGRASP_DEX3 = {
        "right_hand_index_0_joint": 0.615,
        "right_hand_index_1_joint": 1.058,
        "right_hand_middle_0_joint": 0.759,
        "right_hand_middle_1_joint": 1.226,
        "right_hand_thumb_0_joint": -0.642,
        "right_hand_thumb_1_joint": 0.285,
        "right_hand_thumb_2_joint": -0.830,
    }
    _PREGRASP_REL = {
        "inspire": ([0.1587, 0.0227, -0.1215], [0.6748, -0.0549, 0.0751, 0.7322]),  # rel pos, quat wxyz
        "dex3": ([0.1021, 0.0239, -0.1334], [0.6536, -0.0622, 0.0843, 0.7496]),
    }
    _hand = args.hand.lower()
    _PREGRASP_JOINTS = _PREGRASP_INSPIRE if _hand == "inspire" else _PREGRASP_DEX3 if _hand == "dex3" else {}
    _pregrasp_on = (
        _settle_mode == "pregrasp"
        and _right_starts_closed
        and _payload_name is not None
        and bool(_PREGRASP_JOINTS)
        and all(n in robot.joint_names for n in _PREGRASP_JOINTS)
    )
    _rp, _rq = _PREGRASP_REL.get(_hand, ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]))
    _PREGRASP_REL_POS = torch.tensor([_rp], device=dev)
    _PREGRASP_REL_QUAT = torch.tensor([_rq], device=dev)  # wxyz
    if _pregrasp_on:
        _pregrasp_fidx = [robot.joint_names.index(n) for n in _PREGRASP_JOINTS]
        _pregrasp_fval = torch.tensor([[_PREGRASP_JOINTS[n] for n in _PREGRASP_JOINTS]], device=dev)

    def _pin_settle_bulb():
        """Hold the pre-grasp end-state (right-hand fingers + bulb seat) through a settle step, so the
        grasp starts already formed -- no open->close fling, no penetration jam. No-op unless on."""
        if not _pregrasp_on:
            return
        payload = env.scene[_payload_name]
        robot.write_joint_state_to_sim(_pregrasp_fval, torch.zeros_like(_pregrasp_fval), joint_ids=_pregrasp_fidx)
        ee = robot.data.body_state_w[:, _ee_bid, 0:7]
        p_new, q_new = combine_frame_transforms(ee[:, 0:3], ee[:, 3:7], _PREGRASP_REL_POS, _PREGRASP_REL_QUAT)
        payload.write_root_pose_to_sim(torch.cat([p_new, q_new], dim=-1))
        payload.write_root_velocity_to_sim(torch.zeros((env.num_envs, 6), device=dev))
        env.scene.write_data_to_sim()
        env.sim.forward()

    def rest_arm_action():
        parts = []
        for ee_name in ("right_wrist_yaw_link", "left_wrist_yaw_link"):
            bid = robot.body_names.index(ee_name)
            ee_w = robot.data.body_state_w[:, bid, 0:7]
            p_b, q_b = subtract_frame_transforms(
                robot.data.root_pos_w, robot.data.root_quat_w, ee_w[:, 0:3], ee_w[:, 3:7]
            )
            parts += [p_b[0], q_b[0], torch.full((1,), _rest_grip[ee_name], device=dev)]
        return torch.cat(parts)

    # Capture the "hold arms still" IK target ONCE, here at spawn, while the elbows are at the
    # commanded 90 deg. The settle used to call rest_arm_action() fresh every frame, which re-anchored
    # the target to the arm's just-sagged pose each step -- so under gravity the elbow ratcheted DOWN
    # (1.57 -> ~0.84) before "ready". Holding ONE fixed target preserves the 90 deg pose (a small PD
    # sag aside) and also stops the redundant-7-DOF null-space from drifting while it "holds still".
    last_arm = rest_arm_action()

    # SETTLE: PIN the base perfectly upright at the spawn while the legs/feet plant + the arms reach
    # their hold pose, THEN hand a level, zero-velocity robot to SONIC. Without this the free base
    # settles slightly pitched and SONIC lurches forward catching it (a near-faceplant). ~0.35 s.
    # Arm target is re-read each frame here (stable "hold where it is"): a FIXED elbow-90 wrist target
    # is redundant, so gravity drifts the spare DOF and the elbow snaps STRAIGHT to the far limit --
    # worse than a gentle sag. Stiffer arms (set above) keep this hold much closer to the 90 deg spawn.
    pin_pose = robot.data.root_state_w[:, 0:7].clone()
    zero_vel = torch.zeros((env.num_envs, 6), device=dev)
    leg_default = torch.as_tensor(DEFAULT_15, device=dev).unsqueeze(0)

    from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import (
        ROOM_FLOOR_MAX as _RMAX,
    )
    from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import (
        ROOM_FLOOR_MIN as _RMIN,
    )

    # Follow-cam offset expressed in the ROBOT's frame, not the world's: (behind, right, up).
    # A world-fixed offset means a 180 deg turn shows the camera the robot's back, which is
    # exactly when a manipulation demo becomes unreviewable. Rotating the offset by the robot's
    # yaw keeps the same over-the-shoulder angle whichever way it faces.
    _CAM_BEHIND, _CAM_RIGHT, _CAM_UP = -2.0, -1.6, 0.9
    # SONIC sways continuously, so raw yaw would jitter the camera every frame. Track it with a
    # first-order filter instead; 0.04 settles a 180 deg turn in about a second at 50 Hz.
    _CAM_YAW_GAIN = 0.04
    _cam_yaw = [None]  # filtered camera yaw, seeded on the first captured frame
    # The thing the robot is working toward, if this scene has one (socket = the ceiling/wall
    # fixture). Used only to frame the follow-cam; absent on scenes without it.
    # Points the shot must contain: the robot, whatever it is manipulating, and the fixture it is
    # working toward. Framing only the robot loses the fixture; framing robot+fixture crops the
    # ladder, which is the thing the operator is actually steering.
    _rigids = getattr(env.scene, "rigid_objects", {})
    # What the video must keep in frame besides the robot: the task's own objects, not a fixed
    # ladder+fixture pair. The chase cam frames the robot plus these; the side-view modes below
    # are picked for the legs where a shot from beside the robot->target line reads better.
    _sid = re.search(r"-S(\d\d)-", args.task)
    _sid = _sid.group(1) if _sid else None
    _TASK_FRAME = {
        "05": ("ladder", "bin"),
        "06": ("bin",),
        "07": ("table", "fresh_bulb"),
        "08": ("table", "fresh_bulb"),
        "09": ("table", "ladder"),
    }
    _TASK_MODE = {"03": "fixture", "11": "fixture", "07": "bench", "08": "bench", "06": "crate"}
    _want = _TASK_FRAME.get(_sid, ("ladder", "socket"))
    _table_pos = getattr(getattr(getattr(env.cfg.scene, "table", None), "init_state", None), "pos", None)
    _frame_names = [n for n in _want if n in _rigids or (n == "table" and _table_pos is not None)]
    _cam_mode = args.camera if args.camera != "auto" else _TASK_MODE.get(_sid, "follow")
    print(f"[sonic] video camera: {_cam_mode} (task S{_sid or '??'}), framing robot + {list(_frame_names)}", flush=True)

    def _frame_points():
        pts = []
        for _n in _frame_names:
            if _n == "table":  # static asset, no live pose: use the cfg
                pts.append(np.array([_table_pos[0], _table_pos[1], 0.9]))
                continue
            _o = env.scene[_n]
            _pp = _o.data.root_pos_w[0].cpu().numpy()
            pts.append(_pp)
            if _n == "ladder":  # its TOP, not just the base -- that is what gets cropped
                _q = _o.data.root_quat_w[0].cpu().numpy()
                _up = np.array(
                    [
                        2.0 * (_q[1] * _q[3] + _q[0] * _q[2]),
                        2.0 * (_q[2] * _q[3] - _q[0] * _q[1]),
                        1.0 - 2.0 * (_q[1] ** 2 + _q[2] ** 2),
                    ]
                )
                pts.append(_pp + 1.18 * _up)
        return pts

    _CAM_INSET = 0.4
    _static_pose = None
    _CAM_MIN = (_RMIN[0] + _CAM_INSET, _RMIN[1] + _CAM_INSET)
    _CAM_MAX = (_RMAX[0] - _CAM_INSET, _RMAX[1] - _CAM_INSET)
    if _cam_mode == "static":
        # One fixed shot, chosen once from the layout: centre on everything that matters, then
        # search the azimuths for the viewpoint that both fits inside the room and stands furthest
        # off the walls -- a fixed camera that clips a wall renders the flat grey the follow-cam
        # clamp exists to avoid.
        _pts = [robot.data.root_pos_w[0].cpu().numpy()] + _frame_points()
        _c = np.mean(_pts, axis=0)
        _rad = max(float(np.linalg.norm(np.asarray(_p) - _c)) for _p in _pts)
        _dist = 3.0 + 1.1 * _rad
        _best, _best_margin = None, -1e9
        for _k in range(24):
            _th = 2.0 * math.pi * _k / 24.0
            _cx = float(_c[0] + _dist * math.cos(_th))
            _cy = float(_c[1] + _dist * math.sin(_th))
            _margin = min(_cx - _CAM_MIN[0], _CAM_MAX[0] - _cx, _cy - _CAM_MIN[1], _CAM_MAX[1] - _cy)
            if _margin > _best_margin:
                _best, _best_margin = (_cx, _cy), _margin
        _cx = min(max(_best[0], _CAM_MIN[0]), _CAM_MAX[0])
        _cy = min(max(_best[1], _CAM_MIN[1]), _CAM_MAX[1])
        _cz = float(max(2.2, _c[2] + 0.6 * _rad))  # above the fixture, looking down at it
        _static_pose = ((_cx, _cy, _cz), (float(_c[0]), float(_c[1]), float(_c[2])))
        print(
            f"[sonic] static camera at ({_cx:.2f},{_cy:.2f},{_cz:.2f}) "
            f"looking at ({_c[0]:.2f},{_c[1]:.2f},{_c[2]:.2f})  wall margin {_best_margin:.2f} m",
            flush=True,
        )
    elif _cam_mode == "fixture":
        # Side view of the socket. Standing on the ROBOT's side of the socket put the robot and
        # the ladder in the line of sight (S10: socket hidden behind both), and aiming at the
        # mid-drop height left the socket clipped at the top edge. So: stand off to the SIDE of
        # the robot->socket line, 4 m out, level with 1.5 m and aiming there, which puts the
        # socket (2.2 m) ten degrees above centre and the floor beneath it in the bottom of frame.
        # Of the two sides, take the one that lands further from the walls.
        _sk = env.scene["socket"].data.root_pos_w[0].cpu().numpy()
        _rb = robot.data.root_pos_w[0].cpu().numpy()
        _dir = _rb[:2] - _sk[:2]
        _dir = _dir / (np.linalg.norm(_dir) + 1e-6)
        _best, _best_m = None, -1e9
        for _sgn in (1.0, -1.0):
            _px, _py = -_dir[1] * _sgn, _dir[0] * _sgn  # perpendicular
            _cx = float(_sk[0] + 4.0 * _px)
            _cy = float(_sk[1] + 4.0 * _py)
            _m = min(_cx - _CAM_MIN[0], _CAM_MAX[0] - _cx, _cy - _CAM_MIN[1], _CAM_MAX[1] - _cy)
            if _m > _best_m:
                _best, _best_m = (_cx, _cy), _m
        _cx = min(max(_best[0], _CAM_MIN[0]), _CAM_MAX[0])
        _cy = min(max(_best[1], _CAM_MIN[1]), _CAM_MAX[1])
        _static_pose = ((_cx, _cy, 1.5), (float(_sk[0]), float(_sk[1]), 1.5))
        print(
            f"[sonic] fixture camera at ({_cx:.2f},{_cy:.2f},1.50) side-on to the socket at "
            f"({_sk[0]:.2f},{_sk[1]:.2f},{_sk[2]:.2f}), wall margin {_best_m:.2f} m",
            flush=True,
        )

    def _side_pose(entity, aim_z, eye_z, dist=3.2):
        """Fixed shot of the robot and ``entity`` from the side of the line between them, aimed at
        their midpoint, so neither hides the other. Side chosen for wall clearance."""
        _tp = env.scene[entity].data.root_pos_w[0].cpu().numpy()
        _rb = robot.data.root_pos_w[0].cpu().numpy()
        _mid = (_rb + _tp) / 2.0
        _dir = _tp[:2] - _rb[:2]
        _dir = _dir / (np.linalg.norm(_dir) + 1e-6)
        _best, _best_m = None, -1e9
        for _sgn in (1.0, -1.0):
            _px, _py = -_dir[1] * _sgn, _dir[0] * _sgn
            _cx = float(_mid[0] + dist * _px)
            _cy = float(_mid[1] + dist * _py)
            _m = min(_cx - _CAM_MIN[0], _CAM_MAX[0] - _cx, _cy - _CAM_MIN[1], _CAM_MAX[1] - _cy)
            if _m > _best_m:
                _best, _best_m = (_cx, _cy), _m
        _cx = min(max(_best[0], _CAM_MIN[0]), _CAM_MAX[0])
        _cy = min(max(_best[1], _CAM_MIN[1]), _CAM_MAX[1])
        print(
            f"[sonic] side camera at ({_cx:.2f},{_cy:.2f},{eye_z}) on robot->{entity}, "
            f"aimed at ({_mid[0]:.2f},{_mid[1]:.2f},{aim_z}), wall margin {_best_m:.2f} m",
            flush=True,
        )
        return ((_cx, _cy, eye_z), (float(_mid[0]), float(_mid[1]), aim_z))

    if _cam_mode == "bench":
        # Robot + bulb + bench: the bench is under the bulb, so aiming at tabletop height frames it.
        _static_pose = _side_pose("fresh_bulb" if "fresh_bulb" in _rigids else "bulb", 0.85, 1.45)
    elif _cam_mode == "crate":
        _static_pose = _side_pose("bin", 0.45, 1.3)

    _follow_dir = [None]  # camera direction from the objects' centre, kept until a point leaves frame
    _HFOV, _VFOV = 41.0 * 0.9, 26.5 * 0.9  # 12 mm lens, half-angles, used at 90 %

    def _view_margin(eye, c, pts):
        """Smallest angular margin (deg) by which every point sits inside the frame; <0 = cut."""
        fwd = c - eye
        fwd = fwd / (np.linalg.norm(fwd) + 1e-6)
        right = np.cross(fwd, [0.0, 0.0, 1.0])
        right = right / (np.linalg.norm(right) + 1e-6)
        up = np.cross(right, fwd)
        m = 1e9
        for p in pts:
            v = np.asarray(p) - eye
            az = math.degrees(math.atan2(float(v @ right), float(v @ fwd)))
            el = math.degrees(math.atan2(float(v @ up), float(v @ fwd)))
            m = min(m, _HFOV - abs(az), _VFOV - abs(el))
        return m

    def _follow_pose():
        """Third-person shot that keeps the robot AND the task's objects in frame.

        The room, not the geometry, decides where a camera can stand: a fixed rule (behind the
        robot, or beside the robot->objects line) gets clamped by a wall and cuts something off.
        So search: 24 directions around the objects' centre, each at the longest distance the
        room allows, scored by the smallest angular margin any point has inside the 12 mm lens's
        field. The winning direction is kept frame to frame and only re-searched when a point
        actually leaves the frame, so the shot does not swing as the robot moves.
        """
        _b = robot.data.root_pos_w[0].cpu().numpy()
        _pts = [_b] + _frame_points()
        _c = np.mean(_pts, axis=0)
        _rad = max(float(np.linalg.norm(np.asarray(_p) - _c)) for _p in _pts)
        _ez = float(_c[2] + 0.6)

        def _eye_for(_th, _dist):
            _ex = min(max(float(_c[0] + math.cos(_th) * _dist), _CAM_MIN[0]), _CAM_MAX[0])
            _ey = min(max(float(_c[1] + math.sin(_th) * _dist), _CAM_MIN[1]), _CAM_MAX[1])
            return np.array([_ex, _ey, _ez])

        _want = max(3.0, _rad / 0.70 + 0.5)
        if _follow_dir[0] is not None and _view_margin(_eye_for(_follow_dir[0], _want), _c, _pts) > 0.0:
            _eye = _eye_for(_follow_dir[0], _want)
        else:
            _best, _best_m = 0.0, -1e9
            for _k in range(24):
                _th = 2.0 * math.pi * _k / 24.0
                _m = _view_margin(_eye_for(_th, _want), _c, _pts)
                if _m > _best_m:
                    _best, _best_m = _th, _m
            _follow_dir[0] = _best
            _eye = _eye_for(_best, _want)
        return ((float(_eye[0]), float(_eye[1]), _ez), (float(_c[0]), float(_c[1]), float(_c[2])))

    def _video_pose():
        """The pose to capture the third-person video from this frame, whatever --camera says."""
        return _static_pose if _static_pose is not None else _follow_pose()

    _probe_seen: dict = {}

    def _probe_extras():
        """Per-object arm contact forces (N) as bag columns, over _probe_names.

        Also NAMES any touch of something that should never be touched -- the socket, the room
        walls, the fixture, a bulb. Those are failures whether or not the thing moves (the socket
        is bolted in place, so it cannot move and a graze would otherwise leave no trace), and the
        summed column cannot say which link did it.
        """
        _out = {}
        if not _probe_names:
            return _out
        for _side in ("r", "l"):
            _sensor = env.scene.sensors.get(f"probe_{_side}") if hasattr(env.scene, "sensors") else None
            if _sensor is None:
                continue
            _fm = _sensor.data.force_matrix_w  # (envs, bodies, targets, 3)
            _out[f"probe_{_side}_force"] = _fm.sum(dim=1).norm(dim=-1)[0].cpu().numpy().astype(np.float32)[None]
            _per = _fm.norm(dim=-1)[0]  # (bodies, targets)
            for _ti, _tn in enumerate(_probe_names):
                if _tn == "ladder" or _ti >= _per.shape[1]:
                    continue
                _bi = int(_per[:, _ti].argmax())
                _f = float(_per[_bi, _ti])
                if _f >= 2.0 and _probe_seen.get((_side, _tn), 0.0) < _f:
                    _probe_seen[(_side, _tn)] = _f
                    print(f"[sonic] TOUCH: {_sensor.body_names[_bi]} -> {_tn} at {_f:.0f} N", flush=True)
        return _out

    def _settle_step(_arm):
        """env.step during the settle, recorded when --record-settle asks for it.

        The recorder normally only sees the main loop, so the settle -- where a staged payload
        is actually lost -- never appears in a bag. Tag these rows in extras so a reader can
        split "settle" from "operator" without guessing at step indices.
        """
        _out = env.step(_arm)
        if args.record_settle and recorder is not None and recording_on:
            # Same extras as the main loop, on every row: a column present on some rows only is
            # stored shorter than the state columns, not padded, so reading it by row index puts
            # it 90 rows early. Settle rows are simply the first ones -- 40 pin + 50 SONIC warm =
            # 90 -- before the main loop; cmds are zero and sonic_action is whatever SONIC last
            # produced (zeros during the pin).
            _settle_extras = {
                "loco_cmd": np.asarray(loco_cmd, dtype=np.float32)[None],
                "rpy_cmd": np.asarray(rpy_cmd, dtype=np.float32)[None],
                "sonic_action": np.asarray(last_action, dtype=np.float32)[None],
            }
            if _rail is not None:
                _settle_extras["rail_contact_force"] = _rail_force().cpu().numpy().astype(np.float32)[None]
            _settle_extras.update(_probe_extras())
            recorder.record_step(_out[0], _arm, _out[1], _out[2], _out[3], extras=_settle_extras)
            if video is not None:
                video.capture(pose=_video_pose())
            if ego_video is not None:
                ego_video.capture()
        return _out

    # ---- RAIL HAND (FIATLUX_RAIL_HOLD auto/1/0, see the scene cfg above): a third support point ----
    # On the on-ladder stance the tread leaves ~12 cm behind the heel and SONIC's backward
    # catch-glide is ~15 cm, so roughly a quarter of idle spawns topple off the back (ladder-diag
    # batches: 29% inspire / 27% dex3; every stance/height/pin change tried was worse). A person
    # on a platform ladder keeps a hand on the top cap. This stages the LEFT hand hanging flat
    # over the cap's FAR edge -- an open "paddle", fingers down, palm toward the robot -- while
    # the base is still pinned, so it is engaged before the glide starts. A backward slide then
    # presses the cap's far face into the palm and the straight fingers (loaded into their 0-rad
    # limit, so the 0.1 N.m finger motors are not what holds), and the load goes up the arm's PD:
    # real contact through real joints, nothing filtered. The wrist IK target stays ROOT-relative
    # (a stiffened arm), which is what makes the hand a brace rather than a hand that merely
    # rides along. Geometry (AlumStep_D, ladder frame, m): top cap z 1.80-1.86, y 0.155-0.27,
    # full width; the robot faces +y with its left at -x; the pin holds the pelvis ~0.07 above
    # SONIC's stance, so the hand is staged that much higher and settles onto the face on release.
    _rail = None
    if _rail_on:
        if "ladder" not in getattr(env.scene, "rigid_objects", {}):
            print("[sonic] RAIL HAND requested but this scene has no ladder; ignored", flush=True)
        else:
            from isaaclab.utils.math import quat_apply, quat_inv, quat_mul  # noqa: E402

            _ladder = env.scene["ladder"]
            _lw_bid = robot.body_names.index("left_wrist_yaw_link")

            # Staged wrist height per hand: the fingertips must end ~2 cm ABOVE the cap plate
            # (top z 1.86) before the release drop, or the approach sweeps them through it and
            # the arm jams behind the cap (inspire probe 1). Fingertip reach below the wrist:
            # dex3 0.165 m, inspire 0.21 m; plus the ~6 cm the DLS IK stalls short of the target.
            _RAIL_WRIST_L = (-0.13, 0.30, 2.11) if args.hand.lower() == "inspire" else (-0.13, 0.30, 2.05)
            _RAIL_APPROACH_STEPS, _RAIL_DWELL_STEPS = 30, 10  # pinned steps to reach the cap, then settle
            _rail = {
                # wrist target in the LADDER frame, staged (pre-release) height
                "wrist_l": torch.tensor([_RAIL_WRIST_L], device=dev),
                # wrist orientation in the LADDER frame (w,x,y,z): the paddle, R_y(+90 deg) =
                # fingers straight down, palm side (-y_wrist on both hands) toward the robot.
                "quat_l": torch.tensor([[0.7071068, 0.0, 0.7071068, 0.0]], device=dev),
                "approach": _RAIL_APPROACH_STEPS,
                "dwell": _RAIL_DWELL_STEPS,
                # The brace HOLDS until the operator takes the arm (keyboard H, VR left grip). There
                # is no timed return to the rest pose: the hand stays on the ladder and control
                # passes to the operator from wherever it is, with no repositioning in between.
                "phase": "idle",  # idle -> approach -> hold ; the operator toggle -> idle
                "i": 0,
                "from": None,  # root-frame L pose the approach starts from (= the spawn rest pose)
                "to": None,  # root-frame L pose it ends at (= the hold target)
                "rest": None,  # the spawn rest pose, kept so a retract has somewhere to go
                "cap": None,  # the (re-anchored) cap pose, kept so a re-brace has somewhere to go
                "held": True,  # operator toggle (keyboard H / VR): False hands the left arm back
                "staged": True,  # False once a spawn's approach failed its check (brace off for it)
            }

            # The dex3 thumb stands 6 cm proud of the palm when open, straight at the cap top once
            # the hand hangs over the far edge. Fold it across the palm (its grasp preset) as part
            # of the OPEN pose so the paddle is the flat palm + straight fingers only.
            _lterm = env.action_manager.get_term("left_hand_action")
            _lthumb_open = _lterm._open_command.clone()  # the REAL open pose, kept to put back
            _lnames_hand = [robot.joint_names[i] for i in _lterm._joint_ids]
            _lthumb_idx = [_k for _k, _n in enumerate(_lnames_hand) if "thumb" in _n]

            def _rail_thumb_fold():
                """The paddle: thumb across the palm as part of the OPEN pose (also on re-brace)."""
                for _k in _lthumb_idx:
                    _lterm._open_command[_k] = _lterm._close_command[_k]

            def _rail_thumb_restore():
                """Undo the fold when the operator takes the arm, or open and close stay the
                same pose and the hand can never let go."""
                _lterm._open_command.copy_(_lthumb_open)

            _rail_thumb_fold()
            print(
                f"[sonic] RAIL HAND: left thumb folded across the palm in the open pose ({len(_lthumb_idx)} joints)",
                flush=True,
            )

            def _rail_force():
                """Left hand's total force against the ladder (N), from the filtered sensor."""
                if "rail_contact" not in getattr(env.scene, "sensors", {}):
                    return torch.zeros(1, device=dev)
                _fm = env.scene["rail_contact"].data.force_matrix_w  # (N, B, M, 3)
                return _fm.sum(dim=2).norm(dim=-1).sum(dim=1)  # (N,)

            def _rail_target_b():
                """Ladder-frame wrist target -> ROOT frame (pose 7). Computed while pinned."""
                p_w = _ladder.data.root_pos_w + quat_apply(_ladder.data.root_quat_w, _rail["wrist_l"])
                q_w = quat_mul(_ladder.data.root_quat_w, _rail["quat_l"])
                p_b, q_b = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, p_w, q_w)
                return torch.cat([p_b[0], q_b[0]])

            def _rail_begin():
                _rail["phase"] = "approach"
                _rail["i"] = 0
                _rail["from"] = rest_arm_action()[8:15].clone()
                _rail["rest"] = _rail["from"].clone()
                _rail["to"] = _rail_target_b()
                _rail["held"] = True
                _rail["staged"] = True

            def _rail_staged_ok():
                """End of the pinned approach: did the hand land where it was sent?

                Three ways it does not, all seen: the arm pressing on the ladder (the elbow inside
                a side rail after the IK wandered off), the wrist parked well ABOVE its target, or
                the wrist stalled well BELOW its target with no ladder contact at all (an
                unreachable/stalled approach leaves the hand dangling in free air -- near-zero
                force, since it touches nothing). The normal DLS stall is only a few cm below
                target, so a small negative dz is fine; a large one means there is no brace.
                Any of these releasing the base now throws the robot (S11 inspire seed 4 in the
                VR batch: 1.1 m/s kick, ladder 58 cm), so the brace is worth less than nothing on
                that spawn.
                """
                f = float(_rail_force())
                dz = float(rest_arm_action()[10] - _rail["to"][2])
                ok = f < 20.0 and abs(dz) < 0.05
                if not ok:
                    print(
                        f"[sonic] RAIL HAND NOT STAGED (arm-ladder {f:.0f} N, wrist {dz * 100:+.0f} cm vs target): "
                        "retracting to the rest pose while pinned; brace OFF for this spawn",
                        flush=True,
                    )
                return ok

            def _rail_retract_pinned(pin, legs):
                """Bring the left arm back to its spawn rest pose over 20 steps, base still pinned."""
                _rail["from"] = rest_arm_action()[8:15].clone()
                _rail["to"] = _rail["rest"].clone()
                _rail["phase"], _rail["i"] = "approach", 0
                _rail["approach"], _rail["dwell"] = 20, 0
                for _ in range(20):
                    robot.set_joint_position_target(legs, joint_ids=act_idx)
                    robot.write_root_pose_to_sim(pin)
                    robot.write_root_velocity_to_sim(zero_vel)
                    _settle_step(_settle_arm())
                    _pin_settle_bulb()
                _rail["approach"], _rail["dwell"] = _RAIL_APPROACH_STEPS, _RAIL_DWELL_STEPS
                _rail["phase"], _rail["held"], _rail["staged"] = "idle", False, False

            def _rail_main_L():
                """Per main-loop step: the LEFT pose (7) to enforce while braced, or None (operator's arm)."""
                if _rail["held"]:
                    return _rail["to"]
                return None

            def _rail_release():
                """The operator takes the left arm: stop enforcing the cap pose and give the hand its
                real open pose back. The arm is handed over where it is -- on the ladder -- with no
                return to a rest pose first."""
                _rail["held"], _rail["phase"] = False, "idle"
                _rail_thumb_restore()

            def _rail_rebrace():
                """Operator asked for the brace back: aim at the cap pose again (the IK slews)."""
                if _rail["cap"] is not None:
                    _rail["to"] = _rail["cap"].clone()
                _rail_thumb_fold()
                _rail["phase"], _rail["held"] = "hold", True

            def _rail_settled():
                """After SONIC's settle-in: re-anchor the hold ONCE to where the hand actually rests.

                The staged target is 7 cm above the resting height (the release drop) and the IK
                stalls a few cm short of it anyway, so keeping it would leave the arm pressing the
                hand down onto the cap (~50-100 N in probe 2) for the whole session -- a steady
                nose-up moment and a chunk of the arm's effort budget. Anchoring to the settled
                pose leaves the brace unloaded until the body moves; one-time, not per frame (a
                per-frame re-anchor is the sag ratchet the settle comments warn about).
                """
                _live = rest_arm_action()[8:15].clone()
                _d = (_live[0:3] - _rail["to"][0:3]) * 100.0
                _rail["to"] = _live
                _lw_l = quat_apply(
                    quat_inv(_ladder.data.root_quat_w),
                    robot.data.body_state_w[:, _lw_bid, 0:3] - _ladder.data.root_pos_w,
                )[0]
                print(
                    f"[sonic] RAIL HAND re-anchored to the resting pose: wrist ladder-frame "
                    f"({_lw_l[0]:.3f},{_lw_l[1]:.3f},{_lw_l[2]:.3f}), "
                    f"moved ({_d[0]:+.1f},{_d[1]:+.1f},{_d[2]:+.1f}) cm "
                    f"in the root frame from the staged target; brace force {float(_rail_force()):.1f} N",
                    flush=True,
                )

            def _rail_arm(arm):
                """Replace the LEFT arm part of a rest arm action with the rail approach/hold."""
                if _rail["phase"] == "idle":
                    return arm
                if _rail["phase"] == "approach":
                    a = min(1.0, _rail["i"] / max(1, _rail["approach"]))
                    pose = (1 - a) * _rail["from"] + a * _rail["to"]
                    pose[3:7] = pose[3:7] / torch.linalg.norm(pose[3:7])
                    _rail["i"] += 1
                    if _rail["i"] >= _rail["approach"] + _rail["dwell"]:
                        _rail["phase"] = "hold"
                else:
                    pose = _rail["to"]
                arm = arm.clone()
                arm[8:15] = pose
                arm[15] = 1.0  # open paddle
                return arm

            print(
                f"[sonic] RAIL HAND: left wrist -> ladder-frame {_rail['wrist_l'][0].tolist()} "
                f"(ladder-frame quat {_rail['quat_l'][0].tolist()}), "
                f"{_rail['approach']} approach + {_rail['dwell']} dwell steps while pinned",
                flush=True,
            )

    def _settle_arm():
        _a = rest_arm_action()
        return (_rail_arm(_a) if _rail is not None else _a).repeat(env.num_envs, 1)

    # The rail approach starts AFTER the 40 plant steps, not before: the spawn's depenetration
    # kick can shove the free ladder 10-40 cm in the first physics step (seen in 3 of 84 baseline
    # takes and in rail-B-inspire seed3/try3), and a target computed from the pre-kick ladder pose
    # stages the hand behind the cap, where it pushes the robot backward instead of bracing it.
    # Place the left arm on the rest pose before the settle. default_joint_pos alone does not
    # reach it: the post-settle restore skips the left arm while the rail hand is staged
    # (_arm_idx), and the rail hand captures its return target from wherever the arm is.
    if _rail is not None and _ARM_REST_L.strip():
        _lnames = [
            f"left_{_n}_joint"
            for _n in (
                "shoulder_pitch",
                "shoulder_roll",
                "shoulder_yaw",
                "elbow",
                "wrist_roll",
                "wrist_pitch",
                "wrist_yaw",
            )
        ]
        if all(_n in robot.joint_names for _n in _lnames):
            _lids = [robot.joint_names.index(_n) for _n in _lnames]
            _jp_rest = robot.data.joint_pos.clone()
            _jp_rest[:, _lids] = robot.data.default_joint_pos[:, _lids]
            _carry_payload(lambda: robot.write_joint_state_to_sim(_jp_rest, torch.zeros_like(robot.data.joint_vel)))
            print("[sonic] left arm placed on the recorded rest pose before the settle", flush=True)

    _pin_steps = 40 + ((_rail["approach"] + _rail["dwell"]) if _rail is not None else 0)
    for _k in range(_pin_steps):
        if _rail is not None and _k == 40:
            _rail_begin()  # feet planted, ladder settled: aim at where the cap actually is
        robot.set_joint_position_target(leg_default, joint_ids=act_idx)
        robot.write_root_pose_to_sim(pin_pose)  # hold base upright while feet plant
        robot.write_root_velocity_to_sim(zero_vel)
        _settle_step(_settle_arm())
        _pin_settle_bulb()  # carry the seated bulb through the grip-close transient
    if _rail is not None and not _rail_staged_ok():
        _rail_retract_pinned(pin_pose, leg_default)
    robot.write_root_pose_to_sim(pin_pose)  # final: level + still, then release to SONIC
    robot.write_root_velocity_to_sim(zero_vel)
    obs_hist = collections.deque([build_obs()] * HIST_LEN, maxlen=HIST_LEN)  # warm history w/ real state

    # SONIC settle-in: let it center its weight (the one-time "back up") NOW, before the operator
    # connects. Short (faster-ready pref) -> a bit of settle-in movement may still show on connect.
    for _s in range(50):
        obs_hist.append(build_obs())
        flat = np.concatenate(obs_hist).astype(np.float32)[None]
        last_action = bal_sess.run(None, {in_name: flat})[0][0]
        leg_target = torch.as_tensor(last_action * ACTION_SCALE + DEFAULT_15, device=dev)
        robot.set_joint_position_target(leg_target.unsqueeze(0), joint_ids=act_idx)
        _settle_step(_settle_arm())
        _pin_settle_bulb()  # keep carrying through SONIC settle-in, released after this loop
    spawn_root = robot.data.root_state_w[:, 0:7].clone()  # centered pose = re-home + hold target
    home_xy = spawn_root[0, 0:2].cpu().numpy().copy()
    # Capture the hold-arms IK target ONCE, now, after the settle -- re-solving it every main-loop
    # frame lets the redundant null-space drift the arm; one fixed target keeps it steady.
    # The settle plants the FEET, but it also lets the arm sag: SONIC drives the legs while the
    # arm merely holds whatever IK target it had, and the redundant null-space bows the shoulder
    # (measured ~0.4 rad of droop at the right shoulder on the carry tasks). That moves the open
    # palm the payload is staged to rest on. Put the arm back on the pose the task authored, then
    # capture the IK target FROM that pose so the hold does not pull it back down.
    # (Rail hand: the LEFT arm is where it was staged -- on the cap -- so only the right arm is
    # put back; a state write would yank the brace off the rail. A spawn whose brace was retracted
    # by the staging check gets BOTH arms restored, like a stock spawn: its left arm sagged through
    # the SONIC settle-in and, left there, rested on the cap at ~19 N -- guard test 2.)
    _rail_braced = _rail is not None and _rail["staged"]
    _arm_idx = torch.tensor(
        [
            i
            for i, n in enumerate(robot.joint_names)
            if any(k in n for k in ("shoulder", "elbow", "wrist")) and (not _rail_braced or n.startswith("right_"))
        ],
        device=dev,
    )
    _staged_arm = robot.data.default_joint_pos[:, _arm_idx].clone()
    _jp = robot.data.joint_pos.clone()
    _jp[:, _arm_idx] = _staged_arm
    _carry_payload(lambda: robot.write_joint_state_to_sim(_jp, torch.zeros_like(robot.data.joint_vel)))
    print(
        f"[sonic] arm restored to the staged pose (settle droop removed: "
        f"{float((robot.data.default_joint_pos[:, _arm_idx] - _staged_arm).abs().max()):.3f} rad)",
        flush=True,
    )

    rest_arm = rest_arm_action()
    if _rail is not None and _rail["staged"]:
        _rail_settled()
        _rail["cap"] = _rail["to"].clone()
        rest_arm[8:15] = _rail["to"]
        rest_arm[15] = 1.0
        _lw_l = quat_apply(
            quat_inv(_ladder.data.root_quat_w), robot.data.body_state_w[:, _lw_bid, 0:3] - _ladder.data.root_pos_w
        )[0]
        print(
            f"[sonic] RAIL HAND staged: left wrist at ladder-frame ({_lw_l[0]:.3f},{_lw_l[1]:.3f},{_lw_l[2]:.3f}) "
            f"vs target {_rail['wrist_l'][0].tolist()} (pre-release); H toggles the brace",
            flush=True,
        )

    # OPERATOR GRASP (Inspire on the thumb-fix asset): the env's close preset is the settle-time
    # support curl (fingers ~0.3 rad) that keeps the seated bulb from being squeezed while SONIC
    # settles in. From "Teleop ready" the operator's close is the NORMAL full grasp preset, same
    # as the left hand -- the fingers simply stop where the bulb stops them.
    if _right_starts_closed and args.hand == "inspire" and os.environ.get("FIATLUX_TELEOP_ROBOT_USD"):
        from fiatlux_task.robots.g1 import G1_HAND_GRASP  # noqa: E402

        # Thumb bend joints go to their full range instead of the preset's partial curl: with the
        # re-authored rotation the thumb stands over the palm at the preset yaw, and the partial
        # bend left it visibly half-closed on an empty close. Full bend keeps the same opposed
        # rotation and simply curls all the way (or stops on the bulb).
        _thumb_full = {
            "R_thumb_proximal_pitch_joint": 0.6,
            "R_thumb_intermediate_joint": 0.8,
            "R_thumb_distal_joint": 1.2,
        }
        _hterm = env.action_manager.get_term("hand_action")
        for _k, _n in enumerate(robot.joint_names[i] for i in _hterm._joint_ids):
            _hterm._close_command[_k] = _thumb_full.get(_n, G1_HAND_GRASP[_n])
        # Same full thumb bend on the LEFT close (its limits: pitch 0.5, intermediate 0.8,
        # distal 1.2), so both hands close alike.
        _lthumb_full = {
            "L_thumb_proximal_pitch_joint": 0.5,
            "L_thumb_intermediate_joint": 0.8,
            "L_thumb_distal_joint": 1.2,
        }
        _lterm = env.action_manager.get_term("left_hand_action")
        for _k, _n in enumerate(robot.joint_names[i] for i in _lterm._joint_ids):
            if _n in _lthumb_full:
                _lterm._close_command[_k] = _lthumb_full[_n]
        print("[sonic] full grasp restored for the operator (settle used the support curl)", flush=True)

    # STAGED CLOSE (Inspire): with self-collisions on, driving the fingers and the thumb to the
    # fist preset simultaneously wedges the fingertips on the thumb tip mid-flight -- the close
    # jams into a hollow "beak" (thumb shoved off its pose) and a tabletop bulb is squeezed out
    # instead of enveloped. Close the way a hand actually makes a fist: one group leads, the
    # other folds in against it once it has landed. A close ON the bulb is unaffected -- the
    # leading group simply stops on the glass and the trailing group clamps.
    # Sim steps (20 ms each) the trailing group waits after the leading one starts. 2 = 40 ms:
    # enough head start that the tips don't meet edge-on mid-flight, short enough that the close
    # feels like one motion in VR.
    _STAGE_STEPS = max(1, int(os.environ.get("FIATLUX_TELEOP_STAGE_STEPS", "2")))
    _staged_close = None
    if args.hand == "inspire":
        _lead_thumb = os.environ.get("FIATLUX_TELEOP_STAGE", "fingers") == "thumb"
        _staged_close = []
        for _term_name, _grip_col, _closed0 in (
            ("hand_action", 7, _right_starts_closed),
            ("left_hand_action", 15, False),
        ):
            _term = env.action_manager.get_term(_term_name)
            _full = _term._close_command.clone()
            _lead = _term._close_command.clone()
            for _k, _n in enumerate(robot.joint_names[i] for i in _term._joint_ids):
                _is_thumb_bend = "thumb" in _n and "yaw" not in _n
                if _is_thumb_bend != _lead_thumb:  # the trailing group waits at its open pose
                    _lead[_k] = _term._open_command[_k]
            _staged_close.append(
                {"term": _term, "col": _grip_col, "full": _full, "lead": _lead, "closed": _closed0, "since": 10**6}
            )
        print(
            f"[sonic] staged close armed ({'thumb' if _lead_thumb else 'fingers'} lead, "
            f"{_STAGE_STEPS * 20} ms stagger)",
            flush=True,
        )

        # REAL-HAND TORQUE CAP: the RH56DFTP's fingertips top out around 10 N (~0.3-0.5 N.m at
        # the joint); the cfg's 2.0 N.m ceiling lets every blocked joint press with 4x that.
        # With self-collisions on, a held bulb closes a finger->bulb->thumb->palm force loop of
        # saturated PD torques and the vibration tips SONIC (2 of 3 hands-off settles fell).
        # Cap the finger joints at the hardware figure so a blocked close rests instead of
        # grinding -- the same force-bounded stop the real hand's FORCE_SET gives.
        _hand_eff = os.environ.get("FIATLUX_TELEOP_HAND_EFFORT", "0.6")
        if _hand_eff != "off":
            _hand_ids = [i for i, n in enumerate(robot.joint_names) if n[:2] in ("R_", "L_")]
            robot.write_joint_effort_limit_to_sim(
                torch.full((1, len(_hand_ids)), float(_hand_eff), device=dev), joint_ids=_hand_ids
            )
            print(f"[sonic] hand effort limit capped at {_hand_eff} N.m ({len(_hand_ids)} joints)", flush=True)

    if args.input == "vr":
        # Re-anchor the controller_rel arm retargeters to THIS scene's live robot. They default to the
        # Insert *table* world coords, so on any other scene (e.g. the ladder Carry env) the arm reaches
        # for a world point far from the robot and flails. Rebake root + EE-start + workspace to live.
        from scipy.spatial.transform import Rotation as _Rot  # noqa: E402

        _rpos = robot.data.root_pos_w[0].cpu().numpy()
        _rq = robot.data.root_quat_w[0].cpu().numpy()  # w, x, y, z
        _R = _Rot.from_quat([_rq[1], _rq[2], _rq[3], _rq[0]])
        for _rt in getattr(teleop, "_retargeters", None) or []:
            if not hasattr(_rt, "_root_pos"):  # only the Se3Rel arm retargeters
                continue
            _rt._root_pos = _rpos.astype(np.float32)
            _rt._root_R = _R
            _rt._root_R_T = _R.as_matrix().T.astype(np.float32)
            _right = getattr(_rt, "_target", None) == DeviceBase.TrackingTarget.CONTROLLER_RIGHT
            _eeb = robot.body_names.index("right_wrist_yaw_link" if _right else "left_wrist_yaw_link")
            _ee_w = robot.data.body_state_w[0, _eeb, 0:3].cpu().numpy().astype(np.float32)
            _ee = (_R.as_matrix().T @ (_ee_w - _rpos)).astype(np.float32)  # EE in the BASE frame (target lives there)
            _rt._init_pos = _ee.copy()
            _rt._pos = _ee.copy()
            _rt._lo = _ee - np.array([0.45, 0.45, 0.45], dtype=np.float32)
            _rt._hi = _ee + np.array([0.45, 0.45, 0.45], dtype=np.float32)
            _rt._prev = None
            _rt._smooth = None
            # Orientation too: the cfg's initial_orientation is the Insert-table rest quat, so the
            # first clutch snapped the wrist there (e.g. off the carry staging's palm-up pose --
            # keyboard, which holds the captured settle pose, never did this). Start the rotation
            # ratchet from the LIVE wrist orientation instead, root frame like the command.
            _ee_q = robot.data.body_state_w[0, _eeb, 3:7].cpu().numpy()  # w, x, y, z
            _live_R = _R.inv() * _Rot.from_quat([_ee_q[1], _ee_q[2], _ee_q[3], _ee_q[0]])
            _rt._init_R = _live_R
            _rt._quat_R = _live_R

    # ---- keyboard input (desktop, no headset): full VR parity -- BOTH arms (position + wrist
    # rotation + grip), walk, and lean. TAB switches which arm the manipulation keys drive. ----
    kb = None
    if args.input == "keyboard":
        _pressed = collections.deque()
        _scripted_keys = sorted(
            (float(spec.split("@")[1]), spec.split("@")[0].strip().upper())
            for spec in args.keys.split(",")
            if spec.strip()
        )
        try:
            import carb  # noqa: E402
            import omni.appwindow  # noqa: E402

            _kbd_iface = carb.input.acquire_input_interface()
            _kbd = omni.appwindow.get_default_app_window().get_keyboard()

            def _on_key(e):
                if e.type == carb.input.KeyboardEventType.KEY_PRESS:
                    _pressed.append(e.input.name)
                return True

            _kbd_iface.subscribe_to_keyboard_events(_kbd, _on_key)
        except Exception as _e:  # noqa: BLE001  (headless / no window -> just run SONIC with no input)
            print(f"[sonic] keyboard listener unavailable ({_e}); running with no input", flush=True)

        _box = torch.tensor([0.30, 0.30, 0.35], device=dev)  # per-arm reach half-extent
        kb = {
            "active": "R",  # arm the manip keys drive
            "R_ee": rest_arm[0:7].clone(),
            "L_ee": rest_arm[8:15].clone(),  # per-arm EE target (root frame)
            "R_lo": rest_arm[0:3] - _box,
            "R_hi": rest_arm[0:3] + _box,
            "L_lo": rest_arm[8:11] - _box,
            "L_hi": rest_arm[8:11] + _box,
            "R_grip_open": not _right_starts_closed,
            "L_grip_open": True,
            "lean": 0.0,
            "quit": False,
        }
        _POS_KEYS = {
            "W": (0, 0.02),
            "S": (0, -0.02),
            "A": (1, 0.02),  # key -> (xyz axis, dpos m)
            "D": (1, -0.02),
            "Q": (2, 0.02),
            "E": (2, -0.02),
        }
        _ROT_KEYS = {
            "U": (0, 0.10),
            "O": (0, -0.10),
            "I": (1, 0.10),  # key -> (axis, dangle rad)
            "K": (1, -0.10),
            "J": (2, 0.10),
            "L": (2, -0.10),
        }
        _WALK_KEYS = {
            "UP": (0, 0.1),
            "DOWN": (0, -0.1),
            "LEFT": (2, 0.1),  # key -> (vx/vy/wz idx, step)
            "RIGHT": (2, -0.1),
            "COMMA": (1, 0.1),
            "PERIOD": (1, -0.1),
        }
        _AXES = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

        def _qmul(a, b):  # wxyz Hamilton product (single quats)
            return torch.stack(
                [
                    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
                    a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
                    a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
                    a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
                ]
            )

        def _rotate(q, axis_i, ang):  # pre-multiply q by a root-axis rotation
            ax = _AXES[axis_i]
            s = math.sin(ang / 2.0)
            dq = torch.tensor([math.cos(ang / 2.0), ax[0] * s, ax[1] * s, ax[2] * s], device=dev)
            out = _qmul(dq, q)
            return out / torch.linalg.norm(out)

        def kb_drain():
            while _pressed:
                k = _pressed.popleft()
                a = kb["active"]
                if k in _POS_KEYS:
                    i, d = _POS_KEYS[k]
                    kb[a + "_ee"][i] += d
                elif k in _ROT_KEYS:
                    i, d = _ROT_KEYS[k]
                    kb[a + "_ee"][3:7] = _rotate(kb[a + "_ee"][3:7], i, d)
                elif k in _WALK_KEYS:
                    i, d = _WALK_KEYS[k]
                    loco_cmd[i] += d
                elif k == "SPACE":
                    loco_cmd[:] = 0.0
                elif k == "G":
                    kb[a + "_grip_open"] = not kb[a + "_grip_open"]
                elif k == "TAB":
                    kb["active"] = "L" if a == "R" else "R"
                    print(f"[sonic] active arm -> {kb['active']}", flush=True)
                elif k == "T":
                    kb["lean"] = min(kb["lean"] + 0.2, 1.0)
                elif k == "Y":
                    kb["lean"] = max(kb["lean"] - 0.2, -1.0)
                elif k == "C":
                    rec_flag["toggle"] = True
                elif k == "R":
                    _reset()
                elif k == "H" and _rail is not None:
                    if _rail["held"]:
                        _rail_release()
                    else:  # back onto the rail: re-aim the left target at the cap pose
                        _rail_rebrace()
                        kb["L_ee"] = _rail["to"].clone()
                        kb["L_grip_open"] = True
                    print(f"[sonic] rail hand {'ON' if _rail['held'] else 'OFF (left arm is yours)'}", flush=True)
                elif k == "ESCAPE":
                    kb["quit"] = True
            kb["R_ee"][0:3] = torch.clamp(kb["R_ee"][0:3], kb["R_lo"], kb["R_hi"])
            kb["L_ee"][0:3] = torch.clamp(kb["L_ee"][0:3], kb["L_lo"], kb["L_hi"])
            np.clip(loco_cmd, [-0.8, -0.5, -1.0], [1.0, 0.5, 1.0], out=loco_cmd)
            rpy_cmd[1] = kb["lean"] * LEAN_MAG

        def kb_arm_action():
            # env action = [R pose(7), R grip(1), L pose(7), L grip(1)] -- both arms driven from keys.
            rg = torch.ones(1, device=dev) if kb["R_grip_open"] else -torch.ones(1, device=dev)
            lg = torch.ones(1, device=dev) if kb["L_grip_open"] else -torch.ones(1, device=dev)
            return torch.cat([kb["R_ee"], rg, kb["L_ee"], lg])

        print(
            "KEYBOARD ready (bimanual).  TAB = switch active arm (R/L).  active arm: W/S A/D Q/E = move "
            "X/Y/Z, U/O I/K J/L = roll/pitch/yaw, G = grip.  walk: arrows (UP/DOWN fwd, LEFT/RIGHT turn), "
            ",/. strafe, T/Y lean, SPACE stop.  R reset, C record on/off, ESC quit.",
            flush=True,
        )

    _elb_i = robot.joint_names.index("right_elbow_joint")
    if abs(float(spawn_root[0, 2]) - float(staged_root[0, 2])) > 0.05:
        print(
            f"[sonic] NOTE: settled {float(staged_root[0, 2]) - float(spawn_root[0, 2]):.2f} m below "
            f"the staged pose ({float(staged_root[0, 2]):.2f} m); [R] re-homes to the STAGED pose",
            flush=True,
        )
    print(
        f"[sonic] settled+centered at pelvis=({spawn_root[0, 0]:.2f},{spawn_root[0, 1]:.2f},"
        f"{spawn_root[0, 2]:.2f}) right_elbow={float(robot.data.joint_pos[0, _elb_i]):.2f}rad "
        f"(target 1.57 = 90deg)",
        flush=True,
    )
    # DIAGNOSTIC: full arm joint values, right vs left, to see the asymmetric IK elbow resolution.
    _dbg_j = [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
    ]
    _dbg_v = robot.data.joint_pos[0, [robot.joint_names.index(j) for j in _dbg_j]].cpu().numpy()
    print(
        "[sonic] arm joints  R[sp,sr,sy,elb]="
        + ",".join(f"{v:+.2f}" for v in _dbg_v[:4])
        + "  L[sp,sr,sy,elb]="
        + ",".join(f"{v:+.2f}" for v in _dbg_v[4:]),
        flush=True,
    )
    if args.input == "vr":
        print(
            "Teleop ready. In the Isaac Sim UI: AR panel -> Start AR, then connect the Pico. "
            "LEFT stick = walk, RIGHT stick X = turn, RIGHT btn = stop, LEFT X/Y = lean. "
            "Arms: the usual controller_rel teleop (grip-clutch + move, trigger to grasp)."
            + (
                " LEFT HAND IS ON THE LADDER CAP (rail brace): squeeze the LEFT grip to take the "
                "left arm back, press both LEFT face buttons to put it back on the cap."
                if _rail is not None
                else ""
            ),
            flush=True,
        )
    else:
        print(
            "Teleop ready (keyboard). Click the Isaac Sim viewport to focus it, then use the keys "
            "listed above (TAB switches arm; W/S A/D Q/E move + U/O I/K J/L rotate; arrows walk)."
            + (" LEFT HAND IS ON THE LADDER CAP (rail brace): H toggles it." if _rail is not None else ""),
            flush=True,
        )

    def resettle():
        """Re-plant the free base after a re-home, using the SAME pin-upright + SONIC settle-in as
        startup. A reset teleports the root back to the spawn, but if we then hand the robot straight
        to SONIC from a cold (zeroed) obs history it lurches to "catch" itself and flies/faceplants.
        Pinning it level + zero-velocity while the feet re-plant, then warming SONIC in place, hands
        balance a level, still robot -- the same reason startup settles before "Teleop ready"."""
        nonlocal obs_hist, last_action, home_xy, last_arm
        last_action = np.zeros(N_ACT, dtype=np.float32)  # start SONIC's action history clean, like startup
        pin = staged_root.clone()
        # Pin the pelvis at the RAISED spawn height (>=0.80), not the settled ~0.74, so the feet
        # re-plant WITH clearance -- exactly what startup does (init_state z is raised to 0.80). Pinning
        # at the settled height drops the feet onto/through the floor and the depenetration kick, plus a
        # cold SONIC catch, is what tipped the robot over "randomly" on reset.
        pin[:, 2] = max(float(staged_root[0, 2]), 0.80)
        zv = torch.zeros((env.num_envs, 6), device=dev)
        ld = torch.as_tensor(DEFAULT_15, device=dev).unsqueeze(0)

        def _rehome_writes():
            robot.write_joint_state_to_sim(
                robot.data.default_joint_pos.clone(), torch.zeros_like(robot.data.default_joint_vel)
            )
            robot.write_root_pose_to_sim(pin)
            robot.write_root_velocity_to_sim(zv)

        _carry_payload(_rehome_writes)  # joints AND root move the wrist: the payload rides along
        # Reset re-settles like startup, so re-apply the pre-grasp here too (else [R]/RESET drops the bulb).
        if _rail is not None:
            _rail["phase"] = "idle"  # the re-home put the left arm back on the staged pose: hold it
        for _k in range(_pin_steps):  # pin level while feet plant
            if _rail is not None and _k == 40:
                _rail_begin()  # re-stage the brace once the feet and the ladder have settled
            robot.set_joint_position_target(ld, joint_ids=act_idx)
            robot.write_root_pose_to_sim(pin)
            robot.write_root_velocity_to_sim(zv)
            env.step(_settle_arm())  # re-read (like startup): symmetric elbows
            _pin_settle_bulb()  # carry the re-seated bulb through the grip-close transient
        if _rail is not None and not _rail_staged_ok():
            _rail_retract_pinned(pin, ld)
        robot.write_root_pose_to_sim(pin)
        robot.write_root_velocity_to_sim(zv)
        obs_hist = collections.deque([build_obs()] * HIST_LEN, maxlen=HIST_LEN)  # warm w/ real state
        for _ in range(50):  # SONIC settles its weight in place
            obs_hist.append(build_obs())
            flat = np.concatenate(obs_hist).astype(np.float32)[None]
            # CRITICAL: update the *nonlocal* last_action every step (a throwaway local left the obs's
            # "previous action" channel stale, so SONIC's history was incoherent, it never balanced, and
            # it fell the instant the pin released). This is what startup does.
            last_action = bal_sess.run(None, {in_name: flat})[0][0]
            lt = torch.as_tensor(last_action * ACTION_SCALE + DEFAULT_15, device=dev)
            robot.set_joint_position_target(lt.unsqueeze(0), joint_ids=act_idx)
            env.step(_settle_arm())  # re-read (like startup): symmetric elbows
            _pin_settle_bulb()  # keep carrying through the re-settle, released after this loop
        home_xy = robot.data.root_pos_w[0, 0:2].cpu().numpy().copy()  # hold where it actually stands
        if _rail is not None and _rail["staged"]:
            _rail_settled()
            _rail["cap"] = _rail["to"].clone()
            rest_arm[8:15] = _rail["to"]
            rest_arm[15] = 1.0
        elif _rail is not None:
            rest_arm[8:15] = _rail["rest"]
            rest_arm[15] = 1.0
        last_arm = rest_arm

    # Room interior for the follow-cam, inset from the walls (ROOM_FLOOR_MIN/MAX in scene_cfg).

    n_walk = 8
    vr_rec_prev = False  # rising-edge detect for the VR record-toggle button
    vr_rail_prev = [False, False]  # rising-edge detect: left squeeze (take the arm), both left buttons (re-brace)

    def _vr_take_left_arm():
        """Hand the left arm to the controller without a jump: re-reference the left Se3Rel
        retargeter to the LIVE wrist pose (its own target is stale from before the brace)."""
        if args.input != "vr":
            return
        _rq_ = robot.data.root_quat_w[0].cpu().numpy()
        _R_ = _Rot.from_quat([_rq_[1], _rq_[2], _rq_[3], _rq_[0]])
        _rp_ = robot.data.root_pos_w[0].cpu().numpy()
        _eeb = robot.body_names.index("left_wrist_yaw_link")
        for _rt in getattr(teleop, "_retargeters", None) or []:
            _is_left = getattr(_rt, "_target", None) == DeviceBase.TrackingTarget.CONTROLLER_LEFT
            if not hasattr(_rt, "_root_pos") or not _is_left:
                continue
            _ee_w = robot.data.body_state_w[0, _eeb, 0:3].cpu().numpy().astype(np.float32)
            _ee = (_R_.as_matrix().T @ (_ee_w - _rp_)).astype(np.float32)
            _rt._init_pos = _ee.copy()
            _rt._pos = _ee.copy()
            _rt._lo = _ee - np.array([0.45, 0.45, 0.45], dtype=np.float32)
            _rt._hi = _ee + np.array([0.45, 0.45, 0.45], dtype=np.float32)
            _rt._prev = None
            _rt._smooth = None
            _ee_q = robot.data.body_state_w[0, _eeb, 3:7].cpu().numpy()
            _live_R = _R_.inv() * _Rot.from_quat([_ee_q[1], _ee_q[2], _ee_q[3], _ee_q[0]])
            _rt._init_R = _live_R
            _rt._quat_R = _live_R

    # Grasp-validity test hooks (default off): at FIATLUX_ROTATE_AT roll the right wrist
    # FIATLUX_ROTATE_DEG (90) about world FIATLUX_ROTATE_AXIS (x), then at FIATLUX_UNGRASP_AT open the
    # hand. A real grasp drops the bulb; one "held" by interpenetration stays stuck (false positive).
    _ungrasp_at = int(os.environ["FIATLUX_UNGRASP_AT"]) if os.environ.get("FIATLUX_UNGRASP_AT") else None
    _ungrasped = [False]
    _rotate_at = int(os.environ["FIATLUX_ROTATE_AT"]) if os.environ.get("FIATLUX_ROTATE_AT") else None
    _rotate_deg = float(os.environ.get("FIATLUX_ROTATE_DEG", "90"))
    _rotate_axis = os.environ.get("FIATLUX_ROTATE_AXIS", "x")
    _rotated = [False]
    step_i = 0
    while simulation_app.is_running():
        try:
            with torch.inference_mode():
                if rec_flag["toggle"]:
                    rec_flag["toggle"] = False
                    if recorder is not None:
                        recording_on = not recording_on
                        _ep_dir = os.path.join(args.out, f"ep{ep_idx:02d}")
                        if recording_on:
                            if video is None and images is None:
                                try:
                                    video, ego_video, images = _open_take_capture(_ep_dir)
                                except Exception as _e:  # noqa: BLE001
                                    print(f"[sonic] !! take writers FAILED to open: {_e!r}", flush=True)
                            print(
                                f"[sonic] RECORDING ON -> {_ep_dir} "
                                f"(video={'yes' if video is not None else 'NO'}, "
                                f"ego={'yes' if ego_video is not None else 'NO'})",
                                flush=True,
                            )
                        else:
                            # OFF closes the take: its own folder gets the bag, meta and a
                            # finalized video; buffers reset so the next take starts fresh.
                            _took = recorder.mark_episode_end()
                            _info = None
                            if _took:
                                _info = recorder.write(_ep_dir, fmt=args.record_format)
                                recorder.reset_buffers()
                                recorder.reset_gate()
                                _sc = _info.get("score") or {}
                                if _sc.get("mean_score") is not None:
                                    ep_scores.append(_sc["mean_score"])
                            # close the writers BEFORE sealing -- the rename moves the folder
                            if video is not None:
                                print(f"[sonic] take had {len(video)} follow-cam frames", flush=True)
                                if len(video):
                                    video.write()
                                video = None
                            else:
                                print("[sonic] !! no follow-cam writer for this take", flush=True)
                            if ego_video is not None:
                                if len(ego_video):
                                    ego_video.write()
                                ego_video = None
                            if images is not None:
                                images = None
                            if _took:
                                _sealed = _seal_take(_ep_dir, _info)
                                print(f"[sonic] RECORDING OFF -- take saved: {_sealed}", flush=True)
                                print(f"[sonic] {_score_line(_info)}", flush=True)
                                ep_idx += 1
                            else:
                                print("[sonic] RECORDING OFF (nothing buffered)", flush=True)

                if reset_flag["do"]:
                    reset_flag["do"] = False
                    if recorder is not None and recording_on and recorder.mark_episode_end():
                        # Flush NOW, not just at exit: Kit's SIGINT handler fast-exits past any
                        # finally/atexit, so the bag on disk must always hold every closed episode.
                        _ep_dir = os.path.join(args.out, f"ep{ep_idx:02d}")
                        _info = recorder.write(_ep_dir, fmt=args.record_format)
                        recorder.reset_buffers()
                        recorder.reset_gate()
                        _sc = _info.get("score") or {}
                        if _sc.get("mean_score") is not None:
                            ep_scores.append(_sc["mean_score"])
                        if video is not None and len(video):
                            video.write()  # close before the seal renames
                        if ego_video is not None and len(ego_video):
                            ego_video.write()
                        _sealed = _seal_take(_ep_dir, _info)
                        print(f"[sonic] take saved on reset: {_sealed}", flush=True)
                        print(f"[sonic] {_score_line(_info)}", flush=True)
                        ep_idx += 1
                        # recording stays ON across the reset -> next take gets fresh writers
                        video, ego_video, images = _open_take_capture(os.path.join(args.out, f"ep{ep_idx:02d}"))
                    env.reset()  # reset the task (bulb/socket, episode buffers)
                    # A FREE base isn't re-homed by env.reset() -> teleport the root back to the good
                    # centered spawn (zero velocity), THEN re-plant with the same pin + SONIC settle-in
                    # as startup. Handing a cold/zeroed-history robot straight to SONIC made it lurch
                    # and fly; resettle() lands it level, still, and balanced before control resumes.
                    robot.write_root_pose_to_sim(staged_root)
                    robot.write_root_velocity_to_sim(torch.zeros((env.num_envs, 6), device=dev))
                    resettle()
                    if args.input == "vr":
                        for _rt in getattr(teleop, "_retargeters", None) or []:
                            if hasattr(_rt, "reset"):
                                _rt.reset()  # re-reference arm targets to the re-homed robot
                            if hasattr(_rt, "relatch"):
                                _rt.relatch()  # in-hand legs: grip closed again until the trigger is pulled
                    elif kb is not None:  # re-home both keyboard EE targets + grips
                        kb["R_ee"] = rest_arm[0:7].clone()
                        kb["L_ee"] = rest_arm[8:15].clone()
                        kb["R_grip_open"] = not _right_starts_closed
                        kb["L_grip_open"] = True
                        kb["lean"] = 0.0
                    loco_cmd[:] = 0.0
                    rpy_cmd[:] = 0.0
                    print("[sonic] reset done", flush=True)

                if args.input == "vr":
                    # Sync each arm retargeter's base ROTATION to the LIVE base every frame, but KEEP its
                    # baked base POSITION. The retargeter maps its world-frame EE target into the root frame
                    # the IK expects; the transform was baked ONCE assuming a bolted/static base, but SONIC
                    # now sways + walks the base.
                    #  - Live rotation: a stale root rotation aims the command the wrong way as the base
                    #    yaws/sways, so the arm lunges. Keeping it live kills that swing.
                    #  - Baked position (NOT updated to live): the world target then rides the base's
                    #    TRANSLATION, so the hand FOLLOWS the body when you walk instead of hanging in world.
                    # Never touch _pos/_init_pos: those hold the target.
                    _rq = robot.data.root_quat_w[0].cpu().numpy()  # w, x, y, z
                    _rR = _Rot.from_quat([_rq[1], _rq[2], _rq[3], _rq[0]])
                    for _rt in getattr(teleop, "_retargeters", None) or []:
                        if hasattr(_rt, "_root_pos"):
                            _rt._root_R = _rR
                            _rt._root_R_T = _rR.as_matrix().T.astype(np.float32)

                    # Take arm + walk from the controller ONLY when it's streaming; otherwise hold.
                    out = teleop.advance()
                    if out is not None:
                        last_arm = out[:-n_walk]
                        _rail_pose = _rail_main_L() if _rail is not None else None
                        if _rail_pose is not None:
                            # the brace keeps the left arm; the left controller is ignored meanwhile
                            last_arm = last_arm.clone()
                            last_arm[8:15] = _rail_pose
                            last_arm[15] = 1.0
                        walk = out[-n_walk:].detach().cpu().numpy()
                        loco_cmd[:] = walk[:3]
                        if walk[3] > 0.5:
                            loco_cmd[:] = 0.0
                        rpy_cmd[1] = walk[4] * LEAN_MAG
                        _rec_now = walk[5] > 0.5
                        if _rec_now and not vr_rec_prev:
                            rec_flag["toggle"] = True
                        vr_rec_prev = _rec_now
                        if _rail is not None:
                            _sq_now, _both_now = walk[6] > 0.5, walk[7] > 0.5
                            if _sq_now and not vr_rail_prev[0] and _rail["held"]:
                                _rail_release()
                                _vr_take_left_arm()
                                print("[sonic] rail hand OFF (left grip squeezed): the left arm is yours", flush=True)
                            elif _both_now and not vr_rail_prev[1] and not _rail["held"]:
                                _rail_rebrace()
                                print("[sonic] rail hand ON (both left buttons): left arm back on the rail", flush=True)
                            vr_rail_prev = [_sq_now, _both_now]
                    else:
                        last_arm = rest_arm  # no controller -> FIXED rest pose (no IK re-solve jitter)
                        _rail_pose = _rail_main_L() if _rail is not None else None
                        if _rail_pose is not None:
                            last_arm = last_arm.clone()
                            last_arm[8:15] = _rail_pose
                            last_arm[15] = 1.0
                        loco_cmd[:] = 0.0
                        rpy_cmd[:] = 0.0
                else:  # keyboard: keys persist loco_cmd + move the arm EE target
                    while _scripted_keys and step_i / 50.0 >= _scripted_keys[0][0]:
                        _t, _name = _scripted_keys.pop(0)
                        _pressed.append(_name)
                        print(f"[sonic] scripted key {_name} at t={_t:.1f}s", flush=True)
                    kb_drain()
                    if kb["quit"]:
                        break
                    if _rotate_at is not None and step_i >= _rotate_at and not _rotated[0]:
                        from scipy.spatial.transform import Rotation as _RotK  # keyboard-path (VR imports its own)

                        _q = kb["R_ee"][3:7].detach().cpu().numpy()  # w, x, y, z
                        _Rc = _RotK.from_quat([_q[1], _q[2], _q[3], _q[0]])
                        _Rn = _RotK.from_euler(_rotate_axis, _rotate_deg, degrees=True) * _Rc  # world-frame roll
                        _nq = _Rn.as_quat()  # x, y, z, w
                        kb["R_ee"][3:7] = torch.tensor(
                            [_nq[3], _nq[0], _nq[1], _nq[2]], device=dev, dtype=kb["R_ee"].dtype
                        )
                        _rotated[0] = True
                        print(
                            f"[sonic] ROTATE hand {_rotate_deg:.0f} deg about world-{_rotate_axis} at step {step_i}",
                            flush=True,
                        )
                    if _ungrasp_at is not None and step_i >= _ungrasp_at and not _ungrasped[0]:
                        kb["R_grip_open"] = True  # open the hand: a real grasp drops the bulb now
                        _ungrasped[0] = True
                        print(
                            f"[sonic] AUTO-UNGRASP step {step_i}: hand OPEN (bulb should fall if gripped)",
                            flush=True,
                        )
                    _rail_pose = _rail_main_L() if _rail is not None else None
                    if _rail_pose is not None:
                        kb["L_ee"] = _rail_pose.clone()
                        kb["L_grip_open"] = True
                    last_arm = kb_arm_action()

                # Position hold: SONIC is a velocity policy with no position feedback, so cmd=0 slowly
                # glides the base away. Steer a gentle velocity back to home -- but only AFTER a warmup so
                # it doesn't fight the fragile settle; while walking, home follows the robot.
                base_xy = robot.data.root_pos_w[0, 0:2].cpu().numpy()
                _base_speed = float(torch.linalg.norm(robot.data.root_lin_vel_w[0, 0:2]))
                if step_i < WARMUP or np.linalg.norm(loco_cmd) > WALK_TH or _base_speed > HOLD_ANCHOR_V:
                    home_xy = base_xy.copy()
                    _hold_on = False
                else:
                    e = home_xy - base_xy
                    _en = float(np.linalg.norm(e))
                    if not _hold_on and _en > HOLD_ENGAGE:
                        _hold_on = True
                    elif _hold_on and _en < HOLD_RELEASE:
                        _hold_on = False
                    if _hold_on:
                        q = robot.data.root_quat_w[0].cpu().numpy()
                        yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
                        _cx = HOLD_KP * (np.cos(yaw) * e[0] + np.sin(yaw) * e[1])
                        _cy = HOLD_KP * (-np.sin(yaw) * e[0] + np.cos(yaw) * e[1])
                        _cm = float(np.hypot(_cx, _cy))
                        _scale = np.clip(_cm, HOLD_VMIN, HOLD_VMAX) / max(_cm, 1e-6)
                        loco_cmd[0] = float(_cx * _scale)
                        loco_cmd[1] = float(_cy * _scale)

                # SONIC runs EVERY frame -- balance is not optional. (Gating this on `out` made the
                # free-base robot collapse whenever the headset wasn't streaming.)
                obs_hist.append(build_obs())
                flat = np.concatenate(obs_hist).astype(np.float32)[None]
                sess = walk_sess if np.linalg.norm(loco_cmd) > 0.05 else bal_sess
                last_action = sess.run(None, {in_name: flat})[0][0]
                leg_target = torch.as_tensor(last_action * ACTION_SCALE + DEFAULT_15, device=dev)
                if args.lock_base:
                    # hold the staged stance; SONIC's balance output means nothing on a fixed root
                    leg_target = torch.as_tensor(DEFAULT_15, device=dev)
                robot.set_joint_position_target(leg_target.unsqueeze(0), joint_ids=act_idx)
                if _staged_close is not None:
                    for _sc in _staged_close:
                        _now_closed = float(last_arm.reshape(-1)[_sc["col"]]) < 0
                        if _now_closed and not _sc["closed"]:
                            _sc["term"]._close_command.copy_(_sc["lead"])
                            _sc["since"] = 0
                        elif _now_closed and _sc["since"] < _STAGE_STEPS:
                            _sc["since"] += 1
                            if _sc["since"] == _STAGE_STEPS:
                                _sc["term"]._close_command.copy_(_sc["full"])
                        _sc["closed"] = _now_closed
                arm_action = last_arm.repeat(env.num_envs, 1)
                step_out = env.step(arm_action)  # arms via the real env action manager
                if recorder is not None and recording_on:
                    _obs_t, _rew_t, _term_t, _trunc_t = step_out[0], step_out[1], step_out[2], step_out[3]
                    # extras: the operator/policy signals that BYPASS the env action -- the walk
                    # command the human gives, the lean, and SONIC's raw leg action for this step.
                    # (joint_pos_target inside the recorder covers the full commanded joint vector.)
                    _extras = {
                        "loco_cmd": np.asarray(loco_cmd, dtype=np.float32)[None],
                        "rpy_cmd": np.asarray(rpy_cmd, dtype=np.float32)[None],
                        "sonic_action": np.asarray(last_action, dtype=np.float32)[None],
                    }
                    if _rail is not None:
                        _extras["rail_contact_force"] = _rail_force().cpu().numpy().astype(np.float32)[None]
                    _extras.update(_probe_extras())
                    recorder.record_step(_obs_t, arm_action, _rew_t, _term_t, _trunc_t, extras=_extras)
                    if args.stop_on_success and getattr(recorder, "_gate_fired", False):
                        # The gate latched: end the take here. The scene does NOT reset (the
                        # success termination is cleared), so without this the operator keeps
                        # recording a second episode into the same take and the score is averaged
                        # down -- exactly the 0.50 that a clean success produced before.
                        print("[sonic] *** SUCCESS -- gate satisfied; closing the take ***", flush=True)
                        rec_flag["toggle"] = True
                    if video is not None:
                        video.capture(pose=_video_pose())
                    if ego_video is not None:
                        ego_video.capture()  # head-mounted: pose comes from the robot
                    if images is not None:
                        images.maybe_capture(len(recorder._buf["done"]) - 1)  # flat recorded-step index
                step_i += 1
                if args.max_steps and step_i >= args.max_steps:
                    print(f"[sonic] --max_steps {args.max_steps} reached; ending session.", flush=True)
                    break
                if step_i % (20 if step_i <= 200 else 100) == 0:
                    p = robot.data.root_pos_w[0].cpu().numpy()
                    _re = float(robot.data.joint_pos[0, robot.joint_names.index("right_elbow_joint")])
                    _tag = "FELL" if p[2] < 0.4 else "ok"
                    print(
                        f"[sonic] step {step_i} pelvis=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f}) "
                        f"r_elbow={_re:.2f} loco_cmd={np.round(loco_cmd, 2)} {_tag}",
                        flush=True,
                    )
        except KeyboardInterrupt:
            break
        except Exception as e:  # noqa: BLE001
            print(f"[sonic] step skipped while view recovers: {e}", flush=True)
            with contextlib.suppress(Exception):
                env.sim.render()

    if recorder is not None:
        recorder.mark_episode_end()  # close a trailing (still-recording) take
        _trailing = None
        if recorder._buf:
            _ep_dir = os.path.join(args.out, f"ep{ep_idx:02d}")
            _trailing = recorder.write(_ep_dir, fmt=args.record_format)
            _sc = _trailing.get("score") or {}
            if _sc.get("mean_score") is not None:
                ep_scores.append(_sc["mean_score"])
        if video is not None and len(video):
            video.write()  # close the stream BEFORE the seal renames
        if ego_video is not None and len(ego_video):
            ego_video.write()
        if images is not None and len(images):
            print(f"[sonic] trailing take images: {len(images)} sets", flush=True)
        if _trailing is not None:
            _sealed = _seal_take(os.path.join(args.out, f"ep{ep_idx:02d}"), _trailing)
            print(f"[sonic] trailing take saved: {_sealed}", flush=True)
            print(f"[sonic] {_score_line(_trailing)}", flush=True)
            ep_idx += 1
        if ep_idx:
            # The session dir keeps its plain timestamp: the score lives on each take
            # (epNN_score<X.XX>), so a listing reads as per-demo results rather than one
            # averaged number that a single mis-press could drag down.
            _mean = sum(ep_scores) / len(ep_scores) if ep_scores else 0.0
            print(f"[sonic] session: {ep_idx} take(s) under {args.out} (mean of take scores {_mean:.2f})", flush=True)
        else:
            print("[sonic] recording was on but nothing was captured -- no take written.", flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
