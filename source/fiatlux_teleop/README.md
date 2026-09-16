# fiatlux_teleop — VR + keyboard teleoperation for the Fiatlux benchmark

Teleop **on top of** the `fiatlux_task` benchmark. It's a separate package so the benchmark
installs/runs without teleop's deps (OpenXR / CloudXR / the SONIC onnxruntime stack). The dependency
arrow points **teleop → benchmark**: each teleop env here subclasses a benchmark env and swaps its RL
whole-body action for a human-drivable **arm-IK + binary-grip** interface, driven over a VR headset
(CloudXR) **or the keyboard** (`sonic_teleop.py --input vr|keyboard`).

- **Operator** (running a session): **no code** — one launcher command.
- **Developer** (making a *new* task teleop-able): a ~30-line cfg + one registration, once.

---

## Setup (one-time)

One command installs everything (idempotent; `verify` mode checks without installing):

```bash
./scripts/teleop/setup_sim_teleop.sh            # keyboard tier: sim env + assets + SONIC onnx
./scripts/teleop/setup_sim_teleop.sh vr         # + the CloudXR headset tier
./scripts/teleop/setup_sim_teleop.sh verify     # check every piece
```

By default it builds a **fresh uv env** (`uv sync --extra teleop` — Isaac Sim/Lab as pinned wheels,
no external IsaacLab checkout to depend on). If you already have a correct sim env and know its
path, opt in explicitly: `SIM_PYTHON=/path/to/env/bin/python ./scripts/teleop/setup_sim_teleop.sh`
(the script validates it — imports, versions, and where its `isaaclab` really lives).

Teleop runs as **two processes in two envs**, kept separate so the CloudXR deps never touch the sim:

| Env | What's in it | Role |
|---|---|---|
| **sim env** (uv `.venv`, or your `SIM_PYTHON`) | Isaac Sim 5.1 / Isaac Lab 2.3.2; `fiatlux_task` + `fiatlux_teleop` on `PYTHONPATH`; `onnxruntime` via the `teleop` extra (SONIC legs) | renders + runs the sim, reads XR input |
| **`vr_teleop`** | `pip install 'isaacteleop[cloudxr,retargeters]~=1.3.0'` (1.3.131 verified) | the CloudXR streaming runtime only |

> **uv gotcha:** `uv sync` without the extra makes the env match the lockfile *exactly* — it
> uninstalls `onnxruntime` again. Launch teleop with **`uv run --extra teleop python …`** (as the
> setup script prints) and it self-heals regardless of what was synced before.

Also needed:
- **Headset** — Quest 3S/3/2 or Pico 4 Ultra (any headset the CloudXR web client profiles) on
  the **same Tailscale tailnet** as the GPU box, or simply the same LAN.
- **`~/.cloudxr/`** — CloudXR install dir with `openxr_cloudxr.json` + a self-signed cert whose SAN
  carries your **tailnet IP** (else the headset's browser can't get past the cert warning).

The launcher activates both envs for you — you never switch them by hand. Full first-time install,
firewall ports, network topology, and every hard-won gotcha:
**[journal/specs/vr-teleop-cloudxr-setup.md](../../journal/specs/vr-teleop-cloudxr-setup.md)**.

---

## Quickstart — teleop an existing task (operator)

Three teleop tasks ship ready to run: `FIATLUX-Insert-Teleop-v0`, `FIATLUX-Carry-Teleop-v0`,
`FIATLUX-LadderGallery-Teleop-v0`.

```bash
# from the repo root. Walking + arm teleop (SONIC legs):
FIATLUX_TASK=FIATLUX-Carry-Teleop-v0 FIATLUX_HAND=dex3 bash scripts/teleop/restart_sonic_teleop.sh
```

That one command starts **both** processes (CloudXR runtime in `vr_teleop` + the sim/driver in
`env_isaaclab`) and prints `[5/5] READY`.

- `FIATLUX_TASK` — which teleop task id to run.
- `FIATLUX_HAND` — `dex3` or `inspire`.
- `FIATLUX_TELEOP_ROBOT_USD` — optional path to an alternative robot USD, teleop only (the benchmark
  envs never see it). Used for the Inspire thumb fix: `python scripts/omniverse/inspire_thumb_frame.py
  assets/unitree_g1/wholebody_inspire/g1_29dof_with_inspire_rev_1_0.usd` writes a `_thumbfix.usd`
  wrapper next to the vendor file whose thumb rotation (both hands) lifts the thumb off the palm (the stock joint
  frame tops out 55° above the palm plane; the RH56DFTP manual gives 90°). With it set, the in-hand
  legs (S04/S05/S06/S09/S10/S11) on Inspire are staged open with the thumb raised and the close
  preset only supports the glass with the fingers during the settle (`FIATLUX_TELEOP_INHAND_CLOSE`,
  default `0.2:1.0`; `FIATLUX_TELEOP_INHAND_THUMB_YAW`, default `1.0`, or `cup` to keep the task's
  own staging). At "Teleop ready" the driver restores the normal full
  grasp preset for the operator — the support curl exists only inside the settle; the operator's
  hand closes like the left one and stops where the bulb stops it. The wrapper *references* the vendor USD and overrides only that one joint's
  two frame quaternions (rest pose and limits unchanged) — the vendor file itself is never edited,
  both files stay gitignored, and re-running the script after an asset re-sync reproduces the
  wrapper. Verify against a physical RH56DFTP (sweep thumb-rotation `ANGLE_SET(5)` and watch the
  tip rise over its base) before adopting it as the default asset.
- Inspire close behaviour (active whenever `--hand inspire`; the teleop twins run with the robot's
  self-collisions ON, as the benchmark authors them in `robots/g1.py`):
  - **Staged close** — driving the fingers and the thumb to the fist preset simultaneously wedges
    the fingertips on the thumb tip mid-flight (the close jams into a hollow "beak" and a tabletop
    bulb is squeezed out instead of enveloped). The driver closes the way a hand makes a fist: the
    fingers lead, the thumb bend follows `FIATLUX_TELEOP_STAGE_STEPS` sim steps later (20 ms each,
    default `2` = 40 ms). `FIATLUX_TELEOP_STAGE=thumb` flips the order. A close ON the bulb is
    unaffected — the leading group stops on the glass and the trailing group clamps.
  - **Hand torque cap** — `FIATLUX_TELEOP_HAND_EFFORT` (N·m per finger joint, default `0.6`, `off`
    to disable) caps the hand actuators at the physical RH56DFTP's fingertip force (~10 N). Without
    it, a blocked close grinds saturated PD torques through the bulb→thumb→palm loop and the
    vibration can tip SONIC (2 of 3 hands-off in-hand settles fell before the cap; 0 after).
- `NV_CXR_ENDPOINT_IP` — **required** for the VR launchers (your GPU box's tailnet IP); they exit with
  a clear error if it's unset. e.g. `NV_CXR_ENDPOINT_IP=100.x.y.z FIATLUX_TASK=… bash …restart_sonic_teleop.sh`.

### Keyboard (whole-body, no headset)
Same SONIC walking + arm teleop, from the desktop — run the driver directly (no CloudXR, no headset):
```bash
conda activate env_isaaclab
export PYTHONPATH=$PWD/source/fiatlux_task:$PWD/source/fiatlux_teleop
python scripts/teleop/sonic_teleop.py --task FIATLUX-Carry-Teleop-v0 --input keyboard
```
Click the Isaac Sim viewport to focus it. **Bimanual** — **Tab** switches the active arm (R ↔ L):
- active arm: **W/S A/D Q/E** move X/Y/Z, **U/O I/K J/L** roll/pitch/yaw, **G** toggles grip
- walk: **arrows** (↑↓ forward/back, ←→ turn), **, / .** strafe, **T/Y** lean, **Space** stop
- **R** reset, **Esc** quit

On the six in-hand legs the right grip **starts closed** on the seated bulb, so the first **G**
releases it. For hands-off tests, `--keys "G@4,R@8,ESCAPE@15"` injects timed key presses
(seconds of teleop time) into the same queue as real ones.

### Activate XR (in the headset)
Once `[5/5] READY` prints, the launcher echoes these — in order:
1. **In the Isaac Sim window** → **AR** panel → Output Plugin **OpenXR**, Runtime **System OpenXR
   Runtime** → **Start AR**.
2. **In the headset's browser** → `https://<tailnet-ip>:48322/client/` → cert warning → **Advanced →
   Proceed**.
3. Client **Settings**: Device Profile **your headset** (Quest 3S/3/2 or Pico 4 Ultra), Server IP **`<tailnet-ip>`**, **Port
   `48322`** — **not** the default `49100` (48322 is the TLS/WSS proxy; 49100 is the raw backend).
4. **Connect** → the scene streams to the headset. Then drive with the controls below.

### Controls (motion controllers)
| Input | Action |
|---|---|
| **grip-clutch + move** controller | move the arm (release grip to reposition without moving the arm) |
| **trigger** | grasp / release the hand |
| **left stick** | walk (SONIC) · **right stick X** | turn |
| **right A** (lower button) | stop walking |
| **right B** (upper button) | recording on/off (when `--record` is active) |
| **left X / Y buttons** | lean forward / back |

---

## Record demo sessions

`sonic_teleop.py` can record every session as a **demo bag** in the same robomimic-style format as
the benchmark's `record_run.py` policy bags -- so human demos and policy rollouts share one format,
one scorer, one toolchain. (The repo itself has no demo-consuming trainer yet; the bags simply
record everything below, in a standard layout.)

```bash
# keyboard
... sonic_teleop.py --task FIATLUX-Carry-Teleop-v0 --input keyboard --record bag
# VR (the launcher forwards these):
FIATLUX_RECORD=1 [FIATLUX_RECORD_VIDEO=1] [FIATLUX_RECORD_START=toggle] [FIATLUX_RECORD_FORMAT=npz] \
    NV_CXR_ENDPOINT_IP=<ip> FIATLUX_TASK=... bash scripts/teleop/restart_sonic_teleop.sh
```

**What a session records** (per step, all envs' joints -- dex3 = 43 columns, inspire = 53):
`policy_obs`, the 16-dim teleop `actions` (EE pose + grip per arm), **`joint_pos_target`** (the
complete commanded joint vector -- including SONIC's legs, which bypass the action manager),
`loco_cmd`/`rpy_cmd` (the operator's walk/lean), `sonic_action`, measured joint states, object
poses, contact forces, rewards, and termination flags. `meta.json` carries the joint-name column
map (resolved from the LIVE robot), per-term action breakdown, and the **benchmark score** --
which is also appended to the session folder name on clean exit, so a directory listing reads as
a ranking.

**Episodes** split at the operator's boundaries: `[R]` reset, recording toggled off (keyboard
`C` / VR right **B**), or exit. Each boundary **flushes the bag to disk** -- a crashed or killed
session keeps every closed episode (Kit's SIGINT handler skips the final write, so only the
trailing unclosed episode can be lost; use `--max_steps N` for clean scripted endings).

**Options** (each is a flag; see `--help`):
- `--record-start auto|toggle` -- record from launch, or start OFF until the operator toggles.
- `--record-format hdf5|npz` -- robomimic-style HDF5 (default) or flat npz.
- `--record-video` -- `video.mp4` (third-person) + `ego.mp4` (head camera) + poster PNGs, streamed
  to disk (review footage). `--camera auto|follow|static|fixture|bench|crate` picks the third-person
  shot; `auto` (default) chooses per task so the robot and the task's own objects stay in frame.
- `--record-settle` -- include the ~90-step startup settle in the take (spawn-time failures happen
  there; a take that starts at the main loop only shows the aftermath).
- `--no-arm-pin` -- don't pin the idle arm at its settle joints; the pin makes a hands-off robot
  fall at ~3 s, so use this when the robot must still be standing when you connect.
- `--record-images --images-stride N` -- the env's own `wrist_camera`/`ego_camera` as JPEGs in
  `<session>/images/<camera>/f<step>.jpg` (default every 5th step = 10 Hz). The filename index is
  the bag's flat row number, so each frame pairs 1:1 with that row's `policy_obs`/`actions`; the
  MP4 is review footage only.

**Storage** is dataset-first, outside the git tree (`$FIATLUX_CAPTURES_DIR`, default
`../teleop-captures` beside the repo). One dimension per level, so a training dataset is always
one glob and always schema-homogeneous:

```
teleop-captures/<task>/<hand>/<kind>/<input>/<YYYY-MM-DD>/<HHMMSS>_score<X>/
                        |      |       |                    run.h5|run.npz, meta.json,
                        |      |       keyboard|vr          [video.mp4, images/...]
                        |      hdf5|npz[+images]
                        dex3|inspire  (derived from the LIVE robot -- 43 vs 53 joint columns)

# sessions with images:   <task>/dex3/hdf5+images/**
# sessions without:       <task>/dex3/hdf5/**
# only good demos:        .../**/*_score0.[5-9]*/
```

---

## Add teleop to a NEW task (developer)

You've made a benchmark task, say `FIATLUX-MyTask-v0` (a `mytask_env_cfg.py` in `fiatlux_task` + its
gym registration). To make it teleop-able:

### 1. Write a teleop cfg (copy the closest template)
Create `source/fiatlux_teleop/fiatlux_teleop/mytask_teleop_env_cfg.py`. Start from the closest
existing one and tweak:
- robot **walks around** the scene → copy `carry_teleop_env_cfg.py` (free base, SONIC drives legs),
- **stationary** manipulation (arms only) → copy `insert_teleop_env_cfg.py` (base bolted).

A teleop cfg subclasses your benchmark env and swaps **three** things:

```python
from isaaclab.utils import configclass
from fiatlux_task.tasks.manager_based.fiatlux_task.mytask_env_cfg import MyTaskEnvCfg
from .xr_controller_retargeters import Se3RelControllerRetargeterCfg, ControllerGripperRetargeterCfg
# + DifferentialInverseKinematicsActionCfg / BinaryJointPositionActionCfg, DevicesCfg, OpenXRDeviceCfg, XrCfg

@configclass
class MyTaskTeleopEnvCfg(MyTaskEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actions = MyTaskTeleopActionsCfg()   # 1. RL whole-body action -> arm-IK + binary grip
        self.teleop_devices = DevicesCfg({         # 2. the XR device + retargeters (arm pose + grip)
            "controller_rel": OpenXRDeviceCfg(retargeters=[
                Se3RelControllerRetargeterCfg(bound_hand=..., root_pos=..., sim_device=self.sim.device),
                ControllerGripperRetargeterCfg(bound_hand=..., sim_device=self.sim.device),
                # (left-hand pair too for bimanual)
            ], sim_device=self.sim.device, xr_cfg=self.xr),
        })
        self.xr = XrCfg(anchor_prim_path="/World/envs/env_0/Robot/pelvis", ...)  # 3. follow-camera
        for t in (...):                            # disable auto-terminations (don't reset mid-session)
            setattr(self.terminations, t, None)
```
Everything above (the IK action, the retargeters, the grip, the follow-cam) is reused — you're just
pointing them at your scene. See `carry_teleop_env_cfg.py` for a full worked example (bimanual Dex3,
ladder placement, follow-cam).

### 2. Register it
In `fiatlux_teleop/__init__.py`:
```python
gym.register(
    id="FIATLUX-MyTask-Teleop-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.mytask_teleop_env_cfg:MyTaskTeleopEnvCfg"},
)
```

### 3. (walking tasks only) let the driver recognize it
`scripts/teleop/sonic_teleop.py` branches on the **task name** for two things — the **hand swap**
(`"Insert"`/`"Carry"` in the task id) and the **arm spawn pose** (`"Carry"`/`"Gallery"`). So a new
walking task either **names to match** an existing pattern (e.g. a ladder task with `Carry` in the id
inherits the Carry arm pose) **or** you add one small branch there. This is the only place a new task
might touch the *driver* — there's only one, the whole-body `sonic_teleop.py`.

### 4. Run it
```bash
FIATLUX_TASK=FIATLUX-MyTask-Teleop-v0 FIATLUX_HAND=dex3 bash scripts/teleop/restart_sonic_teleop.sh
```

**Effort per new task:** the cfg (copy + tweak, ~30–50 lines) + one `gym.register` + maybe a 1-line
driver branch. You never wire teleop from scratch — you plug a new scene into the existing harness.

---

## Layout / how it's wired

```
source/fiatlux_teleop/fiatlux_teleop/
  __init__.py                       # registers the FIATLUX-*-Teleop gym ids (import this to register)
  carry_teleop_env_cfg.py           # walking template (free base + SONIC), bimanual Dex3, ladder scene
  insert_teleop_env_cfg.py          # stationary template (bolted base), bulb-insert scene
  ladder_gallery_teleop_env_cfg.py  # all ladder designs on an open floor (a Carry-Teleop subclass)
  xr_controller_retargeters.py      # controller pose -> arm target, controller trigger -> grip
  teleop_recording.py               # demo recording: TeleopTrajectoryRecorder (task-agnostic robomimic
                                    #   bags + operator episode boundaries + score-in-meta),
                                    #   StreamingVideoRecorder (constant-memory MP4), ImageCapture (ACT frames)
scripts/teleop/
  sonic_teleop.py                   # whole-body driver: SONIC legs + arm teleop, --input vr|keyboard,
                                    #   --record bag [--record-video --record-images ...] demo recording
  restart_sonic_teleop.sh           # VR launcher (CloudXR runtime + sim + sonic_teleop.py --input vr);
                                    #   forwards recording via FIATLUX_RECORD/_VIDEO/_START/_FORMAT
```

- `import fiatlux_task` registers the **benchmark** tasks; `import fiatlux_teleop` registers the
  **teleop** tasks. The teleop scripts import both. Launchers put both on `PYTHONPATH`
  (`source/fiatlux_task:source/fiatlux_teleop`).
- Registration is **lazy** (string entry points): importing either package is cheap; a cfg module
  loads only when its task is actually made.

## Scope note
This is a *bespoke* harness (G1 arm-IK + SONIC legs + CloudXR), tuned for these scenes — not a generic
"point at any env and teleop." Adding a new teleop task is the ~30-line cfg above, following the
templates. For the CloudXR headset setup itself, see `journal/specs/vr-teleop-cloudxr-setup.md`.
