# fiatlux_teleop — VR + keyboard teleoperation for the Fiatlux benchmark

Teleop **on top of** the `fiatlux_task` benchmark. It's a separate package so the benchmark
installs/runs without teleop's deps (OpenXR / CloudXR / the SONIC onnxruntime stack). The dependency
arrow points **teleop → benchmark**: each teleop env here subclasses a benchmark env and swaps its RL
whole-body action for a human-drivable **arm-IK + binary-grip** interface, driven over a Pico headset
(CloudXR) **or the keyboard** (`sonic_teleop.py --input vr|keyboard`).

- **Operator** (running a session): **no code** — one launcher command.
- **Developer** (making a *new* task teleop-able): a ~30-line cfg + one registration, once.

---

## Setup (one-time)

One command installs everything (idempotent; `verify` mode checks without installing):

```bash
./scripts/teleop/setup_sim_teleop.sh            # keyboard tier: sim env + assets + SONIC onnx
./scripts/teleop/setup_sim_teleop.sh vr         # + the CloudXR/Pico tier
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
- **Headset** — Pico 4 Ultra (or any CloudXR-compatible OpenXR headset) on the **same Tailscale
  tailnet** as the GPU box (install the Tailscale APK on the Pico, log into the same tailnet).
- **`~/.cloudxr/`** — CloudXR install dir with `openxr_cloudxr.json` + a self-signed cert whose SAN
  carries your **tailnet IP** (else the Pico browser can't get past the cert warning).

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

### Activate XR (in the headset)
Once `[5/5] READY` prints, the launcher echoes these — in order:
1. **In the Isaac Sim window** → **AR** panel → Output Plugin **OpenXR**, Runtime **System OpenXR
   Runtime** → **Start AR**.
2. **On the Pico browser** → `https://<tailnet-ip>:48322/client/` → cert warning → **Advanced →
   Proceed**.
3. Client **Settings**: Device Profile **Pico 4 Ultra**, Server IP **`<tailnet-ip>`**, **Port
   `48322`** — **not** the default `49100` (48322 is the TLS/WSS proxy; 49100 is the raw backend).
4. **Connect** → the scene streams to the headset. Then drive with the controls below.

### Controls (Pico controllers)
| Input | Action |
|---|---|
| **grip-clutch + move** controller | move the arm (release grip to reposition without moving the arm) |
| **trigger** | grasp / release the hand |
| **left stick** | walk (SONIC) · **right stick X** | turn · **right button** | stop |
| **left X / Y buttons** | lean |

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
scripts/teleop/
  sonic_teleop.py                   # whole-body driver: SONIC legs + arm teleop, --input vr|keyboard
  restart_sonic_teleop.sh           # VR launcher (CloudXR runtime + sim + sonic_teleop.py --input vr)
```

- `import fiatlux_task` registers the **benchmark** tasks; `import fiatlux_teleop` registers the
  **teleop** tasks. The teleop scripts import both. Launchers put both on `PYTHONPATH`
  (`source/fiatlux_task:source/fiatlux_teleop`).
- Registration is **lazy** (string entry points): importing either package is cheap; a cfg module
  loads only when its task is actually made.

## Scope note
This is a *bespoke* harness (G1 arm-IK + SONIC legs + CloudXR), tuned for these scenes — not a generic
"point at any env and teleop." Adding a new teleop task is the ~30-line cfg above, following the
templates. For the CloudXR / Pico setup itself, see `journal/specs/vr-teleop-cloudxr-setup.md`.
