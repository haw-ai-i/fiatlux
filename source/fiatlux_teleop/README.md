# fiatlux_teleop — VR teleoperation for the Fiatlux benchmark

VR teleop **on top of** the `fiatlux_task` benchmark. It's a separate package so the benchmark
installs/runs without teleop's deps (OpenXR / CloudXR / the SONIC onnxruntime stack). The dependency
arrow points **teleop → benchmark**: each teleop env here subclasses a benchmark env and swaps its RL
whole-body action for a human-drivable **arm-IK + binary-grip** interface over a Pico headset (CloudXR).

- **Operator** (running a session): **no code** — one launcher command.
- **Developer** (making a *new* task teleop-able): a ~30-line cfg + one registration, once.

---

## Quickstart — teleop an existing task (operator)

Three teleop tasks ship ready to run: `FIATLUX-Insert-Teleop-v0`, `FIATLUX-Carry-Teleop-v0`,
`FIATLUX-LadderGallery-Teleop-v0`.

```bash
# from the repo root. Walking + arm teleop (SONIC legs):
FIATLUX_TASK=FIATLUX-Carry-Teleop-v0 FIATLUX_HAND=dex3 bash scripts/teleop/restart_sonic_teleop.sh
```

The launcher starts the whole stack (CloudXR runtime + Isaac Sim + the teleop driver) and prints
`[5/5] READY`. Then on the **Pico**: AR panel → **Start AR**, open the browser client, connect.

- `FIATLUX_TASK` — which teleop task id to run.
- `FIATLUX_HAND` — `dex3` or `inspire`.
- `NV_CXR_ENDPOINT_IP` — your tailnet IP (has a default).

Stationary manipulation (no legs) uses a different launcher:
```bash
FIATLUX_TASK=FIATLUX-Insert-Teleop-v0 bash scripts/teleop/restart_xr_teleop.sh
```

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
might touch the *driver*; the stationary path (`insert_teleop.py` / `xr_teleop.py`) has no such checks.

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
  sonic_teleop.py                   # driver: env's arm teleop + SONIC legs, over CloudXR
  xr_teleop.py / insert_teleop.py   # stationary drivers (no SONIC)
  restart_sonic_teleop.sh           # one-command launcher (CloudXR runtime + sim + driver)
  restart_xr_teleop.sh              # stationary launcher
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
