# VR (Meta Quest) teleoperation — setup guide

Goal: drive the G1's right hand in `FIATLUX-Insert-Teleop-v0` with a **Meta Quest** via NVIDIA
**IsaacTeleop** (CloudXR streaming), instead of the keyboard. The env already takes an *absolute
end-effector pose + a binary grip* — exactly what a VR controller provides — so VR just swaps the
input source.

> **Note:** IsaacTeleop / CloudXR are an **extension dependency**, kept *out* of the benchmark's
> requirements. Install them in a **separate venv**, not in `env_isaaclab`.

---

## ⭐ Findings & working recipe — Pico 4 Ultra Enterprise, remote over Tailscale (2026-07)

Hard-won results from getting a **Pico 4 Ultra Enterprise** connected to a **remote** GPU PC
(`yujin-chen-MS-7C37`, RTX 3090, Tailscale `100.112.32.21`) over Tailscale. Read this before
re-deriving anything.

### What worked (the breakthroughs)

1. **Pico beats Quest for sideloading.** The Quest was blocked because Developer Mode requires the
   headset's Meta account to be a registered developer — impossible on a borrowed account. On the
   **Pico 4 Ultra Enterprise** you don't need dev mode at all: **Settings → Security → Install
   unknown apps** (enable for the PICO Browser), then download the **Tailscale APK directly** from
   `pkgs.tailscale.com/stable/tailscale-android-universal-<ver>.apk` (the Google-Play button needs a
   Google account; the direct APK does not). Install, log into the **same tailnet** as the PC.

2. **The remote-streaming breakthrough — advertise the Tailscale IP to CloudXR.** By default the
   CloudXR runtime only gathers its **LAN + docker** interfaces (`10.94.x`, `172.17.0.1`) as ICE
   candidates — never the Tailscale IP — so the video media path never forms over Tailscale
   (signaling connects on TCP 48322, then the stream dies at the ~15 s ICE timeout →
   `0xC0F22226 "no response from media server"`). Fix: **tell CloudXR its reachable endpoint** via
   env vars before launching the server:

   ```bash
   export NV_CXR_ENDPOINT_IP=100.112.32.21   # the PC's Tailscale IP
   export NV_CXR_MEDIA_PORT=47998            # media UDP port (both IP *and* port required —
                                             # with only the IP set, the log shows port=0 and it
                                             # still fails: "endpoint not configured")
   python -m isaacteleop.cloudxr --accept-eula --host-client
   ```
   Confirm in `~/.cloudxr/logs/cxr_streamsdk.*.log`: `Added public endpoint candidate:
   100.112.32.21:47998`. After this, the media connects **directly over Tailscale** — no TURN needed.

3. **coturn TURN relay (fallback, ended up unnecessary).** IsaacTeleop only wires TURN into
   `--usb-local` mode, not the WiFi path. For a manual relay: `sudo apt-get install -y coturn` (then
   `sudo systemctl stop/disable coturn` to free port 3478), and run
   `turnserver -n --listening-ip=0.0.0.0 --relay-ip=<tailscale-ip> --listening-port=3478
   --min-port=49160 --max-port=49200 --realm=cloudxr --lt-cred-mech --user=cxr:cxrsecret --no-tls
   --no-dtls --no-cli`. The web client accepts it via URL params
   `?turnServer=turn:<ip>:3478&turnUsername=cxr&turnCredential=cxrsecret&iceRelayOnly=1`. **Gotcha:**
   coturn returns `403 Forbidden IP` for a permission to its **own** relay IP, so with everything on
   one box the return path is confused — the `NV_CXR_ENDPOINT_IP` route (direct) is cleaner and made
   TURN moot.

4. **Cap the streaming resolution.** The Pico 4 Ultra requests **4096×4032 @ 90 Hz**, which NVENC
   chokes on (`CUDA invalid device context` during encoder init). Client URL params
   `?perEyeWidth=1280&perEyeHeight=1280&deviceFrameRate=72&maxStreamingBitrateMbps=40` are the knobs
   (note: on Pico these didn't visibly override the native framebuffer request in testing — worth
   revisiting).

### What is NOT solved yet

- **Controller buttons/trigger don't forward.** Decisive test (instrumenting `GripperRetargeter`):
  `ctrl_is_none=False, trigger=0.0` **even while squeezing** — the controller **pose** streams to the
  app, but **trigger/button values stay 0**. Suspected cause: the stock examples run **headless**
  (`Created OpenXR session (headless mode)`), so the OpenXR session may never reach the *focused*
  state where button actions are delivered (poses arrive via CloudXR device-push; buttons may not).
  Unconfirmed whether it's headless-focus or a Pico input-mapping gap — **retest with a rendering
  app.**

- **No scene in the headset with stock examples.** Every Python example under
  `examples/teleop/python/` (including the misleadingly-named `isaac_lab_gripper_example.py`) is a
  **headless input/retargeting demo** — it renders nothing, so the headset is correctly black. To see
  a scene you must run a **rendering XR app** that submits frames: either `camera_streamer` (C++), or
  the real goal — **Isaac Sim rendering the fiatlux scene with VR/OpenXR output pointed at CloudXR.**
  That integration (+ the Isaac Lab 3.0 vs our 2.3.2 gap, + the two-env dep conflict) is the next
  chunk of work.

### Verified-good server-side state (this box)

- conda env `vr_teleop` with `isaacteleop 1.3.131`; repo `~/robotica_project/IsaacTeleop @ v1.3.131`;
  CloudXR runtime 6.2.0 at `~/.cloudxr`.
- Start: `conda activate vr_teleop && source ~/.cloudxr/run/cloudxr.env &&
  export NV_CXR_ENDPOINT_IP=100.112.32.21 NV_CXR_MEDIA_PORT=47998 &&
  python -m isaacteleop.cloudxr --accept-eula --host-client` (ports 48322 + 49100 listen).
- Headless input smoke test (no scene, black headset — expected):
  `python examples/teleop/python/gripper_retargeting_example_simple.py`. It must be launched **after**
  the headset is connected (else `Failed to get OpenXR system: -35`).
- Pico connect URL: `https://100.112.32.21:48322/client/` → accept the self-signed cert on the
  `:48322/` page (separate tab), then **Connect** on `/client/`, then **Play** (= Start Teleop).

---

## ⭐⭐ Isaac Lab NATIVE XR teleop — the working integration path (2026-07)

**This supersedes the IsaacTeleop device path.** IsaacTeleop's own Isaac Lab device
(`IsaacTeleopDevice`) targets Isaac Lab 3.0 / Isaac Sim 6.0 — we run **Isaac Sim 5.1.0 / Isaac Lab
0.54.3**, so instead we use **Isaac Lab's own** built-in OpenXR teleop (`isaaclab.devices.OpenXRDevice`
+ retargeters). Isaac Sim renders the sim itself and streams it via CloudXR; the headset sends **hand
tracking** (pinch), which sidesteps the controller-trigger gap entirely.

### Architecture (two processes, no dep conflict)
- **Process A — CloudXR runtime** (conda env `vr_teleop`): `python -m isaacteleop.cloudxr
  --accept-eula --host-client` with `NV_CXR_ENDPOINT_IP=100.112.32.21` + `NV_CXR_MEDIA_PORT=47998`.
  Serves the web client (:48322) the Pico connects to. Only isaacteleop's CloudXR *runtime* is used
  here — not its device layer.
- **Process B — Isaac Lab** (conda env `env_isaaclab`): the fiatlux sim, launched with `--xr` and
  `XR_RUNTIME_JSON` → our CloudXR runtime (via `source ~/.cloudxr/run/cloudxr.env`). Isaac Sim's
  native OpenXR output renders the fiatlux scene into CloudXR; `OpenXRDevice` reads hand tracking and
  retargets it to the G1. **No `isaacteleop` import here** → the two-env dep conflict is avoided.

### Code changes (in the fiatlux repo)
- `source/fiatlux_task/.../insert_teleop_env_cfg.py`: added a **`handtracking`** entry to
  `teleop_devices` (an `OpenXRDeviceCfg` with `Se3AbsRetargeterCfg` [right-wrist pose → EE target,
  `use_wrist_position=True`, `zero_out_xy_rotation=True`] + `GripperRetargeterCfg` [thumb-index pinch
  → open/close]), plus `self.xr = XrCfg(anchor_pos=(0.5, 0.7, 0.0), ...)` for scene placement.
  Mirrors Isaac Lab's proven `stack_ik_abs_env_cfg.py` (Franka IK-abs) template. Verified: headless
  smoke test loads the device + both retargeters (env action space already matches — absolute IK +
  binary grip).
- `scripts/xr_teleop.py` (**new**): a fiatlux-aware adaptation of Isaac Lab's
  `scripts/environments/teleoperation/teleop_se3_agent.py`; the substantive addition is
  `import fiatlux_task.tasks` so `FIATLUX-*` envs register. Auto-enables `--xr` for handtracking,
  builds the device from the env cfg, runs `advance()`→`env.step()` with START/STOP/RESET callbacks.

### Run recipe
```bash
# Process A (env vr_teleop) — CloudXR runtime, if not already running:
conda activate vr_teleop && source ~/.cloudxr/run/cloudxr.env
export NV_CXR_ENDPOINT_IP=100.112.32.21 NV_CXR_MEDIA_PORT=47998
python -m isaacteleop.cloudxr --accept-eula --host-client

# Process B (env env_isaaclab) — the fiatlux XR sim:
conda activate env_isaaclab
source ~/.cloudxr/run/cloudxr.env          # sets XR_RUNTIME_JSON -> CloudXR
cd ~/robotica_project/fiatlux/fiatlux
export PYTHONPATH=$PWD/source/fiatlux_task
export DISPLAY=:1001                        # NX display (GUI needed for the AR panel)
python scripts/xr_teleop.py --task FIATLUX-Insert-Teleop-v0 --teleop_device handtracking
```
Then in the Isaac Sim UI: **AR panel** → Output Plugin **OpenXR**, Runtime **System OpenXR Runtime**
→ **Start AR** (viewport shows two eyes). Connect the Pico web client
(`https://100.112.32.21:48322/client/`) → the fiatlux scene renders in the headset. Press **Play**
in the headset UI (or `R` to reset) and move your right hand to drive the G1.

### Status / open items
- ✅ env cfg + `xr_teleop.py` written; headless env load verified.
- ✅ Launches to `Teleop ready`; the `handtracking` `OpenXRDevice` initializes ("Using teleop device:
  OpenXR Hand Tracking Device"). Fixed a latent Isaac Lab bug on the way: `remove_camera_configs`
  deletes the policy obs term by the *scene-camera* name — our obs term is `wrist_rgb` (scene camera
  `wrist_camera`), so it crashed; `xr_teleop.py` now uses a corrected local `_strip_camera_configs`.
- ✅ **RENDERING WORKS END-TO-END.** The fiatlux **G1 scene renders live in the Pico over Tailscale**
  via CloudXR. The earlier `Start AR` crash (`Simulation view object is invalidated ... Failed to get
  root link transforms`) was `env.step()` firing *during* the AR transition (PhysX view transiently
  invalidated); `xr_teleop.py`'s loop is now hardened (catch + keep rendering, don't die). Proven with
  the stock Isaac Lab Franka XR task (`teleop_se3_agent.py --task Isaac-Stack-Cube-Franka-IK-Abs-v0
  --teleop_device handtracking`), which sails through Start AR → Isaac Sim 5.1 + CloudXR-6.2 pipeline
  is sound. `XR_ERROR_FORM_FACTOR_UNSUPPORTED (2→1)` is a harmless probe.
- ❌ **CONTROL BLOCKER — hand tracking is a dead end on the Pico web client.** With controllers down,
  the CloudXR web client does NOT stream real optical hand joints — it falls back to a head-locked
  `'Right Head Device Hand'`, so Isaac Lab's hand retargeter gets garbage (same in both the
  isaacteleop gripper test and Isaac Lab). The Pico DOES reliably stream **controllers**
  (`bytedance/pico4_controller`, pose + trigger, equipped in the hand slots).
- ➡️ **Path forward = controller retargeter.** Isaac Lab's `OpenXRDevice` exposes
  `TrackingTarget.CONTROLLER_LEFT/RIGHT` (enum 3/4), populated with controller pose+inputs **only if
  a retargeter declares `RetargeterBase.Requirement.MOTION_CONTROLLER`**. No stock *simple-arm*
  controller retargeter exists (Isaac Lab's are all G1 whole-body). **TODO:** write a small
  controller→EE retargeter (right controller grip pose → absolute EE target, trigger → grip), add as
  an *optional* teleop device (`--teleop_device controller`) alongside `handtracking`.
- ⏳ **XR anchor pose** `(0.5, 0.7, 0.0)` is a guess — tune in-headset so the operator faces the table
  with the hand reaching the props.
- No `--enable_pinocchio` needed (Se3Abs + Gripper retargeters don't use dex/Pink-IK).

## 0. Prerequisites

- A **local machine with an NVIDIA GPU** (CloudXR renders + streams there).
- A **Meta Quest** (2 / 3 / Pro), charged, on the same network as the machine (or reachable via
  Tailscale — see §4).
- Isaac Lab installed (IsaacTeleop targets Isaac Lab 3.0; our env runs on 2.3.2 — we bypass its
  Isaac Lab integration and only use its XR **device** layer, so the version gap is not blocking).

---

## 1. Install IsaacTeleop (separate venv)

```bash
python3 -m venv ~/venvs/isaacteleop
source ~/venvs/isaacteleop/bin/activate
pip install 'isaacteleop[cloudxr,retargeters]~=1.0.0'
git clone https://github.com/NVIDIA/IsaacTeleop.git   # for the example scripts
```

## 2. Connect the Quest

1. Find the PC's IP: `hostname -I` (a LAN `192.168.x`/`10.x`, or the Tailscale `100.x` — see §4).
2. Open firewall ports on the PC: **`47998/udp`** and **`49100,48322/tcp`**.
3. Put on the Quest → open the **browser** → go to `https://nvidia.github.io/IsaacTeleop/client`.
4. Enter the **PC's IP**, click the link to accept the **self-signed certificate** (proceed through
   the browser warning), then click **Connect**.
   - No Quest app to install. To test *without* a headset, open that URL in a **desktop browser** —
     an emulator loads automatically.

## 3. Verify IsaacTeleop works (before touching our env)

```bash
cd IsaacTeleop
python examples/teleop/python/gripper_retargeting_example_simple.py
```
Squeeze the right controller trigger — the printed `gripper_command` value should change. This
proves **Quest → CloudXR → PC** works. Do **not** proceed until this passes.

## 4. Network: local vs remote

- **Local (best):** Quest and PC on the **same WiFi/LAN** → lowest latency. Use the PC's LAN IP.
- **Remote (Tailscale):** the Quest streams to a remote PC over the internet (see §4a).

### 4a. Remote from home (laptop remotes into a remote GPU PC) — via Tailscale

Topology: you're at home with a **Quest + laptop**; the GPU PC (IsaacTeleop + sim) is remote (e.g.
this box, Tailscale IP `100.112.32.21`).

> **Important:** your laptop's **SSH / VSCode remote is just a terminal tunnel — it does NOT carry
> the VR video.** CloudXR needs a **direct network path from the Quest to the PC**. The laptop is
> irrelevant to the VR data path; the Quest talks to the PC itself.

CloudXR is designed for streaming VR from a remote/cloud GPU, so this works — the trick is getting
the Quest onto the same Tailscale tailnet as the PC:

1. **Enable Quest developer mode** (Meta Horizon phone app → your headset → Developer Mode → on).
2. **Sideload the Tailscale Android APK** onto the Quest (Tailscale isn't in the Quest store):
   - Get `tailscale.apk` (from `tailscale.com/download/android` / their GitHub releases), then
   - `adb install tailscale.apk`  (USB), **or** use **SideQuest** to install the APK.
3. **Launch Tailscale** on the Quest (Apps → *Unknown Sources* → Tailscale) and **log into the same
   tailnet** as the PC. Confirm the PC (`100.112.32.21`) shows up / pings.
4. On the PC, make sure Tailscale is up (`tailscale status`) and CloudXR is bound to the Tailscale
   interface. Ports `47998/udp`, `49100,48322/tcp` are reachable **within the tailnet automatically**
   (Tailscale/WireGuard flattens NAT — no port-forwarding needed).
5. In the Quest browser, open the IsaacTeleop client, enter **`100.112.32.21`**, accept the cert,
   **Connect** (§2).

**Latency:** home ↔ remote server is a WAN hop, so expect more latency than local. Teleop tolerates
this better than fast VR games (you're guiding a hand, not head-tracking gameplay), and CloudXR
predicts/reprojects to hide some — so it's usually *usable but not crisp*. For real data collection
prefer a **local GPU box on the same WiFi**; use Tailscale-remote when that isn't available.

## 5. Bridge it into our Insert env

Once §3 passes, use `vr_teleop.py` (the starting-point bridge). It reuses `insert_teleop.py`'s action
interface and maps **controller pose → EE target, trigger → grip**. Two things to finalize on the
machine (marked `CONFIRM` in the script):

1. **Read the controller pose.** The simple example only exposes the gripper; the controller tracker
   also carries the 6-DoF pose. Add `print(result)` after `session.step()`, read the schema, and fill
   in `_read_controller()` (right controller position + quaternion + trigger).
2. **Frame alignment.** Set `FRAME_ALIGN` so pushing the controller *forward* moves the hand *forward*
   (VR is typically Y-up / +Z toward you; the G1 base is Z-up — often a +90° rotation about X).

Run (in an env that has **both** `fiatlux_task`/Isaac Lab **and** `isaacteleop` importable):
```bash
python scripts/vr_teleop.py --task FIATLUX-Insert-Teleop-v0
```
Then move `vr_teleop.py` into `scripts/` once it's validated (keep `isaacteleop` an optional/external
dependency — do not add it to the benchmark's `pyproject.toml`).

---

## Commands cheat-sheet

**On the remote GPU PC:**
```bash
# clone the working env so the extension install can't break it
conda create -y --name env_teleop --clone env_isaaclab
conda activate env_teleop

pip install 'isaacteleop[cloudxr,retargeters]~=1.0.0'   # extension only — NOT in pyproject.toml
git clone https://github.com/NVIDIA/IsaacTeleop.git

tailscale status                 # note the 100.x IP
sudo ufw allow 47998/udp; sudo ufw allow 49100/tcp; sudo ufw allow 48322/tcp   # if ufw is active

# verify (connect the Quest first, then):
cd IsaacTeleop && python examples/teleop/python/gripper_retargeting_example_simple.py
```

**On a computer with the Quest on USB (sideload Tailscale):**
```bash
# enable Quest Developer Mode in the Meta Horizon phone app first
adb devices                      # approve the USB prompt in-headset
adb install tailscale.apk        # from tailscale.com/download/android
# then in-headset: open Tailscale, log into the SAME tailnet as the PC
```

**In the Quest:** browser → `https://nvidia.github.io/IsaacTeleop/client` → enter `100.112.32.21`
→ accept cert → **Connect**.

**Run the bridge (after §3 passes + `vr_teleop.py` is finalized):**
```bash
conda activate env_teleop
cd ~/robotica_project/fiatlux/fiatlux
export PYTHONPATH=$PWD/source/fiatlux_task
export DISPLAY=:1001
python ~/robotica_project/fiatlux/vr_teleop/vr_teleop.py --task FIATLUX-Insert-Teleop-v0
```

> If `pip install isaacteleop` tries to downgrade/upgrade torch or numpy, stop — that's the
> version-conflict risk. Use a plain `python3 -m venv` for the *verify* step only, and switch the
> bridge to a two-process (socket) design so the sim env stays clean.

## Troubleshooting

- **Client won't connect:** ports not open, wrong IP, or Quest not on the same network/tailnet.
- **Cert warning:** expected (self-signed) — proceed past it.
- **Hand moves the wrong way:** tune `FRAME_ALIGN` (§5.2).
- **Laggy over Tailscale:** switch to the same LAN for real sessions.
- **`isaacteleop` import clashes with Isaac Lab:** the bridge needs both in one interpreter; if the
  pins conflict, keep IsaacTeleop's device process separate and pass the controller pose over a socket
  (advanced — only if a single env can't hold both).
