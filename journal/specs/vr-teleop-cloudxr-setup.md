# VR (Pico 4 Ultra) teleoperation — setup guide

Goal: drive the G1's right hand in `FIATLUX-Insert-Teleop-v0` with a **Pico 4 Ultra** via NVIDIA
**IsaacTeleop** (CloudXR streaming), instead of the keyboard. The env already takes an *absolute
end-effector pose + a binary grip* — exactly what a VR controller provides — so VR just swaps the
input source.

> **Note:** IsaacTeleop / CloudXR are an **extension dependency**, kept *out* of the benchmark's
> requirements. Install them in a **separate venv**, not in `env_isaaclab`.

---

## ⭐ Findings & working recipe — Pico 4 Ultra Enterprise, remote over Tailscale (2026-07)

Hard-won results from getting a **Pico 4 Ultra Enterprise** connected to a **remote** GPU PC
(`<GPU_HOST>`, RTX 3090, Tailscale `<TAILNET_IP>`) over Tailscale. Read this before
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
   export NV_CXR_ENDPOINT_IP=<TAILNET_IP>   # the PC's Tailscale IP
   export NV_CXR_MEDIA_PORT=47998            # media UDP port (both IP *and* port required —
                                             # with only the IP set, the log shows port=0 and it
                                             # still fails: "endpoint not configured")
   python -m isaacteleop.cloudxr --accept-eula --host-client
   ```
   Confirm in `~/.cloudxr/logs/cxr_streamsdk.*.log`: `Added public endpoint candidate:
   <TAILNET_IP>:47998`. After this, the media connects **directly over Tailscale** — no TURN needed.

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
  export NV_CXR_ENDPOINT_IP=<TAILNET_IP> NV_CXR_MEDIA_PORT=47998 &&
  python -m isaacteleop.cloudxr --accept-eula --host-client` (ports 48322 + 49100 listen).
- Headless input smoke test (no scene, black headset — expected):
  `python examples/teleop/python/gripper_retargeting_example_simple.py`. It must be launched **after**
  the headset is connected (else `Failed to get OpenXR system: -35`).
- Pico connect URL: `https://<TAILNET_IP>:48322/client/` → accept the self-signed cert on the
  `:48322/` page (separate tab), then **Connect** on `/client/`, then **Play** (= Start Teleop).

---

## ⭐⭐ Isaac Lab NATIVE XR teleop — the working integration path (2026-07)

> **REMOVED 2026-08-09** — this hand-tracking path (`xr_teleop.py`) was deleted; teleop now uses
> **whole-body control** (`scripts/teleop/sonic_teleop.py --input vr|keyboard`). Kept as history; the
> CloudXR connection steps below still apply.

**This supersedes the IsaacTeleop device path.** IsaacTeleop's own Isaac Lab device
(`IsaacTeleopDevice`) targets Isaac Lab 3.0 / Isaac Sim 6.0 — we run **Isaac Sim 5.1.0 / Isaac Lab
0.54.3**, so instead we use **Isaac Lab's own** built-in OpenXR teleop (`isaaclab.devices.OpenXRDevice`
+ retargeters). Isaac Sim renders the sim itself and streams it via CloudXR; the headset sends **hand
tracking** (pinch), which sidesteps the controller-trigger gap entirely.

### Architecture (two processes, no dep conflict)
- **Process A — CloudXR runtime** (conda env `vr_teleop`): `python -m isaacteleop.cloudxr
  --accept-eula --host-client` with `NV_CXR_ENDPOINT_IP=<TAILNET_IP>` + `NV_CXR_MEDIA_PORT=47998`.
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
export NV_CXR_ENDPOINT_IP=<TAILNET_IP> NV_CXR_MEDIA_PORT=47998
python -m isaacteleop.cloudxr --accept-eula --host-client

# Process B (env env_isaaclab) — the fiatlux XR sim:
conda activate env_isaaclab
source ~/.cloudxr/run/cloudxr.env          # sets XR_RUNTIME_JSON -> CloudXR
cd ~/robotica_project/fiatlux/fiatlux
export PYTHONPATH=$PWD/source/fiatlux_task
export DISPLAY=:1001                        # NX display (GUI needed for the AR panel)
python scripts/teleop/sonic_teleop.py --task FIATLUX-Insert-Teleop-v0 --input vr   # (was scripts/xr_teleop.py)
```
**Connect the Pico (the exact steps that work — ✅ confirmed 2026-07-26):**
1. In the Isaac Sim UI: **AR panel** → Output Plugin **OpenXR**, Runtime **System OpenXR Runtime** →
   **Start AR**.
2. On the Pico browser open `https://<TAILNET_IP>:48322/client/` → cert warning → **Advanced →
   Proceed** (the cert now carries the tailnet IP in its SAN, so it's bypassable, not a hard block).
3. In the client Settings: **Device Profile = Pico 4 Ultra**, **Server IP = <TAILNET_IP>**,
   **Port = 48322** ⬅ **CRITICAL: NOT the default 49100.** 49100 is the raw CloudXR backend (no TLS);
   the browser must hit the **WSS proxy on 48322**, which terminates TLS and bridges to 49100.
4. Tap **Connect** → the fiatlux scene streams to the headset.
5. Drive: **hold grip** on a controller to move that arm (release = freeze/clutch), **trigger** =
   grasp. Left controller → left arm, right → right arm. `R` = full reset.

**Gotchas that cost hours (all fixed/worked-around):**
- **Port 48322 not 49100** (above) — the single biggest connect blocker.
- **`--host-client`** is required or nothing serves the web page on 48322 (`run()` alone only starts
  49100/47998). The runtime prints the LAN URL (`<LAN_IP>`); remote, use the tailnet IP.
- **Self-signed cert**: regenerate with the tailnet IP in the SAN so the Pico browser can proceed:
  `openssl req -x509 -newkey rsa:2048 -nodes -keyout ~/.cloudxr/certs/server.key -out
  ~/.cloudxr/certs/server.crt -days 365 -subj "/CN=<TAILNET_IP>" -addext
  "subjectAltName=IP:<TAILNET_IP>,IP:127.0.0.1,DNS:localhost"`.
- **Greyed CONNECT button (intermittent):** even with valid settings + passed capabilities the button
  can stay disabled on a fresh load. Worked around with a tiny force-enable script appended to
  `~/.cloudxr/static-client/index.html` (re-enables the button when its label is "CONNECT"). Not in
  the repo (client is outside it); re-apply after any client re-download. Root cause not fully known.
- **Controller pose quality is best on a FIRST sim launch after a clean CloudXR runtime restart** —
  then poses are stable (~1 mm); on stale/nth launches they freeze at the anchor origin or jitter.

### Status / open items
- ✅✅ **FULL LOOP CONFIRMED WORKING END-TO-END (2026-07-26):** operator in the Pico drives the G1 arm
  live over Tailscale via `controller_rel` — scene streams to the headset, **hold-grip clutch moves
  the arm smoothly (no jitter/whacking), trigger grasps, both arms independent.** The chain of fixes
  that got here: (1) **world→root frame transform** in `Se3RelControllerRetargeter` output — the arm
  was oscillating because it fed WORLD poses to a ROOT-frame IK (see Problem 4); (2) **clutch** (grip
  gates motion); (3) **untracked/origin-pose rejection** (`min_valid_z`); (4) **EMA smoothing +
  spike-rejection** for CloudXR controller jitter; (5) **1:1 scale**; (6) connect via **Port 48322 +
  SAN cert + `--host-client`** (see Run recipe).
- ✅ **Reset now fully resets.** `OpenXRDevice.reset()` resets only its head/hand caches, NOT the
  retargeters — so the accumulated EE target survived a reset and the IK drove the arm on its own
  after `R`. `xr_teleop.py` now also resets every retargeter (`_pos`→rest, refs cleared) on reset.
- ✅ **Wrist rotation works (confirmed live "perfect").** Controller twist → wrist, clutch-gated,
  applied in the root frame (`dq_root = R_root⁻¹·dq_world·R_root`), with a rotational deadzone +
  per-frame cap; ratchets like position. Full 6-DoF now (`command_type="pose"`). Toggle via
  `Se3RelControllerRetargeterCfg.enable_rotation`; tune `rot_deadzone`/`rot_max_step` if ever twitchy.
- ✅ **One-command restart:** `bash scripts/restart_xr_teleop.sh` does the full clean cycle (kill sim
  + runtime, clear shm/run-state, start runtime `--host-client` + sim, wait, print connect steps).
  Needed because reconnecting a headset to a stale session degrades the CloudXR pose stream.
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
- A **Pico 4 Ultra** (Enterprise, as used here), charged, on the same network as the machine (or
  reachable via Tailscale — see §4).
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

## 2. Connect the Pico

1. Find the PC's IP: `hostname -I` (a LAN `192.168.x`/`10.x`, or the Tailscale `100.x` — see §4).
2. Open firewall ports on the PC: **`47998/udp`** and **`49100,48322/tcp`**.
3. Put on the Pico → open the **PICO Browser** → go to `https://nvidia.github.io/IsaacTeleop/client`.
4. Enter the **PC's IP**, click the link to accept the **self-signed certificate** (proceed through
   the browser warning), then click **Connect**.
   - No app to install. To test *without* a headset, open that URL in a **desktop browser** —
     an emulator loads automatically.

## 3. Verify IsaacTeleop works (before touching our env)

```bash
cd IsaacTeleop
python examples/teleop/python/gripper_retargeting_example_simple.py
```
Squeeze the right controller trigger — the printed `gripper_command` value should change. This
proves **Pico → CloudXR → PC** works. Do **not** proceed until this passes.

## 4. Network: local vs remote

- **Local (best):** Pico and PC on the **same WiFi/LAN** → lowest latency. Use the PC's LAN IP.
- **Remote (Tailscale):** the Pico streams to a remote PC over the internet (see §4a).

### 4a. Remote from home (laptop remotes into a remote GPU PC) — via Tailscale

Topology: you're at home with a **Pico + laptop**; the GPU PC (IsaacTeleop + sim) is remote (e.g.
this box, Tailscale IP `<TAILNET_IP>`).

> **Important:** your laptop's **SSH / VSCode remote is just a terminal tunnel — it does NOT carry
> the VR video.** CloudXR needs a **direct network path from the Pico to the PC**. The laptop is
> irrelevant to the VR data path; the Pico talks to the PC itself.

CloudXR is designed for streaming VR from a remote/cloud GPU, so this works — the trick is getting
the Pico onto the same Tailscale tailnet as the PC. On the **Pico 4 Ultra Enterprise** no developer
mode is needed (see the ⭐ Findings section above):

1. **Allow unknown apps** for the PICO Browser: **Settings → Security → Install unknown apps** → enable
   for the PICO Browser.
2. **Download the Tailscale APK directly** in the PICO Browser from
   `pkgs.tailscale.com/stable/tailscale-android-universal-<ver>.apk` (the Google-Play button needs a
   Google account; the direct APK does not) and install it.
3. **Launch Tailscale** on the Pico and **log into the same tailnet** as the PC. Confirm the PC
   (`<TAILNET_IP>`) shows up / pings.
4. On the PC, make sure Tailscale is up (`tailscale status`) and CloudXR is bound to the Tailscale
   interface. Ports `47998/udp`, `49100,48322/tcp` are reachable **within the tailnet automatically**
   (Tailscale/WireGuard flattens NAT — no port-forwarding needed).
5. In the PICO Browser, open the IsaacTeleop client, enter **`<TAILNET_IP>`**, accept the cert,
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

# verify (connect the Pico first, then):
cd IsaacTeleop && python examples/teleop/python/gripper_retargeting_example_simple.py
```

**On the Pico 4 Ultra Enterprise (sideload Tailscale — no dev mode / no PC needed):**
```text
Settings → Security → Install unknown apps → enable for the PICO Browser
PICO Browser → download pkgs.tailscale.com/stable/tailscale-android-universal-<ver>.apk → install
then in-headset: open Tailscale, log into the SAME tailnet as the PC
```

**In the Pico:** PICO Browser → `https://nvidia.github.io/IsaacTeleop/client` → enter `<TAILNET_IP>`
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

- **Client won't connect:** ports not open, wrong IP, or Pico not on the same network/tailnet.
- **Cert warning:** expected (self-signed) — proceed past it.
- **Hand moves the wrong way:** tune `FRAME_ALIGN` (§5.2).
- **Laggy over Tailscale:** switch to the same LAN for real sessions.
- **`isaacteleop` import clashes with Isaac Lab:** the bridge needs both in one interpreter; if the
  pins conflict, keep IsaacTeleop's device process separate and pass the controller pose over a socket
  (advanced — only if a single env can't hold both).

---

## ⛑️ Known problems & probable causes (issue #51 — teleoperate the fiatlux Insert task with the G1)

**Goal:** drive the G1 right arm + grip to insert the bulb in `FIATLUX-Insert-Teleop-v0`, via the
Pico 4 Ultra Enterprise over CloudXR, using Isaac Lab's native OpenXR teleop (`scripts/xr_teleop.py
--teleop_device controller_rel`). Env: Isaac Sim 5.1.0 / Isaac Lab 0.54.3, isaacteleop CloudXR 6.2.

### What works (proven live)
- Remote VR **streaming** Pico → CloudXR → PC over Tailscale (the `NV_CXR_ENDPOINT_IP=<tailnet-ip>` +
  `NV_CXR_MEDIA_PORT=47998` fix). The fiatlux **scene renders** in the headset.
- Controllers **register** (`bytedance/pico4_controller`, 18 inputs) and the **grasp works** —
  trigger → `ControllerGripperRetargeter` → G1 hand closes.
- The retargeter code path (reads `data[CONTROLLER_RIGHT]`, deadzone, spike-reject); when the
  controller **pose** did stream (one early session), the EE target tracked the controller in the
  correct direction. So the input→retarget→action pipeline is correct.
- **Bimanual** — left controller → left arm, right → right arm (mirrored joints/EE, probed rest pose);
  both follow when a valid pose flows. 16-dim action verified.
- **Clutch** — arm only tracks while grip/squeeze (input idx 3) is held; release → arm freezes and the
  reference is dropped. Confirmed live (clutch=1 tracking on the `sim_clutch` launch). This is what
  makes "set the controller down = arm stays put" work regardless of pose jitter.
- **Arm/IK + frame transform (root-caused & verified this session — see Problem 4).** With a correct
  root-frame command the arm holds a constant pose to **0.0 mm** and tracks a clean input **1:1**. The
  manipulation side is solved; the arm no longer oscillates at rest.

### Problem 1 — controller 6-DoF POSE stream is intermittent / dead  ⬅ current top blocker
- **Symptom:** the right controller's *position* reads **frozen at the anchor origin** (e.g.
  `[0.5, 0.7, 0.0]`) even while the operator moves it in big arcs; the *trigger* still works (grasp
  ok). So the arm has no position data to follow. Worked in one earlier session; dead in others.
- **Evidence + NARROWED (confirmed CloudXR-side):** the controller **model moves in-headset** (Pico
  tracks fine) AND CloudXR **is receiving poses** — `cxr_server` shows continuous `DevicePoseInterval
  ~15.5ms` / `PoseInterarrivalTime ~15.7ms`. Yet `[REL] rawpos` stays frozen at the anchor origin. So
  the pose reaches CloudXR but is **not forwarded to the OpenXR app** (Isaac Lab's
  `_query_controller` → `get_virtual_world_pose()` returns identity/origin). The break is the
  `cxr_server` line **`ERROR [processSystemInfo] Making device configuration`**, which fires
  immediately after the client SystemInfo advertising **`body_tracking: true, controller_haptics:
  true`** (then logs "Processed SystemInfo successfully" — so the failed sub-config is silently
  swallowed). Not a Pico setting (model moves); not our retargeter (it reads what it's given).
- **Probable causes:** (a) CloudXR 6.2 fails to build the controller **pose** device config for the
  Pico `bytedance/pico4_controller` profile (only inputs/haptics get wired, not the 6-DoF pose) — the
  Isaac Lab CloudXR path is designed for Apple Vision Pro *hand tracking* + CloudXR 5.0; Pico
  web-client **controllers** are "early access" and may be incompletely plumbed to OpenXR; (b) the
  client's `body_tracking: true` advertisement trips the device-config builder. It worked in ONE early
  session, so it's intermittent — a warm/degraded runtime state may also contribute.
- **Next-session investigation (CloudXR-side):** diff a working-vs-frozen session's `processSystemInfo`
  block; see if the Pico web client can be told NOT to advertise body tracking (URL param / setting);
  check whether the controller pose is meant to arrive via `NV_CXR_ENABLE_PUSH_DEVICES` and whether
  that path is configured for controllers; consider Isaac Lab's own Docker CloudXR runtime (the
  documented/tested one) instead of isaacteleop's 6.2 for the app-facing device config.
- **First, isolate Pico-side vs CloudXR-side (do this before changing any setting):** in the headset,
  does the **controller *model* move** when you move the controller?
  - **Yes (model moves in-headset)** → the Pico is tracking fine; the frozen pose is **CloudXR-side**
    (the `processSystemInfo` device-config failure / browser bridge). No Pico toggle will fix it —
    investigate the CloudXR device config, restart the runtime clean (or reboot), re-pair the client.
  - **No (frozen in-headset too)** → it's a **Pico-side** tracking/permission issue → check the
    settings below.
  It worked in one earlier session with no setting change, which points at the **CloudXR-side**
  (intermittent) cause, not a missing toggle.
- **Pico requirements the official docs DO state** (check these; "Advanced Tracking Features" from the
  Google AI summary is *not* confirmed in the IsaacTeleop/CloudXR docs and may describe the separate
  Unitree pipeline): **Pico OS 15.4.4U+** (Settings → General → About) and **PICO Browser 4.0.40+,
  "Enterprise enabled"** — the WebXR client runs in that browser, so a non-enterprise / outdated
  browser can drop the pose stream.
- **Pico settings worth checking (names vary by OS version):** Settings → Controllers (paired, fresh
  batteries); Settings → Motion Tracking / Hand & Controller Tracking (on); Settings → General →
  About → tap Software Version ~7× → Developer; enterprise VST/tracking may be gated by PICO Business
  Suite / device management (the owning org).
- **Then verify the fix** with the `[REL] rawpos` debug: raw pose should change when you move.

- **UPDATE (2026-07-26) — this is now the SOLE blocker, and it has three distinct failure modes** (all
  observed in one session's `[REL] raw=` debug):
  1. **Drops entirely** → `raw=None` (no controller device). Whole `sim_smooth` launch #5: 0 of 250,438
     frames had pose data. This is the operator's "then no longer in my controller" — the device
     disappears mid-session.
  2. **Freezes at the anchor origin** → `raw=[0.5, 0.7, 0.0]` (z=0) — the untracked controller reports
     the XR anchor default.
  3. **Jitters ~0.9 m frame-to-frame** when it *does* flow → the arm faithfully whacks around ("moves
     randomly once my controller moves"). Stable at rest, bad under motion → classic **inside-out
     tracking loss** (controller leaves the Pico headset-camera FOV while moving).
- **The architectural weak point** is `isaaclab .../devices/openxr/openxr_device.py::_query_controller`
  (~line 437): `pose = input_device.get_virtual_world_pose()` returns a pose **unconditionally, with no
  tracking-valid flag** — on tracking loss it silently hands back a default/stale pose (→ the origin
  freeze). **Investigation TODO:** check the `input_device` API for an is-tracked/pose-valid flag and
  gate on it instead of our `z<0.3` heuristic.
- **Ranked probable causes:** (1) controllers leaving the Pico inside-out tracking volume during motion
  (best fit for jitter+freeze-when-moving, stable-at-rest); (2) CloudXR device-config only partially
  exposing 6-DoF (the `processSystemInfo`/`body_tracking` race above); (3) reference-space recentering
  on head motion.
- **★ Strategic recommendation — switch to a tracking-INDEPENDENT control scheme.** Pose-based control
  is capped by this stream quality. Drive the EE with the **thumbstick** (→ EE *velocity*) + trigger for
  grasp, via `get_input_gesture_value("thumbstick", ...)`: thumbstick values are **digital inputs, not
  optical tracking**, so they are immune to FOV/jitter/dropouts. Less "natural" than 1:1 hand motion,
  but it would let the operator **actually complete the Insert task** instead of fighting tracking. Add
  it as a new retargeter selectable via `--teleop_device` (keep the pose one). This is the recommended
  next build.

### Problem 2 — CloudXR/Isaac-Sim AR session is fragile (only clean right after a reboot)
- **Symptom:** `Start AR` works on the **first ~2–3 launches after a fresh reboot**, then floods
  `XR_ERROR ... swapchain == NULL` + `[XR] Frame ... did not call ... EndFrame` and freezes.
- **Probable cause:** leaked GPU/Kit/CloudXR state across process kills — stale `carb`/`carbonite`
  shared memory in `/dev/shm` (from SIGKILL'd Kit procs), stale CloudXR run-state
  (`~/.cloudxr/run/{ipc_cloudxr,cloudxr.pid,runtime_started}`), and CloudXR-6.2 compositor session
  state — that a plain restart doesn't clear.
- **Mitigations tried:** clearing `/dev/shm/carb*` + `~/.cloudxr/run` state lets a **sim-only** restart
  survive ~1–2 more launches, but a full **reboot** is the only reliable reset. **Budget one code
  change per reboot; launch fresh; test; don't relaunch mid-session.**

### Problem 3 — CloudXR runtime won't cleanly restart after being killed
- **Symptom:** after `pkill`, `python -m isaacteleop.cloudxr` starts then immediately exits
  (`cxr_server` log: `Server exiting: '0'`), launcher exits 1 with no stdout.
- **Probable cause:** leftover shared resources (same class as Problem 2) block a clean service init.
- **Mitigation:** `rm ~/.cloudxr/run/{cloudxr.pid,ipc_cloudxr,runtime_started}` + `/dev/shm/carb*`
  helps but is unreliable; reboot is the sure fix.

### Problem 4 — arm oscillates / "moves randomly / raised high" — ✅ ROOT-CAUSED & FIXED (2026-07-26)
This was the "the robot moves randomly even when my controller is still" the operator reported many
times. The earlier hypotheses below (near-singular workspace / infeasible fixed orientation / garbage
input) were **wrong**. The real cause was a **coordinate-frame bug**.
- **Root cause — WORLD vs ROOT frame.** `DifferentialInverseKinematicsAction` interprets its command in
  the robot **ROOT (pelvis) frame** (`isaaclab .../mdp/actions/task_space_actions.py::_compute_frame_pose`
  → `subtract_frame_transforms(root, ee)`), but the retargeter was emitting **WORLD-frame** poses. The
  G1 root is at world `pos(0.5, 0.7, 0.75) quat_wxyz(0.7071,0,0,-0.7071)` (−90° yaw). So a world height
  z≈0.83 was read as a *root* z, commanding the hand to world z≈1.6 m — **unreachable** → the arm hunts
  upward and oscillates, **even for a dead-constant command**.
- **Proof (headless, `scratchpad/ik_stability_probe.py` + `frame_probe.py`):** feeding the WORLD-frame
  rest pose as a constant command → EE oscillates **40–56 cm peak-to-peak**, settles ~0.5 m high, torso
  stable to 0.6 mm (so it's the *arm*, not base sway, not the controller). Feeding the **ROOT-frame**
  rest pose → **0.0 mm** — rock steady. Position-only IK oscillated too, confirming it's the frame, not
  the orientation.
- **Fix (in `xr_controller_retargeters.py`, verified end-to-end via `verify_retarget.py`):** transform
  the accumulated world-frame target into the root frame at output — `pos_root = R^T (pos_w − root_pos)`
  with `root_pos`/`root_quat` cfg fields; set `initial_orientation` to the **root-frame** rest quats
  (right `(0.998,-0.006,0.059,0.001)`, left `(0.9975,-0.009,0.069,0.008)`); `initial_position` /
  `workspace_*` stay WORLD frame (the accumulator + clamp run in world, matching the controller poses).
  Verified: HOLD (still controller) → EE p2p ≤0.2 mm; MOVE controller +0.12 m world-x → EE +0.113 m.
  Live-confirmed by the operator: "initially the hands are in place" (was raised/oscillating before).

### Bottom line (updated 2026-07-26)
The **manipulation side is solved and verified**: the arm oscillation ("moves randomly / raised high"),
long blamed on IK/workspace, was a **world-vs-root frame bug** in the retargeter output (Problem 4) —
now fixed; the arm holds a constant command to 0.0 mm and tracks a clean input 1:1. Clutch, z-rejection,
bimanual, and grasp all work.

**The sole remaining blocker is the controller 6-DoF pose STREAM (Problem 1)** — it drops (`raw=None`),
freezes at the anchor origin, and jitters ~0.9 m under motion. This is upstream (Pico inside-out tracking
+ CloudXR device-config), not our code, and more retargeter tuning won't fix garbage input. Two paths:
(A) fix the stream — enable Pico advanced tracking, keep controllers in FOV, resolve the CloudXR
`processSystemInfo` device-config; or (B, **recommended**) add a **thumbstick→EE-velocity** retargeter
that is immune to tracking quality, so the Insert task can actually be completed. AR/runtime fragility
(Problems 2–3) still makes iterating slow — clearing `/dev/shm/carb*` between sim relaunches has been
keeping AR healthy without a reboot (launches #3–#5).
