# VR Teleoperation of the Unitree G1 for the fiatlux "Insert" Task

**Issue #51 — teleoperation harness.** An operator wearing a VR headset drives the simulated Unitree
G1 humanoid, in real time and remotely, to perform the light-bulb insertion task (grasp a bulb and
seat it in a socket) inside Isaac Sim.

---

## 1. Objective

Let a human operator control the G1's arms and hands **spatially** — moving their own hands to move the
robot's hands — from a VR headset, with the operator at home and the GPU workstation in the lab. The
target task is `FIATLUX-Insert-Teleop-v0`: a stationary G1 with Inspire hands at a bench, a socket
fixed on the table, and a graspable bulb.

---

## 2. System architecture

Everything below the retargeter is off-the-shelf; the retargeter is the custom bridge we wrote.

| Layer | Component | Role |
|---|---|---|
| Simulation | **Isaac Sim 5.1 + Isaac Lab** | Physics + the fiatlux G1/bulb/socket scene; renders the stereo XR view |
| Streaming | **NVIDIA CloudXR 6.2** (via **IsaacTeleop**) | Streams the rendered scene to the headset; streams controller pose + buttons back |
| Headset app | **CloudXR web client** (Pico browser) | The page the operator connects from and enters AR |
| Transport | **Tailscale** | Private VPN so the remote headset reaches the lab PC over the internet |
| Input API | **OpenXR** | Standard API through which Isaac Lab reads the controllers (pointed at CloudXR) |
| Control (ours) | **`Se3RelControllerRetargeter`** | Converts a controller into a robot end-effector command |
| Robot control | **Differential IK** (Isaac Lab) | Converts the desired hand pose into arm joint angles |

### Data flow

```
Pico controllers ──(CloudXR / WebRTC over Tailscale)──▶ CloudXR runtime ──(OpenXR)──▶ Isaac Lab
   pose + trigger + grip                                                                  │
                                                                                          ▼
                                     OpenXRDevice  ─▶  Se3RelControllerRetargeter  ─▶  Differential IK  ─▶  G1 arm joints
                                    (reads controller)   (controller → EE pose cmd)     (pose → joint angles)
```

The video path runs the other way: Isaac Lab renders the stereo view, CloudXR encodes and streams it to
the Pico over Tailscale.

### Hardware
- **Pico 4 Ultra Enterprise** headset + its two 6-DoF controllers (controllers, not optical hand
  tracking — the web client streams controllers reliably, hand joints it does not).
- Remote **GPU workstation** (NVIDIA RTX 4090) running Isaac Sim + the CloudXR runtime.

---

## 3. Control approach

### 3.1 Retargeter (controller → end-effector command)
Isaac Lab's `OpenXRDevice` exposes the controllers' 6-DoF pose and button values. Our
`Se3RelControllerRetargeter` turns those into an **end-effector (hand) pose command** each frame:

- **Position** — the controller's frame-to-frame motion is integrated into an accumulated hand-target
  position (relative, 1:1). Moving your hand 10 cm moves the robot hand 10 cm.
- **Rotation** — the controller's twist is applied to the wrist orientation (1:1).
- **Grasp** — the trigger closes/opens the Inspire hand.
- **Clutch** — the arm only tracks while the **grip** button is held; releasing it freezes the arm, so
  you can reposition your own hand and re-engage (a "ratchet"). This is the key to precise, drift-free
  control.
- **Bimanual** — the left controller drives the left arm, the right controller the right arm,
  independently.

### 3.2 Differential Inverse Kinematics (what turns a hand pose into joint angles)
The retargeter outputs *where the hand should be*; the robot is controlled by *joint angles*. **Inverse
kinematics (IK)** is the mapping from a desired end-effector pose back to the joint angles that achieve
it. **Differential IK** solves this **incrementally, every timestep**, using the arm's **Jacobian**
`J` — the matrix relating small joint changes to small hand-pose changes (`Δx = J·Δq`):

1. Compute the pose error `Δx = target_pose − current_hand_pose`.
2. Invert the relationship to get the joint change: `Δq = J⁺·Δx` (`J⁺` = pseudo-inverse of the
   Jacobian).
3. Apply `Δq` and repeat next frame, so the arm smoothly converges onto the moving target.

We use the **damped-least-squares (DLS)** variant, `Δq = Jᵀ(JJᵀ + λI)⁻¹Δx`, which stays stable near
singular arm configurations (where a plain inverse would blow up). It's called *differential* because
it works on small differences each step rather than solving the whole pose analytically — which is
exactly what you want for smooth, responsive real-time teleoperation. This is Isaac Lab's
`DifferentialInverseKinematicsAction` (`command_type="pose"`, `ik_method="dls"`).

---

## 4. Key engineering problems solved

1. **Coordinate-frame bug (the decisive fix).** The IK interprets its target in the robot's **root
   (pelvis) frame**, but the retargeter was emitting **world-frame** poses. Because the pelvis sits
   ~0.75 m up and is rotated 90°, a world target was read as an impossible pose and the arm flew up and
   oscillated *even when the controller was still*. Fix: transform the target world→root before the IK.
2. **Clutch** — grip gates motion, eliminating drift when the controller is set down.
3. **Jitter handling** — CloudXR's controller tracking is noisy; EMA smoothing + per-frame
   spike-rejection keep the arm from "whacking around," and untracked/dropout poses are rejected.
4. **Full reset** — Isaac Lab's device reset didn't reset our retargeter, so a reset left a stale target
   and the arm moved on its own; now reset returns the target to rest.
5. **Wrist rotation** — controller twist mapped into the wrist (root-frame quaternion transform).
6. **Remote connection** — three real blockers: the web client must connect on the **secure proxy port
   48322** (not the raw backend 49100); the runtime must run with **`--host-client`** to serve the page;
   and the self-signed TLS cert needs the **Tailscale IP in its SAN** so the headset browser can trust
   it. Streaming forms directly over Tailscale via `NV_CXR_ENDPOINT_IP`/`NV_CXR_MEDIA_PORT`.

---

## 5. Result

A working end-to-end loop, confirmed live: the operator sees the fiatlux scene in the Pico and drives
the G1 with **position + wrist rotation + grasp**, **both arms independently**, with a clutch and a
clean reset — smoothly, with no drift or jitter, remotely over Tailscale.

---

## 6. How to run

```bash
# one command brings up a clean runtime + sim and prints the connect steps:
bash scripts/restart_xr_teleop.sh
```
Then: in Isaac Sim → **Start AR**; on the Pico → `https://<tailnet-ip>:48322/client/` → accept the cert
→ Server IP `<tailnet-ip>`, **Port 48322**, Device Profile *Pico 4 Ultra* → **Connect**. Hold **grip**
to move an arm, **trigger** to grasp, **R** to reset.

Full setup + gotchas: `journal/specs/vr-teleop-cloudxr-setup.md`.

---

## 7. Limitations / future work

- **Base is stationary** — the G1 stands still; a thumbstick base-glide or a walking policy would add
  locomotion.
- **Controller-tracking quality** is best on the *first* connection after a fresh runtime; reconnecting
  to a stale session degrades it (hence the restart script).
- Cosmetic: a kinematic **socket** logs harmless PhysX "must be non-kinematic" velocity-write warnings
  on reset.
- The insertion itself (bulb → socket) is now operator-driven; a success reward / autonomy is future
  work.
