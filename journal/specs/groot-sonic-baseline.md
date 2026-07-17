# GR00T N1.7 as the FIATLUX-Replace-v0 baseline

The benchmark's first real (non-smoke) baseline entry: **zero-shot GR00T N1.7**
(`nvidia/GR00T-N1.7-3B` base checkpoint, `REAL_G1` embodiment) + the **decoupled GEAR
whole-body controller** inside `FIATLUX-Replace-v0`. GR00T consumes the *standard*
observation mode (torso RGB + proprioception + a language instruction) — no privileged
state. Expected result: ~0 completions (published zero-shot GR00T numbers are 0% even on
tabletop G1 pick-place); the meaningful outputs are survival, the partial-progress
channels, and a working submission path for fine-tuned models.

**Why not SONIC:** the plan's original `UNITREE_G1_SONIC` path (VLA emits 64-d SONIC
motion tokens) requires a *finetuned* checkpoint — the base N1.7 release ships no SONIC
action head (its G1 head is `REAL_G1`, projector-shared with
`unitree_g1_full_body_with_waist_height_nav_cmd`), and no public SONIC finetune exists
(NVIDIA's `GR00T-N1.7-ApplePnP-V1` is a G1 finetune but of the *decoupled* embodiment,
shipped as deployment ONNX with the instruction baked in). The SONIC decoder integration
is kept (`SonicDecoder` + the `sonic_stand` gate) for when a finetuned checkpoint exists.

## Architecture

```
Isaac Lab process (fiatlux venv)                GR00T venv process
┌────────────────────────────────────┐          ┌─────────────────────────┐
│ FIATLUX-Replace-v0                 │  ZMQ REQ/│ Isaac-GR00T PolicyServer│
│  └─ GrootPolicy (adapter)          │  REP     │ (N1.7-3B base, REAL_G1) │
│      ├─ 2 ego frames + wrist FK +  │◄────────►└─────────────────────────┘
│      │  joint state → server       │  msgpack-numpy
│      ├─ chunk: arms/waist targets, │
│      │  navigate + height commands │
│      ├─ GearWbcDecoder (Balance/   │
│      │  Walk ONNX, 50 Hz, in-proc) │
│      └─ torque retarget + inv.     │
└────────────────────────────────────┘
```

- The 3B VLA cannot be a TorchScript artifact (HF multimodal backbone, string inputs,
  flow-matching loop, flash-attn kernels): it runs in its own CUDA 12.8 / py3.12 uv venv
  behind NVIDIA's own PolicyServer, and the benchmark side is a thin ZMQ client.
- Both WBC decoders *are* plain artifacts (≤40 MB ONNX) and run in-process on CPU.

## External stack (all under `~/tools/`)

- `~/tools/Isaac-GR00T` — the VLA repo (`uv sync`; needs `UV_HTTP_TIMEOUT` bumped).
  Server: `scripts/groot/serve.sh` → `run_gr00t_server.py --model-path
  nvidia/GR00T-N1.7-3B --embodiment-tag REAL_G1 --port 5555`. The VLM backbone
  `nvidia/Cosmos-Reason2-2B` is a gated HF repo (auto-approve): accept its license +
  `hf auth login` before first start.
- `~/tools/GR00T-WholeBodyControl` — WBC source. The decoupled Balance/Walk ONNX pair
  lives *in the repo* under `decoupled_wbc/sim2mujoco/resources/robots/g1/policy/`
  (`git lfs pull --include` that path).
- `~/tools/sonic_models/policy/release/` — SONIC `model_decoder.onnx` (+ encoder, unused:
  it encodes motion-tracking / teleop / SMPL references, not the VLA path) and
  `observation_config.yaml`, from `download_from_hf.py` (HF `nvidia/GEAR-SONIC`).
- GR00T-N1.7-3B checkpoint in the HF cache (`hf download nvidia/GR00T-N1.7-3B`).

## REAL_G1 VLA contract (probed against the live server)

- Observation: `video.ego_view` **(1, 2, H, W, 3) uint8** — delta_indices `[-20, 0]`,
  i.e. the frame from 0.4 s ago plus the current one; `state.{left,right}_wrist_eef_9d`
  (xyz + rot6d wrist pose — computed here by FK relative to the pelvis),
  `state.{left,right}_arm` (7 each), `state.waist` (3), `state.{left,right}_hand`
  (7 each, Dex3 convention — sent as zeros), all `(1, 1, ·) float32`;
  `language["annotation.human.task_description"] = [[prompt]]`.
- Action chunk (40 steps): `left/right_wrist_eef_9d` (RELATIVE, unused here),
  `left/right_arm` (RELATIVE joints, decoded absolute against the state we send),
  `waist` (3), `left/right_hand` (Dex3, unused), `base_height_command` (1),
  `navigate_command` (3). Server round-trip ≈ **0.11 s** on the 3090.
- Zeros for the wrist rot6d crash the server pipeline ("SVD did not converge") — always
  send a valid rotation.
- The adapter maps: `navigate_command` → WBC nav cmd, `base_height_command` → WBC height,
  VLA `waist` joint targets → WBC torso-rpy command (small-angle stand-in for the
  reference pipeline's FK), `left/right_arm` → direct joint-position targets. Chunks are
  re-fetched every 20 env steps; the sim clock stops during the request, so no latency
  compensation.

## Decoupled GEAR WBC contract (`decoupled_wbc`, mirrored in `GearWbcDecoder`)

- Balance/Walk ONNX pair `[B, 516] → [B, 15]`; Balance serves `|nav cmd| < 0.05`.
- 516 = 6-frame history (oldest first, zero-padded at episode start) of an 86-d frame:
  `[cmd*(2,2,0.5), height, rpy_cmd, ω*0.5, projected gravity, (q29 − defaults),
  dq29*0.05, last_action15]`, q/dq in URDF body order (legs, waist, arms) with defaults
  only for the lower 15.
- `q_target15 = action*0.25 + defaults15`; plant gains kp `[150,150,150,200,40,40]×2 +
  [250]×3`, kd `[2,2,2,4,2,2]×2 + [5]×3`; torque-retargeted through the env's own PD
  (same scheme as SONIC below).

## SONIC decoder contract (reverse-engineered from `gear_sonic_deploy`)

Verified against the ONNX graph: input `obs_dict [1, 994]`, output `action [1, 29]`.

Input layout, in `observation_config.yaml` order (the "436" comment in that file is
stale — the 10-frame names are what the graph wants):

| slice | obs | notes |
| --- | --- | --- |
| 64 | `token_state` | from the VLA (or the standing token) |
| 30 | `his_base_angular_velocity_10frame_step1` | base-frame gyro, raw |
| 290 | `his_body_joint_positions_10frame_step1` | **(q − default_angles), IsaacLab order** |
| 290 | `his_body_joint_velocities_10frame_step1` | raw, IsaacLab order |
| 290 | `his_last_actions_10frame_step1` | raw previous policy outputs |
| 30 | `his_gravity_dir_10frame_step1` | `quat_conj(base_quat) ⊗ (0,0,−1)` = projected gravity |

Histories: 10 frames at the 50 Hz control rate, **oldest first**, backfilled with the
current frame at episode start.

Output: 29 actions in IsaacLab BFS order. Hardware target (per MuJoCo/URDF index `i`):
`q_target[i] = default_angles[i] + action[isaaclab_of(i)] * g1_action_scale[i]`, where
`g1_action_scale = 0.25 * effort_limit / stiffness` per motor type, and SONIC's plant
gains/defaults come from `policy_parameters.hpp` (armature-derived kp/kd, e.g. hips
kp≈99/kd≈13, arms kp≈14/kd≈1.8; standing pose hip_pitch −0.312, knee 0.669,
ankle_pitch −0.363, shoulder_pitch 0.2, elbow 0.6).

Constants land in `fiatlux_task/groot.py` (`SONIC_*`), joints matched **by name** to our
articulation (our G1 shares the 29 URDF body-joint names; the Inspire fingers are not
SONIC's problem — held at defaults).

Standing latent for the no-VLA milestone: `LATENT_INITIAL_MOTION_TOKEN` in
`gear_sonic/utils/inference/initial_poses.py` (64-d, checkpoint-specific).

## Gain mismatch and action inversion (adapter-side, env untouched)

Our env's action space is `JointPositionActionCfg(scale=0.5, use_default_offset=True)`
with the benchmark's own actuator gains (uniform legs kp 200/kd 10 ≠ either WBC's
plant). Both decoders convert the controller's intended behavior into our action units
per step:

1. Controller torque intent: `τ = kp_c (q_des − q) − kd_c q̇` (per joint, its gains).
2. Torque-equivalent target under our PD: `q_t = q + (τ + kd_ours q̇) / kp_ours`.
3. Our action: `a = (q_t − default_ours) / 0.5` (targets clamped to joint limits).

This matches the commanded torque at the moment of computation; the *impedance* across
the 20 ms hold still differs (our stiffer PD). If a controller is unstable under our
plant, that is reported as a finding — the benchmark env is not retuned for a baseline.

## Wire protocol (Isaac-GR00T PolicyServer)

Request (msgpack_numpy over ZMQ REQ/REP):
`{"endpoint": "get_action", "data": {"observation": obs, "options": None}}` →
response `[action, info]`. `{"endpoint": "ping"}` for liveness,
`get_modality_config` for the embodiment's obs/action spec. Reimplemented in
`fiatlux_task/groot.py::_Gr00tClient` (~40 lines; the Isaac venv only gains
`pyzmq` + `msgpack-numpy` + `onnxruntime`).

## Startup blend + domain gaps (documented, zero-shot)

- Both stand-gate policies and the baseline blend the controller's joints from the
  spawn pose to the controller's standing pose over the first 1 s of each episode
  (the deploy stacks' "blend to initial pose"); histories/actions engage after
  handover. Without it the controllers receive a ~0.5 rad step input at t=0.
- Our camera is torso-mounted, not the real G1's head-mounted ego view; our hands are
  Inspire, not Dex3 (hand states sent as zeros, hand actions ignored, fingers held).
  Acceptable zero-shot domain gaps, documented with the results.

## Findings log

- Driver is CUDA 12.6 (560.35.05) vs GR00T's CUDA 12.8 target: torch 2.9 cu128 +
  flash-attn 2.8.3 cp312 verified working (CUDA available, model loads, 0.11 s
  inference).
- `uv sync` in Isaac-GR00T needs `UV_HTTP_TIMEOUT=300` (wandb wheel download timeout).
- Base N1.7 checkpoint supported tags (from the server's own error): OXE_DROID…,
  **REAL_G1**, REAL_R1_PRO_SHARPA…, XDOF… — no `UNITREE_G1` / `UNITREE_G1_SONIC`
  posttrain heads.
- G1 spawn height must be 0.79 (settled standing height 0.787 m at the bent-knee
  pose); lower spawns interpenetrate the floor and PhysX launches the robot at reset.
- The startup blend window must be ≤ 1 control step: the spawn pose is not statically
  stable (passive tip-over by ~1.3 s), so the balancer has to own the robot from the
  first tick. Swept 0.02/0.2/0.5/1.0 s; only the 1-step handover gives full stands.

## Results (seed 0, 20 episodes, protocol runs)

| policy | mean episode len | success | terminations |
| --- | --- | --- | --- |
| `zero` / `basic_standard` | 62.0 | 0 | fell_over 100% |
| `wbc_stand` (gate, 4 eps) | 2000.0 | — | time_out 100% |
| `sonic_stand` (gate, 4 eps) | 1068.25 | — | time_out 50% |
| `groot` (zero-shot) | 990.4 | 0 | time_out 20%, fell_over 55%, fell_below 40% |

`groot` diagnostics: server latency ≈ 118 ms/chunk, `nav_cmd_norm` ≈ 0.05 (the VLA
holds position rather than walking — zero-shot, all progress channels 0). Artifacts:
`logs/runs/groot-replace-seed0/` (eval.json + video/run.mp4).

Instruction-following probe (same live frame, eight prompts from the canonical task
sentence to atomic "walk forward" / "turn left" / "crouch down"): commanded velocities
stay in a 0.015–0.07 noise band and the height command never moves — the base
checkpoint's behavior in this domain is language-invariant. Steering it at the ladder
requires fine-tuning on task demos, not prompt engineering.
- ONNX decode cost is negligible on CPU (SONIC decoder 0.36 ms; WBC nets are 1.9 MB).
