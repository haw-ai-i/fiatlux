"""GEAR-SONIC policy I/O contract, verified against the released ONNX models.

Source of truth (checked 2026-08-24, GR00T-WholeBodyControl @ gear-sonic-v1.1):
  * ``gear_sonic_deploy/policy/release/model_{encoder,decoder}.onnx`` -- actual tensor shapes
  * ``gear_sonic_deploy/policy/release/observation_config.yaml`` -- which terms are enabled
  * ``gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp`` -- per-term dims

CAUTION: the YAML header comment claims "Total dimension: 436 (64+12+116+116+116+12)".
That describes the **4-frame** history variants and does NOT match the shipped model or the
YAML's own enabled list. The decoder really takes **994** (10-frame histories). Trust the
ONNX shapes, which this module reproduces and asserts.

    encoder: obs_dict[1, 1762] -> encoded_tokens[1, 64]
    decoder: obs_dict[1,  994] -> action[1, 29]        (29 = G1 body DOF: 12 leg + 3 waist + 14 arm)

The encoder is *modular*: its 1762-wide input carries the slots for ALL reference modes, and
``encoder_mode_4`` (one-hot, 4 wide) selects which are meaningful for this step.

    mode 0 "g1"      -- imitate a G1 motion clip (motion joint pos/vel + anchor orientation)
    mode 1 "teleop"  -- LOWER body from motion + VR 3-point head/hands targets   <- our Mode 2
    mode 2 "smpl"    -- SMPL human-skeleton joints + wrists
"""

from __future__ import annotations

import os

# --- decoder ("policy") input, in order -------------------------------------------------
DECODER_LAYOUT: tuple[tuple[str, int], ...] = (
    ("token_state", 64),                            # encoder output
    ("his_base_angular_velocity_10frame_step1", 30),   # 3 x 10 frames
    ("his_body_joint_positions_10frame_step1", 290),   # 29 x 10
    ("his_body_joint_velocities_10frame_step1", 290),  # 29 x 10
    ("his_last_actions_10frame_step1", 290),           # 29 x 10
    ("his_gravity_dir_10frame_step1", 30),             # 3 x 10
)

# --- encoder input, in order ------------------------------------------------------------
ENCODER_LAYOUT: tuple[tuple[str, int], ...] = (
    ("encoder_mode_4", 4),                              # one-hot mode select
    ("motion_joint_positions_10frame_step5", 290),      # mode 0
    ("motion_joint_velocities_10frame_step5", 290),     # mode 0
    ("motion_root_z_position_10frame_step5", 10),
    ("motion_root_z_position", 1),
    ("motion_anchor_orientation", 6),                   # 6D rotation representation
    ("motion_anchor_orientation_10frame_step5", 60),    # 6 x 10
    ("motion_joint_positions_lowerbody_10frame_step5", 120),   # mode 1: 12 legs x 10
    ("motion_joint_velocities_lowerbody_10frame_step5", 120),  # mode 1
    ("vr_3point_local_target", 9),                      # mode 1: head+2 hands, xyz each
    ("vr_3point_local_orn_target", 12),                 # mode 1: 3 x 4 (quat) or 2 x 6
    ("smpl_joints_10frame_step1", 720),                 # mode 2: 24 joints x 3 x 10
    ("smpl_anchor_orientation_10frame_step1", 60),      # mode 2
    ("motion_joint_positions_wrists_10frame_step1", 60),# mode 2: wrists
)

# NB: encoder_mode_4 carries the mode ID as a SCALAR in its first slot, then zeros --
# it is NOT a one-hot (GatherEncoderMode(..., fill_zeros_num=3)).
ENCODER_MODES = {"g1": 0, "teleop": 1, "smpl": 2}

ACTION_DIM = 29
ENCODER_DIM = sum(d for _, d in ENCODER_LAYOUT)
DECODER_DIM = sum(d for _, d in DECODER_LAYOUT)
TOKEN_DIM = 64
HISTORY_FRAMES = 10

assert ENCODER_DIM == 1762, ENCODER_DIM
assert DECODER_DIM == 994, DECODER_DIM


def offsets(layout: tuple[tuple[str, int], ...]) -> dict[str, tuple[int, int]]:
    """name -> (start, end) slice bounds within the flat vector."""
    out, pos = {}, 0
    for name, dim in layout:
        out[name] = (pos, pos + dim)
        pos += dim
    return out


DECODER_OFFSETS = offsets(DECODER_LAYOUT)
ENCODER_OFFSETS = offsets(ENCODER_LAYOUT)


# --- robot-side constants (from gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/
#     policy_parameters.hpp, values computed from their motor model) ----------------------
#   kp = armature * (2*pi*10)^2 ; kd = 2 * 2 * armature * (2*pi*10)
#   action_scale = 0.25 * effort_limit / kp
#   JOINT TARGET: target = action * ACTION_SCALE + DEFAULT_ANGLES   (per joint)
#
# Joint order below is the POLICY order (the same limb-sequential order our existing SONIC
# port uses for its first 15 entries). Map to a live Isaac articulation BY NAME -- our G1
# carries 43 joints (29 body + 14 hand), so positional index maps from the vendor
# (isaaclab_to_mujoco / mujoco_to_isaaclab in policy_parameters.hpp) do NOT apply directly.
JOINT_ORDER: tuple[str, ...] = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

DEFAULT_ANGLES = [
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,        # left leg
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,        # right leg
    0.0, 0.0, 0.0,                                # waist
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,            # left arm
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,           # right arm
]

KP = [99.0984, 99.0984, 40.1792, 99.0984, 14.2506, 14.2506, 99.0984, 99.0984, 40.1792, 99.0984, 14.2506, 14.2506, 40.1792, 14.2506, 14.2506, 14.2506, 14.2506, 14.2506, 14.2506, 14.2506, 16.7783, 16.7783, 14.2506, 14.2506, 14.2506, 14.2506, 14.2506, 16.7783, 16.7783]
KD = [6.3088, 6.3088, 2.5579, 6.3088, 0.9072, 0.9072, 6.3088, 6.3088, 2.5579, 6.3088, 0.9072, 0.9072, 2.5579, 0.9072, 0.9072, 0.9072, 0.9072, 0.9072, 0.9072, 0.9072, 1.0681, 1.0681, 0.9072, 0.9072, 0.9072, 0.9072, 0.9072, 1.0681, 1.0681]
ACTION_SCALE = [0.35066, 0.35066, 0.54755, 0.35066, 0.43858, 0.43858, 0.35066, 0.35066, 0.54755, 0.35066, 0.43858, 0.43858, 0.54755, 0.43858, 0.43858, 0.43858, 0.43858, 0.43858, 0.43858, 0.43858, 0.07450, 0.07450, 0.43858, 0.43858, 0.43858, 0.43858, 0.43858, 0.07450, 0.07450]

assert len(JOINT_ORDER) == len(DEFAULT_ANGLES) == len(KP) == len(KD) == len(ACTION_SCALE) == ACTION_DIM


def policy_to_robot_index(robot_joint_names) -> list[int]:
    """Indices into a live Isaac articulation for each policy joint, matched BY NAME."""
    return [list(robot_joint_names).index(n) for n in JOINT_ORDER]


def verify_against_onnx(release_dir: str) -> None:
    """Assert the shipped models still match this contract (run after any vendor update)."""
    import os

    import onnxruntime as ort

    enc = ort.InferenceSession(os.path.join(release_dir, "model_encoder.onnx"),
                               providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(os.path.join(release_dir, "model_decoder.onnx"),
                               providers=["CPUExecutionProvider"])
    assert enc.get_inputs()[0].shape[-1] == ENCODER_DIM, enc.get_inputs()[0].shape
    assert enc.get_outputs()[0].shape[-1] == TOKEN_DIM, enc.get_outputs()[0].shape
    assert dec.get_inputs()[0].shape[-1] == DECODER_DIM, dec.get_inputs()[0].shape
    assert dec.get_outputs()[0].shape[-1] == ACTION_DIM, dec.get_outputs()[0].shape
    print(f"contract OK: encoder {ENCODER_DIM}->{TOKEN_DIM}, decoder {DECODER_DIM}->{ACTION_DIM}")


if __name__ == "__main__":
    import sys
    verify_against_onnx(sys.argv[1] if len(sys.argv) > 1 else
                        os.path.expanduser("~/robotica_project/GR00T-WholeBodyControl/"
                                           "gear_sonic_deploy/policy/release"))
