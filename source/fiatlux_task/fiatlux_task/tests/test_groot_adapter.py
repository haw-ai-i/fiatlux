# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Guardrails for the GR00T/GEAR-WBC adapter's data contract with NVIDIA's real
artifacts (Isaac-GR00T's REAL_G1 checkpoint, GR00T-WholeBodyControl's
decoupled_wbc). Each golden value here was read directly off the actual upstream
source this session, not reconstructed from memory:

- Body-joint order: ``decoupled_wbc/sim2mujoco/resources/robots/g1/g1_gear_wbc.xml``
  (MJCF ``<joint>`` tree order; ``get_joint_group_indices`` sorts by this canonical
  index, per its own docstring, not by ``joint_groups`` dict listing order).
- WBC gains/scales/dims: ``decoupled_wbc/sim2mujoco/resources/robots/g1/g1_gear_wbc.yaml``.
- Dex3 hand-channel order: ``decoupled_wbc/control/robot_model/supplemental_info/g1/g1_supplemental_info.py``.
- REAL_G1 state dims: the downloaded checkpoint's own
  ``processor_config.json:processor_kwargs.modality_configs["real_g1_relative_eef_relative_joints"]``.

Most of these are pure-Python contract checks and run anywhere. The two that reach into
``fiatlux_task.robots.g1`` / ``.sensors`` pull in ``isaaclab.sim``, which only resolves once the
Kit runtime has bootstrapped -- those two carry ``@pytest.mark.isaacsim_ci`` individually and
``importorskip`` out (as SKIPPED, not failed) when Kit is absent. The mark is per-test on
purpose: marking the whole module hid six tests that pass in a plain shell.
"""

import math

import pytest
import torch


def test_retarget_reduces_to_q_des_when_gains_match():
    """When our PD gains equal the decoder's own training gains, the general
    torque-retargeting derivation must collapse to using the decoder's position
    target directly -- the simplification GearWbcDecoder relies on."""
    from fiatlux_task.groot import retarget_torque_to_position

    torch.manual_seed(0)
    n, d = 4, 15
    q = torch.randn(n, d)
    dq = torch.randn(n, d)
    q_des = torch.randn(n, d)
    kp = torch.rand(n, d) * 200 + 10
    kd = torch.rand(n, d) * 5 + 0.1

    tau = kp * (q_des - q) - kd * dq
    q_t = retarget_torque_to_position(q, dq, tau, kp_ours=kp, kd_ours=kd)

    assert torch.allclose(q_t, q_des, atol=1e-5)


def test_retarget_reproduces_intended_torque_under_different_gains():
    """General case (our gains != the decoder's gains): the returned q_t must
    produce *exactly* the decoder's intended torque under our own gains -- that's
    the entire point of retargeting instead of using q_des directly."""
    from fiatlux_task.groot import retarget_torque_to_position

    torch.manual_seed(1)
    n, d = 4, 29
    q = torch.randn(n, d)
    dq = torch.randn(n, d)
    tau = torch.randn(n, d) * 5  # some other decoder's intended torque
    kp_ours = torch.rand(n, d) * 300 + 10
    kd_ours = torch.rand(n, d) * 8 + 0.1

    q_t = retarget_torque_to_position(q, dq, tau, kp_ours, kd_ours)
    tau_reproduced = kp_ours * (q_t - q) - kd_ours * dq

    assert torch.allclose(tau_reproduced, tau, atol=1e-4)


def test_gear_wbc_gains_match_reference_yaml():
    """GearWbcDecoder's transcription of the WBC's training-time gains -- the ones
    its torque intent is computed with before retargeting through the env's own PD
    gains -- must equal g1_gear_wbc.yaml's kps/kds exactly, not approximately.
    Golden values transcribed directly from that file."""
    from fiatlux_task.groot import _WBC_KD_LOWER, _WBC_KP_LOWER, _WBC_LOWER_DEFAULTS, WBC_HEIGHT_INIT

    golden_kp = [150.0, 150.0, 150.0, 200.0, 40.0, 40.0] * 2 + [250.0, 250.0, 250.0]
    golden_kd = [2.0, 2.0, 2.0, 4.0, 2.0, 2.0] * 2 + [5.0, 5.0, 5.0]
    golden_defaults = [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0] * 2 + [0.0, 0.0, 0.0]

    assert golden_kp == _WBC_KP_LOWER
    assert golden_kd == _WBC_KD_LOWER
    assert golden_defaults == _WBC_LOWER_DEFAULTS
    assert WBC_HEIGHT_INIT == 0.74


def test_wbc_obs_dim_matches_reference_yaml():
    from fiatlux_task.groot import _WBC_HIST, _WBC_OBS_DIM

    assert _WBC_HIST == 6
    assert _WBC_OBS_DIM * _WBC_HIST == 516  # g1_gear_wbc.yaml: num_obs


def test_body_joint_order_matches_gear_wbc_mjcf():
    """The 29-joint order GearWbcDecoder/SonicDecoder index into must match the
    real g1_gear_wbc.xml <joint> tree order (legs, waist, arms) -- this is what
    get_joint_group_indices("body") actually returns (sorted by this canonical
    index), not the joint_groups dict's [waist, legs, arms] listing order."""
    from fiatlux_task.groot import _BODY_JOINT_NAMES

    golden = [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    ]  # fmt: skip
    assert golden == _BODY_JOINT_NAMES


@pytest.mark.isaacsim_ci
def test_dex3_hand_joint_order_matches_supplemental_info():
    """Dex3 hand-channel order golden-sourced from g1_supplemental_info.py's
    joint_groups["left_hand"]/["right_hand"] (index_0/1, middle_0/1, thumb_0/1/2)."""
    pytest.importorskip("isaaclab.sim", reason="needs a bootstrapped Kit runtime")
    from fiatlux_task.robots.g1 import G1_DEX3_LEFT_HAND_JOINTS, G1_DEX3_RIGHT_HAND_JOINTS

    order = ["index_0", "index_1", "middle_0", "middle_1", "thumb_0", "thumb_1", "thumb_2"]
    assert [f"left_hand_{j}_joint" for j in order] == G1_DEX3_LEFT_HAND_JOINTS
    assert [f"right_hand_{j}_joint" for j in order] == G1_DEX3_RIGHT_HAND_JOINTS


def test_state_key_dims_match_checkpoint_processor_config():
    """Golden-sourced from the downloaded checkpoint's processor_config.json
    (processor_kwargs.modality_configs["real_g1_relative_eef_relative_joints"]["state"])."""
    from fiatlux_task.groot import _STATE_KEY_DIMS

    assert _STATE_KEY_DIMS == {
        "left_wrist_eef_9d": 9,
        "right_wrist_eef_9d": 9,
        "left_hand": 7,
        "right_hand": 7,
        "left_arm": 7,
        "right_arm": 7,
        "waist": 3,
    }


@pytest.mark.isaacsim_ci
def test_ego_camera_fov_matches_d435_spec():
    """Real Intel RealSense D435 RGB sensor: ~69.4 deg horizontal FOV. A narrower
    sim FOV understates how much of the scene is visible around an occluding hand."""
    pytest.importorskip("isaaclab.sim", reason="needs a bootstrapped Kit runtime")
    from fiatlux_task.sensors import ego_camera_cfg

    spawn = ego_camera_cfg().spawn
    fov_deg = 2 * math.degrees(math.atan(spawn.horizontal_aperture / (2 * spawn.focal_length)))
    assert math.isclose(fov_deg, 69.4, abs_tol=0.5)
