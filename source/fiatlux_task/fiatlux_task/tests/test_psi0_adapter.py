# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Guardrails for the Psi-0 adapter's data contract with the released SONIC checkpoint
(``USC-PSI-Lab/psi-model:psi0/sonic-checkpoints/multi-task.psi-dream.2609092156``). The golden
values come from that checkpoint's own ``run_config.json`` / ``argv.txt`` and from the Psi0 repo
at ``4f3720d``:

- 45-D state order: ``scripts/data/backfill_joint_names.py`` (the legacy ``g1_sonic_lerobot_0810``
  order is ``leg waist arm hand neck``; hands thumb -> middle -> index, per
  ``scripts/data/merge_posttrain_sonic.py``). The checkpoint's ``state_min``/``state_max`` agree:
  slots 3 and 9 are the only strictly-positive leg joints (knees), 32-35 are <= 0 and 39-42 are
  >= 0 (the mirrored left/right index+middle curl directions).
- Wire format: ``src/psi/deploy/helpers.py`` (``numpy_serialize`` / ``numpy_deserialize``).
- FSQ grid: ``src/psi/deploy/mock_psi0_client_rtc.py`` (``FSQ_MIN, FSQ_MAX, FSQ_STEP``).

All pure Python; none of these import Isaac Lab.
"""

import numpy as np


def test_state_layout_is_leg_waist_arm_hand_neck():
    from fiatlux_task.psi0 import PSI0_STATE_DIM, PSI0_STATE_JOINT_NAMES

    names = PSI0_STATE_JOINT_NAMES
    assert len(names) + 2 == PSI0_STATE_DIM  # + neck (yaw, pitch), zero on this neckless G1
    assert names[:6] == [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
    ]
    assert names[3] == "left_knee_joint" and names[9] == "right_knee_joint"
    assert names[12:15] == ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
    assert names[15] == "left_shoulder_pitch_joint" and names[22] == "right_shoulder_pitch_joint"
    assert names[29:36] == [
        "left_hand_thumb_0_joint",
        "left_hand_thumb_1_joint",
        "left_hand_thumb_2_joint",
        "left_hand_middle_0_joint",
        "left_hand_middle_1_joint",
        "left_hand_index_0_joint",
        "left_hand_index_1_joint",
    ]
    assert names[36:43] == [n.replace("left_", "right_") for n in names[29:36]]


def test_action_hand_slots_follow_state_hand_order():
    """The 80-D action is token(64) ++ the pack's action[:14] ++ neck(2); the 14 hand joints
    share the state's order, which the adapter's hand-joint lookup relies on."""
    from fiatlux_task.psi0 import PSI0_ACTION_DIM, PSI0_HAND_JOINT_NAMES, PSI0_STATE_JOINT_NAMES

    assert PSI0_ACTION_DIM == 64 + 14 + 2
    assert PSI0_STATE_JOINT_NAMES[29:43] == PSI0_HAND_JOINT_NAMES


def test_chunk_rows_hold_30hz_rows_over_50hz_ticks():
    from fiatlux_task.psi0 import chunk_row, ticks_per_rows

    assert [chunk_row(n) for n in range(6)] == [0, 0, 1, 1, 2, 3]
    assert ticks_per_rows(15) == 25 and ticks_per_rows(30) == 50
    # The last tick before a re-query still reads a valid row, and the next would not.
    for rows in (1, 7, 15, 16, 30):
        n = ticks_per_rows(rows)
        assert chunk_row(n - 1) == rows - 1
        assert chunk_row(n) >= rows


def test_wire_blob_round_trip_matches_psi0_helpers_format():
    from fiatlux_task.psi0 import _np_blob, _np_unblob

    image = (np.arange(270 * 480 * 3) % 251).astype(np.uint8).reshape(270, 480, 3)
    state = np.linspace(-1, 1, 45, dtype=np.float32)
    for array in (image, state):
        blob = _np_blob(array)
        assert set(blob) == {"__numpy__", "dtype", "shape"}
        assert blob["dtype"] == np.lib.format.dtype_to_descr(array.dtype)
        np.testing.assert_array_equal(_np_unblob(blob), array)


def test_fsq_quantize_snaps_onto_the_sonic_grid():
    from fiatlux_task.groot import STAND_TOKEN
    from fiatlux_task.psi0 import fsq_quantize

    x = np.array([-2.0, -0.63, -0.04, 0.03, 0.0313, 0.40, 0.9], dtype=np.float32)
    np.testing.assert_allclose(fsq_quantize(x), [-0.625, -0.625, -0.0625, 0.0, 0.0625, 0.375, 0.625])
    # SONIC's own standing latent already lies on the grid.
    np.testing.assert_array_equal(fsq_quantize(STAND_TOKEN), STAND_TOKEN)


def test_every_subtask_has_an_instruction():
    from fiatlux_task.psi0 import TASK_INSTRUCTIONS, instruction_for_task

    subtasks = [k for k in TASK_INSTRUCTIONS if k.startswith("FIATLUX-S")]
    assert len(subtasks) == 12
    dispose = TASK_INSTRUCTIONS["FIATLUX-S06-DisposeBulb-v0"]
    assert instruction_for_task("FIATLUX-S06-DisposeBulb-Training-v0") == dispose
