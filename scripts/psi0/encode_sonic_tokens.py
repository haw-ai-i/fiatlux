# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Turn fiatlux teleop bags into Psi-0 SONIC training arrays (stage A of the fine-tune pack).

The teleop twins drive the arms by IK and the legs by the decoupled walk/balance WBC, so a
take records joint trajectories but no GEAR-SONIC motion token -- and Psi-0's SONIC action is
``[token 64 | Dex3 14 | neck 2]``. This encodes each take's own *executed* whole-body motion as
the reference motion of the released SONIC encoder (``nvidia/GEAR-SONIC:model_encoder.onnx``) in
its ``g1`` mode, exactly the input GEAR-SONIC's deploy stack builds from a joint-space
reference clip (``gear_sonic_deploy/.../g1_deploy_onnx_ref.cpp``, ``GatherEncoderObservations``
with the ``observation_config.yaml`` encoder list):

- ``encoder_mode_4``: the mode id (``g1`` = 0) followed by three zeros;
- ``motion_joint_positions_10frame_step5`` / ``..._velocities_...``: the reference's 29 body
  joints (IsaacLab order, absolute radians -- ``command_multi_future`` in gear_sonic's
  ``TrackingCommand``) at frames ``t, t+5, ..., t+45`` of the 50 Hz clip, clamped to its last
  frame;
- ``motion_anchor_orientation_10frame_step5``: ``conj(q_base(t)) * q_ref_root(t+5k)`` as the
  first two rotation-matrix columns flattened row-major;
- every observation the ``g1`` mode does not require stays zero (``GatherEncoderObservations``:
  "Not required for this mode - leave as zero").

Here the robot base and the reference root are the same recorded pelvis trajectory, so the
deploy stack's ``apply_delta_heading`` is the identity. Velocities are central differences of
the recorded positions (the deploy reader derives reference velocities from positions too).

Output per take (``<out>/<take>.npz``, all at 50 Hz, one row per recorded step):
``token`` (T, 64, FSQ-snapped like Psi-0's own client), ``hand_target`` (T, 14, the commanded
Dex3 targets in ``psi0.PSI0_HAND_JOINT_NAMES`` order), ``state`` (T, 45, ``psi0``'s state
layout), ``step`` (T,). Stage B (``build_psi0_pack.py``, Psi0 venv) resamples to 30 Hz and
writes the LeRobot pack.

Run in the fiatlux venv (needs onnxruntime + h5py; no simulator):
    python scripts/psi0/encode_sonic_tokens.py --takes ~/psi0_ft/raw/success --out ~/psi0_ft/encoded
    python scripts/psi0/encode_sonic_tokens.py --check-stand     # encoder-layout sanity check
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source" / "fiatlux_task"))

from fiatlux_task.groot import SONIC_JOINT_NAMES, STAND_TOKEN, _isaac_order  # noqa: E402
from fiatlux_task.psi0 import PSI0_HAND_JOINT_NAMES, PSI0_STATE_JOINT_NAMES, fsq_quantize  # noqa: E402

DEFAULT_ENCODER = os.path.expanduser("~/tools/sonic_models/policy/release/model_encoder.onnx")
ENCODER_DIM = 1762
_N_FRAMES, _STEP = 10, 5
# Offsets of the blocks the g1 mode fills, from observation_config.yaml's encoder list order:
# encoder_mode_4 (4), motion_joint_positions_10frame_step5 (290),
# motion_joint_velocities_10frame_step5 (290), motion_root_z_position_10frame_step5 (10),
# motion_root_z_position (1), motion_anchor_orientation (6), motion_anchor_orientation_10frame_step5
# (60), then lower-body / vr / smpl / wrist blocks (zero in g1 mode) up to 1762.
_OFF_MODE, _OFF_JPOS, _OFF_JVEL, _OFF_ANCHOR = 0, 4, 294, 601
G1_MODE_ID = 0


def quat_conj(q: np.ndarray) -> np.ndarray:
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, wxyz, broadcasting over leading dims."""
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def rot6d_rowmajor(q: np.ndarray) -> np.ndarray:
    """First two rotation-matrix columns of wxyz ``q``, flattened row by row (R00 R01 R10 R11 R20 R21)."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    r = np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w)], -1),
        ],
        axis=-2,
    )  # (..., 3 rows, 2 cols)
    return r.reshape(*q.shape[:-1], 6)


def encoder_inputs(jpos: np.ndarray, jvel: np.ndarray, root_quat: np.ndarray) -> np.ndarray:
    """(T, 1762) g1-mode encoder observations for every step of a 50 Hz reference clip.

    ``jpos``/``jvel``: (T, 29) in SONIC's IsaacLab order; ``root_quat``: (T, 4) wxyz pelvis.
    """
    t_len = jpos.shape[0]
    obs = np.zeros((t_len, ENCODER_DIM), dtype=np.float32)
    obs[:, _OFF_MODE] = G1_MODE_ID
    offsets = np.arange(_N_FRAMES) * _STEP
    frames = np.minimum(np.arange(t_len)[:, None] + offsets[None, :], t_len - 1)  # (T, 10)
    obs[:, _OFF_JPOS : _OFF_JPOS + 290] = jpos[frames].reshape(t_len, -1)
    obs[:, _OFF_JVEL : _OFF_JVEL + 290] = jvel[frames].reshape(t_len, -1)
    rel = quat_mul(quat_conj(root_quat)[:, None, :], root_quat[frames])  # (T, 10, 4)
    obs[:, _OFF_ANCHOR : _OFF_ANCHOR + 60] = rot6d_rowmajor(rel).reshape(t_len, -1)
    return obs


class SonicEncoder:
    def __init__(self, path: str = DEFAULT_ENCODER):
        import onnxruntime as ort

        self._session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        meta = self._session.get_inputs()[0]
        assert meta.shape[-1] == ENCODER_DIM, f"encoder expects {meta.shape}, layout builds {ENCODER_DIM}"
        self._name = meta.name

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return np.concatenate([self._session.run(None, {self._name: row[None]})[0] for row in obs]).astype(np.float32)


def check_stand(encoder: SonicEncoder) -> None:
    """Encode a clip that stands still in SONIC's default pose; it should land on STAND_TOKEN."""
    default = _isaac_order(0).numpy().astype(np.float64)
    t_len = 60
    jpos = np.tile(default, (t_len, 1))
    jvel = np.zeros_like(jpos)
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (t_len, 1))
    token = encoder(encoder_inputs(jpos, jvel, quat))[0]
    snapped = fsq_quantize(token)
    print("raw token (first 8):", np.round(token[:8], 4))
    print("on FSQ grid already:", bool(np.allclose(token, snapped, atol=1e-4)))
    diff = np.abs(snapped - STAND_TOKEN)
    print(f"vs STAND_TOKEN: {int((diff < 1e-6).sum())}/64 dims equal, max |diff| {diff.max():.4f}")


def load_take(run_h5: Path) -> dict[str, np.ndarray]:
    with h5py.File(run_h5, "r") as f:
        meta = json.loads(f.attrs["meta"])
        demos = sorted(f["data"].keys())
        assert len(demos) == 1, f"{run_h5}: expected one demo per take, got {demos}"
        d = f["data"][demos[0]]
        out = {k: d[k][()] for k in ("joint_pos", "joint_pos_target", "robot_root_quat", "step_in_episode")}
    out["joint_names"] = np.array(meta["joint_names"])
    out["step_dt"] = np.float64(meta["step_dt"])
    return out


def convert_take(encoder: SonicEncoder, run_h5: Path) -> dict[str, np.ndarray]:
    take = load_take(run_h5)
    names = list(take["joint_names"])
    col = {n: i for i, n in enumerate(names)}
    body = [col[n] for n in SONIC_JOINT_NAMES]
    jpos = take["joint_pos"][:, body].astype(np.float64)
    jvel = np.gradient(jpos, float(take["step_dt"]), axis=0)
    token = encoder(encoder_inputs(jpos, jvel, take["robot_root_quat"].astype(np.float64)))
    state = take["joint_pos"][:, [col[n] for n in PSI0_STATE_JOINT_NAMES]]
    state = np.concatenate([state, np.zeros((state.shape[0], 2), np.float32)], axis=1).astype(np.float32)
    hand_target = take["joint_pos_target"][:, [col[n] for n in PSI0_HAND_JOINT_NAMES]].astype(np.float32)
    return {
        "token": fsq_quantize(token),
        "token_raw": token,
        "hand_target": hand_target,
        "state": state,
        "step": take["step_in_episode"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--takes", type=Path, help="directory holding <take>/run.h5 folders")
    ap.add_argument("--out", type=Path, help="output directory for <take>.npz")
    ap.add_argument("--encoder", default=DEFAULT_ENCODER)
    ap.add_argument("--check-stand", action="store_true", help="only run the standing-pose sanity check")
    args = ap.parse_args()

    encoder = SonicEncoder(args.encoder)
    check_stand(encoder)
    if args.check_stand:
        return
    args.out.mkdir(parents=True, exist_ok=True)
    for run_h5 in sorted(args.takes.glob("*/run.h5")):
        arrays = convert_take(encoder, run_h5)
        np.savez(args.out / f"{run_h5.parent.name}.npz", **arrays)
        raw = arrays["token_raw"]
        print(
            f"{run_h5.parent.name}: {len(raw)} steps, |token| mean {np.abs(raw).mean():.3f}, "
            f"off-grid mean {np.abs(raw - arrays['token']).mean():.4f}"
        )


if __name__ == "__main__":
    main()
