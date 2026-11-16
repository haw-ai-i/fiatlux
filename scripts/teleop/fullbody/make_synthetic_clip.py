"""Synthesise the smallest reference that answers 'can SONIC track a reference at all?'

Two motions on flat ground, both reachable by a standing G1:
  1. a slow squat and return  -- symmetric, no weight shift, the easy case
  2. a single high knee lift  -- the climbing primitive: one foot leaves the ground

Written in the GEAR-SONIC clip format (dof (T,29) in JOINT_ORDER, root_rot wxyz,
root_trans_offset, fps) so `gear_sonic_offline_check.py` and `gear_sonic_policy.py`
consume it unchanged.
"""

from __future__ import annotations

import sys

import numpy as np
from gear_sonic_contract import DEFAULT_ANGLES, JOINT_ORDER

FPS = 50
J = {n: i for i, n in enumerate(JOINT_ORDER)}


def _base(T: int) -> np.ndarray:
    return np.tile(np.asarray(DEFAULT_ANGLES, dtype=np.float32), (T, 1))


def squat(seconds: float = 4.0, depth: float = 0.45) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric squat: hips+knees+ankles bend together, both feet stay planted."""
    T = int(seconds * FPS)
    s = np.sin(np.linspace(0.0, np.pi, T)).astype(np.float32) * depth   # 0 -> depth -> 0
    dof = _base(T)
    for side in ("left", "right"):
        dof[:, J[f"{side}_hip_pitch_joint"]] -= s
        dof[:, J[f"{side}_knee_joint"]] += 2.0 * s
        dof[:, J[f"{side}_ankle_pitch_joint"]] -= s
    # the pelvis drops roughly knee-length * (1-cos); a crude but monotone proxy
    dz = -(0.30 * (1.0 - np.cos(s))).astype(np.float32)
    return dof, dz


def knee_lift(seconds: float = 3.0, hip: float = 1.20, knee: float = 1.60) -> tuple[np.ndarray, np.ndarray]:
    """One high knee lift on the RIGHT leg -- the single-tread climbing primitive."""
    T = int(seconds * FPS)
    s = np.sin(np.linspace(0.0, np.pi, T)).astype(np.float32)
    dof = _base(T)
    dof[:, J["right_hip_pitch_joint"]] -= hip * s         # thigh forward/up
    dof[:, J["right_knee_joint"]] += knee * s             # heel toward the buttock
    dof[:, J["right_ankle_pitch_joint"]] -= 0.3 * s
    dof[:, J["left_knee_joint"]] += 0.15 * s              # stance leg takes the load
    dof[:, J["left_hip_pitch_joint"]] -= 0.10 * s
    return dof, np.zeros(T, dtype=np.float32)


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "synthetic_squat_kneelift.npz"
    hold = _base(FPS)                                      # 1 s standing, so histories fill
    d0, z0 = squat()
    d1, z1 = knee_lift()
    dof = np.concatenate([hold, d0, hold, d1, hold], axis=0).astype(np.float32)
    dz = np.concatenate([np.zeros(FPS, np.float32), z0, np.zeros(FPS, np.float32), z1,
                         np.zeros(FPS, np.float32)]).astype(np.float32)
    T = len(dof)
    root_rot = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (T, 1))          # upright, wxyz
    root_trans = np.zeros((T, 3), dtype=np.float32)
    root_trans[:, 2] = 0.793 + dz                                        # nominal stand + squat drop
    np.savez(out, dof=dof, root_rot=root_rot, root_trans_offset=root_trans, fps=FPS)
    print(f"wrote {out}: {T} frames @ {FPS} Hz = {T/FPS:.1f} s")
    print(f"  phases: 1s stand | 4s squat | 1s stand | 3s right knee lift | 1s stand")
    print(f"  right knee peak {dof[:, J['right_knee_joint']].max():.2f} rad, "
          f"hip pitch min {dof[:, J['right_hip_pitch_joint']].min():.2f} rad")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
