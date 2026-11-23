"""Play one stretch of a clip in slow motion: every joint and body together, nothing re-posed.

For the bulb arm's reach at the mount (S10). The clip takes the right wrist from the carry pose to
its rest on tread 4's near edge in about 0.6 s, peaking at 0.83 m/s; a teacher tracking it loosely
arrives at 1.8 m/s and the landing ends the episode (88 N on the hand in one step). Slowing the ARM
alone was tried and rejected: the reach is coordinated with the torso leaning in, and an arm that
runs ahead of the torso put the bulb 23 mm inside the ladder.

So this stretches TIME over `--steps A B` by `--factor`: the same poses in the same order, the
whole body, so every clearance the clip was checked for still holds; velocities inside the window
are divided by the factor. The clip gets longer, and every later step number moves by the amount
printed. The rate changes linearly over `--ease` steps at each end of the window rather than
jumping.

    python dilate_window.py --clip <in> --out <out> --steps 265 300 --factor 3
"""

import argparse
import json
import pathlib
import shutil

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--clip", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, nargs=2, required=True, help="50 Hz steps A B to slow down")
ap.add_argument("--factor", type=float, default=3.0)
ap.add_argument(
    "--ease", type=int, default=4, help="50 Hz steps (of the original) over which the rate changes at each end"
)
a = ap.parse_args()
src, dst = pathlib.Path(a.clip), pathlib.Path(a.out)
d = dict(np.load(src / "motion.npz"))
meta = json.loads((src / "meta.json").read_text())
fps = float(meta["fps"])
per = fps / 50.0
nf = d["joint_pos"].shape[0]
ns = d["object_contact"].shape[0]
A, B = a.steps
fA, fB, fe = A * per, B * per, a.ease * per
# slowness s(tau) = d(new time)/d(old time): 1 outside, `factor` inside, linear over the ease at each end
tau = np.arange(0, nf - 1 + 1e-9, 0.02)
s = 1.0 + (a.factor - 1.0) * np.clip(np.minimum((tau - fA) / fe, (fB - tau) / fe), 0.0, 1.0)
cum = lambda sl: np.concatenate([[0.0], np.cumsum(0.5 * (sl[1:] + sl[:-1]) * 0.02)])
unit = int(fps / np.gcd(int(fps), 50))  # frames in the smallest whole number of 50 Hz steps
extra = cum(s)[-1] - tau[-1]
s = 1.0 + (s - 1.0) * (
    unit * max(1, round(extra / unit)) / extra
)  # the added time is a whole number of steps AND frames,
t_new = cum(s)  # so everything after the window lands on its own frames
n_new = int(round(t_new[-1])) + 1
t_new[tau >= fB] = tau[tau >= fB] + (n_new - nf)
src_t = np.interp(np.arange(n_new), t_new, tau)  # old frame (fractional) for each new frame
rate = np.gradient(src_t)  # old frames per new frame
i0 = np.clip(np.floor(src_t).astype(int), 0, nf - 2)
w = src_t - i0


def lerp(v):
    ww = w.reshape((-1,) + (1,) * (v.ndim - 1))
    return v[i0] * (1 - ww) + v[i0 + 1] * ww


out = {}
for k, v in d.items():
    if k == "object_contact":
        ns_new = int(round((n_new - 1) / per)) + (ns - int(round((nf - 1) / per)))
        st = np.clip(
            np.round(src_t[np.clip(np.round(np.arange(ns_new) * per).astype(int), 0, n_new - 1)] / per).astype(int),
            0,
            ns - 1,
        )
        out[k] = v[st]
    elif v.shape[:1] != (nf,):
        out[k] = v
    elif k.endswith("quat_w"):
        q0, q1 = v[i0], v[i0 + 1]
        q1 = np.where((q0 * q1).sum(-1, keepdims=True) < 0, -q1, q1)
        ww = w.reshape((-1,) + (1,) * (v.ndim - 1))
        q = q0 * (1 - ww) + q1 * ww
        out[k] = q / np.linalg.norm(q, axis=-1, keepdims=True)
    elif "vel" in k:
        out[k] = lerp(v) * rate.reshape((-1,) + (1,) * (v.ndim - 1))
    else:
        out[k] = lerp(v)
# positions: cubic through the original frames (a linear blend makes the velocity a staircase)
from scipy.interpolate import CubicSpline

for k in ("joint_pos", "body_pos_w"):
    out[k] = CubicSpline(np.arange(nf), d[k], axis=0)(src_t)
# velocities are the new clip's own derivatives, the convention the clips are stored in
out["joint_vel"] = np.gradient(out["joint_pos"], 1.0 / fps, axis=0)
out["body_lin_vel_w"] = np.gradient(out["body_pos_w"], 1.0 / fps, axis=0)
_q = out["body_quat_w"]
_n = np.roll(_q, -1, axis=0)
_n[-1] = _q[-1]
_c = np.concatenate([_q[..., :1], -_q[..., 1:]], -1)
_w0, _v0 = _n[..., :1], _n[..., 1:]
_w1, _v1 = _c[..., :1], _c[..., 1:]
_dq = np.concatenate([_w0 * _w1 - (_v0 * _v1).sum(-1, keepdims=True), _w0 * _v1 + _w1 * _v0 + np.cross(_v0, _v1)], -1)
_dq = _dq * np.sign(_dq[..., :1] + 1e-12)
out["body_ang_vel_w"] = 2.0 * _dq[..., 1:] * fps
out["body_ang_vel_w"][-1] = out["body_ang_vel_w"][-2]
out = {k: (v.astype(d[k].dtype) if hasattr(v, "astype") else v) for k, v in out.items()}
dst.mkdir(parents=True, exist_ok=True)
for f in src.iterdir():
    if f.name != "motion.npz":
        shutil.copy(f, dst / f.name)
np.savez(dst / "motion.npz", **out)
added = (n_new - nf) / per
fd = np.gradient(out["joint_pos"], 1.0 / fps, axis=0)
err = float(np.abs(fd[2:-2] - out["joint_vel"][2:-2]).max())
bn = meta["body_names"]
wi = bn.index("right_wrist_yaw_link")
n0, n1 = int(np.searchsorted(src_t, fA)), int(np.searchsorted(src_t, fB))
sp = lambda v, x0, x1: float(np.linalg.norm(v[x0 : x1 + 1, wi], axis=1).max())
print(
    f"[dilate] steps {A}-{B} slowed {a.factor:g}x: {nf} -> {n_new} frames, {ns} -> {out['object_contact'].shape[0]} steps "
    f"(every step after {B} is now +{added:.0f}); the window is now steps {n0 / per:.0f}-{n1 / per:.0f}"
)
print(
    f"[speed ] right wrist peak in the window {sp(out['body_lin_vel_w'], n0, n1):.2f} m/s (was {sp(d['body_lin_vel_w'], int(fA), int(fB)):.2f}); "
    f"joint peak {np.abs(out['joint_vel'][n0 : n1 + 1]).max():.2f} rad/s (was {np.abs(d['joint_vel'][int(fA) : int(fB) + 1]).max():.2f})"
)
print(
    f"[check ] stored joint velocity against the new clip's own finite difference: worst {err:.3f} rad/s; "
    f"first and last pose unchanged: {np.allclose(out['joint_pos'][[0, -1]], d['joint_pos'][[0, -1]], atol=1e-6)}"
)
print(f"[out   ] {dst}")
