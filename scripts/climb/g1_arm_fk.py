"""Offline forward kinematics for the G1 waist + arms, read from the Dex3 USD's own joints.

Exists so a reference clip can be authored and CHECKED in seconds, without a simulator pass:
`g1_dex3_chain.json` holds each joint's parent/child frames, axis and limits exactly as the USD
authors them (extracted with pxr). Validated against body positions Isaac produced -- run this
file directly to repeat that check.
"""
import json
import pathlib

import numpy as np

CHAIN = json.loads((pathlib.Path(__file__).parent / "g1_dex3_chain.json").read_text())
WAIST = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
ARM = ["{s}_shoulder_pitch_joint", "{s}_shoulder_roll_joint", "{s}_shoulder_yaw_joint",
       "{s}_elbow_joint", "{s}_wrist_roll_joint", "{s}_wrist_pitch_joint", "{s}_wrist_yaw_joint"]
SOLVE = ARM[:4]                                    # the joints a wrist POSITION depends on
_AX = {"X": np.array([1.0, 0, 0]), "Y": np.array([0, 1.0, 0]), "Z": np.array([0, 0, 1.0])}


def limits(name):
    j = CHAIN[name]
    return np.radians(j["lo"]), np.radians(j["hi"])


def qmul(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw], -1)


def qrot(q, v):
    w, u = q[..., :1], q[..., 1:]
    return 2*(u*v).sum(-1, keepdims=True)*u + (w*w-(u*u).sum(-1, keepdims=True))*v + 2*w*np.cross(u, v)


def qconj(q):
    return q * np.array([1.0, -1, -1, -1])


def _step(p, q, name, ang):
    """World pose of the joint's child body, from its parent's world pose."""
    j = CHAIN[name]
    half = np.asarray(ang)[..., None] * 0.5
    qj = np.concatenate([np.cos(half), np.sin(half) * _AX[j["axis"]]], -1)
    qc = qmul(qmul(qmul(q, np.asarray(j["r0"])), qj), qconj(np.asarray(j["r1"])))
    pc = p + qrot(q, np.asarray(j["p0"])) - qrot(qc, np.asarray(j["p1"]))
    return pc, qc


def fk(pel_p, pel_q, waist, arm, side):
    """-> dict of world positions (and torso pose). Inputs broadcast over leading dims.

    waist: [...,3] yaw/roll/pitch.  arm: [...,7] in ARM order.
    """
    p, q = np.asarray(pel_p, float), np.asarray(pel_q, float)
    for k, n in enumerate(WAIST):
        p, q = _step(p, q, n, np.asarray(waist)[..., k])
    out = {"torso_p": p, "torso_q": q}
    for k, t in enumerate(ARM):
        n = t.format(s=side)
        p, q = _step(p, q, n, np.asarray(arm)[..., k])
        out[CHAIN[n]["b1"]] = p
        out[CHAIN[n]["b1"] + "_q"] = q
    return out


if __name__ == "__main__":
    import sys
    clip = pathlib.Path(sys.argv[1])
    d = np.load(clip / "motion.npz")
    m = json.loads((clip / "meta.json").read_text())
    jn, bn = m["joint_names"], m["body_names"]
    pel = bn.index("pelvis")
    wq = d["joint_pos"][:, [jn.index(n) for n in WAIST]]
    print(f"validating against {clip.name}: {d['joint_pos'].shape[0]} frames of Isaac FK")
    for s in ("left", "right"):
        aq = d["joint_pos"][:, [jn.index(t.format(s=s)) for t in ARM]]
        o = fk(d["body_pos_w"][:, pel], d["body_quat_w"][:, pel], wq, aq, s)
        for b in (f"{s}_shoulder_pitch_link", f"{s}_elbow_link", f"{s}_wrist_yaw_link"):
            e = np.linalg.norm(o[b] - d["body_pos_w"][:, bn.index(b)], axis=1) * 1000
            print(f"  {b:28s} error mean {e.mean():.3f} mm  max {e.max():.3f} mm")


def fk_all(pel_p, pel_q, joint_pos, joint_names):
    """Every body under the pelvis -> {body: (pos, quat)}, walking the USD's joint tree."""
    out = {"pelvis": (np.asarray(pel_p, float), np.asarray(pel_q, float))}
    todo = [n for n in joint_names if n in CHAIN]
    while todo:
        rest = []
        for n in todo:
            j = CHAIN[n]
            if j["b0"] in out:
                p, q = out[j["b0"]]
                out[j["b1"]] = _step(p, q, n, np.asarray(joint_pos)[..., joint_names.index(n)])
            else:
                rest.append(n)
        if len(rest) == len(todo):
            raise RuntimeError(f"joints with no parent in the tree: {rest}")
        todo = rest
    return out
