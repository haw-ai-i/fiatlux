"""Can the contacts the clip claims actually HOLD the robot? Offline static force balance.

For each sampled frame: the feet that are planted and the hands that are on a hold are the
contact set. A linear programme looks for contact forces that
  * balance gravity (force and moment about every axis),
  * push, never pull, and stay inside the friction cone at every contact,
  * and need no joint to exceed its torque limit (gravity torque of everything outboard of the
    joint plus the contact forces applied outboard of it).
It is solved twice -- feet only, and feet + hands -- minimising the worst joint-torque
utilisation. That answers the question the kinematic gate cannot: does a hand hold add support,
how much load can it take before the 5 N.m wrist gives out, and how much friction it needs.

Static means accelerations are ignored; the clip is slow where it matters (hands move only in
pauses) but this is still a lower bar than a controller actually tracking it.

    python scripts/climb/static_support_check.py --clip <dir> --actuators <fx_actuators.json>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import g1_arm_fk as K  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--clip", required=True)
ap.add_argument("--actuators", required=True, help="resolved fiatlux actuator dump (effort limits)")
ap.add_argument("--mu", type=float, nargs="+", default=[0.5, 1.0])
ap.add_argument("--mass", type=float, default=38.61, help="sim-measured total, kg; link masses are scaled to it")
ap.add_argument("--foot", default="pads", choices=["pads", "sole"],
                help="pads = the sim's foot collision (four 5 mm spheres at the corners, nothing "
                     "under the mid-sole); sole = a full flat sole like the real foot")
ap.add_argument("--stride", type=int, default=12)
ap.add_argument("--sole", type=float, nargs=5, default=[-0.05, 0.13, 0.03, 0.035, 0.025],
                help="sole x back, x front, half width, depth below ankle_roll, height tolerance")
a = ap.parse_args()
HERE = pathlib.Path(__file__).parent
clip = pathlib.Path(a.clip)
m = np.load(clip / "motion.npz")
meta = json.loads((clip / "meta.json").read_text())
bn, fps = list(meta["body_names"]), float(meta["fps"])
jn = meta["joint_names"] if isinstance(meta["joint_names"], list) else json.loads(meta["joint_names"].replace("'", '"'))
holds = json.loads((clip / "holds.json").read_text())
BP, BQ, JQ = m["body_pos_w"], m["body_quat_w"], m["joint_pos"]
T = BP.shape[0]
MASS = json.loads((HERE / "g1_dex3_mass.json").read_text())
sc = a.mass / sum(v["mass"] for v in MASS.values())
EFF = {k: v["effort_limit"] for k, v in json.loads(pathlib.Path(a.actuators).read_text()).items()}
G = 9.81
AX = {"X": np.array([1.0, 0, 0]), "Y": np.array([0, 1.0, 0]), "Z": np.array([0, 0, 1.0])}
JOINTS = [j for j in jn if j in K.CHAIN]
child = {j: K.CHAIN[j]["b1"] for j in JOINTS}
parent_of = {K.CHAIN[j]["b1"]: K.CHAIN[j]["b0"] for j in JOINTS}


def subtree(body):
    out = {body}
    grew = True
    while grew:
        grew = False
        for b, p in parent_of.items():
            if p in out and b not in out:
                out.add(b); grew = True
    return out


SUB = {j: subtree(child[j]) for j in JOINTS}
SURF = [("ground", -9.0, 9.0, 0.0)] + [(k, v["x"][0], v["x"][1], v["z"]) for k, v in holds["holds"].items()
                                       if k.startswith("step") or k == "platform"]
ANK = {s: bn.index(f"{s}_ankle_roll_link") for s in ("left", "right")}
vel = np.linalg.norm(np.gradient(BP[:, [ANK["left"], ANK["right"]]], 1 / fps, axis=0), axis=2)


SPH = [np.array([x, y, -0.030]) for x in (-0.05, 0.12) for y in (-0.03, 0.03)]   # the foot's collision: four 5 mm spheres
SURFXY = {"ground": (-9, 9, -9, 9)}
SURFXY.update({k: (v["x"][0], v["x"][1], v["y"][0], v["y"][1]) for k, v in holds["holds"].items()})


def foot_contacts(i, side, k):
    """The foot's collision spheres that sit on a surface (within the height tolerance), or []."""
    if vel[i, k] > 0.15:
        return [], None
    p, q = BP[i, ANK[side]], BQ[i, ANK[side]]
    tol = a.sole[4]
    pts, names = [], set()
    cand = SPH if a.foot == "pads" else [np.array([x, y, -0.030]) for x in np.arange(-0.05, 0.1201, 0.01) for y in (-0.03, 0.03)]
    for c in cand:
        w = p + K.qrot(q, c)
        for nm, _, _, z in SURF:
            x0, x1, y0, y1 = SURFXY[nm]
            if x0 - 0.004 <= w[0] <= x1 + 0.004 and y0 <= w[1] <= y1 and abs(w[2] - 0.005 - z) <= tol:
                pts.append(np.array([w[0], w[1], z])); names.add(nm)
    if not pts:
        return [], None
    if a.foot == "sole":
        xs = [float(x[0]) for x in pts]
        return pts, f"{'/'.join(sorted(names))} ({(max(xs)-min(xs))*1000:.0f} mm of sole)"
    return pts, f"{'/'.join(sorted(names))} ({len(pts)} of 4 pads)"


def hand_contact(i, side):
    step = i / fps * 50
    for s0, s1, nm, c in holds["schedule"][side]:
        if s0 <= step <= s1:
            wq = BQ[i, bn.index(f"{side}_wrist_yaw_link")]
            n = -K.qrot(wq, np.array([0.0, -1.0 if side == "left" else 1.0, 0.0]))   # out of the palm face, reversed
            return np.array(c), n / np.linalg.norm(n), nm
    return None, None, None


def solve(i, use_hands, mu):
    cons = []      # (point, normal, limb-end body)
    feet = []
    for k, s in enumerate(("left", "right")):
        pts, _ = foot_contacts(i, s, k)
        feet.append(len(pts) > 0)
        cons += [(p, np.array([0, 0, 1.0]), f"{s}_ankle_roll_link") for p in pts]
    nh = 0
    if use_hands:
        for s in ("left", "right"):
            c, n, nm = hand_contact(i, s)
            if c is not None:
                cons.append((c, n, f"{s}_wrist_yaw_link")); nh += 1
    if not cons:
        return None
    n = len(cons)
    ALL = K.fk_all(BP[i, bn.index("pelvis")], BQ[i, bn.index("pelvis")], JQ[i], jn)
    # link world COMs
    W = []
    for b, v in MASS.items():
        k = bn.index(b)
        W.append((b, v["mass"] * sc, BP[i, k] + K.qrot(BQ[i, k], np.array(v["com"]))))
    Mtot = sum(w[1] for w in W)
    com = sum(w[1] * w[2] for w in W) / Mtot
    nv = 3 * n + 1                                         # forces, then s = worst torque utilisation
    Aeq = np.zeros((6, nv)); beq = np.zeros(6)
    for c, (p, _, _) in enumerate(cons):
        Aeq[0:3, 3 * c:3 * c + 3] = np.eye(3)
        Aeq[3:6, 3 * c:3 * c + 3] = np.array([[0, -p[2], p[1]], [p[2], 0, -p[0]], [-p[1], p[0], 0]])
    beq[2] = Mtot * G
    beq[3:6] = np.cross(com, np.array([0, 0, Mtot * G]))
    Aub, bub = [], []
    for c, (p, nrm, _) in enumerate(cons):
        t1 = np.cross(nrm, [0, 1.0, 0]); t1 /= np.linalg.norm(t1); t2 = np.cross(nrm, t1)
        for t in (t1, -t1, t2, -t2):
            r = np.zeros(nv); r[3 * c:3 * c + 3] = t - mu / np.sqrt(2) * nrm
            Aub.append(r); bub.append(0.0)
    tq = []
    for j in JOINTS:
        if j not in EFF:
            continue
        o, cq = ALL[child[j]]
        ax = K.qrot(cq, AX[K.CHAIN[j]["axis"]])
        g0 = sum(ax @ np.cross(w[2] - o, np.array([0, 0, -w[1] * G])) for w in W if w[0] in SUB[j])
        r = np.zeros(nv)
        for c, (p, _, body) in enumerate(cons):
            if body in SUB[j]:
                r[3 * c:3 * c + 3] = [ax @ np.cross(p - o, e) for e in np.eye(3)]
        tq.append((j, r.copy(), g0))
        for sg in (1.0, -1.0):
            rr = sg * r.copy(); rr[-1] = -EFF[j]
            Aub.append(rr); bub.append(-sg * g0)
    cost = np.zeros(nv); cost[-1] = 1.0
    res = linprog(cost, A_ub=np.array(Aub), b_ub=np.array(bub), A_eq=Aeq, b_eq=beq,
                  bounds=[(None, None)] * (3 * n) + [(0, None)], method="highs")
    if not res.success:
        fx = [p[0] for p, _, b in cons if "ankle" in b]
        off = 0.0 if not fx else (com[0] - min(fx) if com[0] < min(fx) else (com[0] - max(fx) if com[0] > max(fx) else 0.0))
        return dict(ok=False, off=off, feet=feet, nh=nh)
    f = res.x[:3 * n].reshape(n, 3)
    util = [(abs(r[:-1] @ res.x[:-1] + g0) / EFF[j], j) for j, r, g0 in tq]
    hand_up = sum(f[c][2] for c, (_, _, b) in enumerate(cons) if "wrist" in b) / (Mtot * G)
    hand_f = [float(np.linalg.norm(f[c])) for c, (_, _, b) in enumerate(cons) if "wrist" in b]
    fx = [p[0] for p, _, b in cons if "ankle" in b]
    off = 0.0 if not fx else (com[0] - min(fx) if com[0] < min(fx) else (com[0] - max(fx) if com[0] > max(fx) else 0.0))
    return dict(ok=True, off=off, s=float(res.x[-1]), lim=max(util)[1], feet=feet, nh=nh, hand_up=float(hand_up), hand_f=hand_f)


def state(i):
    fs = []
    for k, s in enumerate(("left", "right")):
        _, d = foot_contacts(i, s, k)
        fs.append(d or "in the air")
    hs = [hand_contact(i, s)[2] or "moving" for s in ("left", "right")]
    hs = ["handrail edge" if h == "hook" else h for h in hs]
    return fs[0], fs[1], hs[0], hs[1]


runs, cur = [], None
for i in range(0, T, a.stride):
    st = state(i)
    key = tuple(x.split(" (")[0] for x in st)
    if cur is None or cur[2] != key:
        cur = [st, [], key]; runs.append(cur)
    cur[1].append(i)
print(f"[model] {a.mass:.1f} kg; joint torque limits from {pathlib.Path(a.actuators).name}; wrist pitch/yaw {EFF.get('left_wrist_pitch_joint')} N.m")
print("[legend] effort = worst joint torque as % of its limit at the best force distribution; "
      "'cannot balance' = no forces satisfy balance + friction (+ torque limits if >100%)")
for mu in a.mu:
    print(f"\n==== friction {mu} ====")
    print(f"{'steps':>11s}  {'L foot':22s} {'R foot':22s} {'L hand':13s} {'R hand':13s} | {'feet only':52s} | feet + hands")
    for st, idx, _ in runs:
        if len(idx) < 2:
            continue
        cell = []
        for use in (False, True):
            R = [solve(i, use, mu) for i in idx]
            R = [r for r in R if r is not None]
            bad = [r for r in R if not r["ok"]]
            if not R:
                cell.append("no contact at all"); continue
            if bad:
                o = max((r["off"] for r in bad), key=abs) * 1000
                why = f", body centre {abs(o):.0f} mm {'behind' if o < 0 else 'beyond'} the feet" if abs(o) > 1 else ""
                cell.append(f"cannot balance {len(bad)}/{len(R)}{why}"); continue
            w = max(R, key=lambda r: r["s"])
            txt = f"effort {w['s']*100:4.0f}% ({w['lim'].replace('_joint','')})"
            if use and any(r["nh"] for r in R):
                up = [r["hand_up"] * 100 for r in R]
                txt += f"; hands carry {min(up):.0f}..{max(up):.0f}% of weight, up to {max(max(r['hand_f'] or [0]) for r in R):.0f} N per hand"
            cell.append(txt)
        print(f"{idx[0]/fps*50:5.0f}-{idx[-1]/fps*50:<5.0f}  {st[0]:22s} {st[1]:22s} {st[2]:13s} {st[3]:13s} | {cell[0]:52s} | {cell[1]}")
