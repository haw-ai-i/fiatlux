"""Climb clip variant for S10: the right hand carries the bulb and takes no hold.

Takes a finished two-handed clip and changes only the right arm: it starts in the benchmark's
S10 spawn pose (ARM_CRADLE, bulb in front of the pelvis), moves to a fixed CARRY pose before the
ladder is in the way, and stays there to the end. Pelvis, legs, waist and the left arm are
untouched. Every body pose and velocity is rebuilt from the joints; the right hand's contact
label is 0 throughout, so only the left hand is ever asked to hold.

    python make_bulb_carry_clip.py --clip <in> --out <out> --carry sp sr sy el wr wp wy
"""
import argparse, json, pathlib, shutil, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import g1_arm_fk as K

CRADLE = [-0.35, 0.00, 0.00, 0.35, 1.57, 0.0, 0.0]          # fiatlux_task.poses.ARM_CRADLE
ap = argparse.ArgumentParser()
ap.add_argument("--clip", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--carry", type=float, nargs=7, required=True, help="right arm: shoulder pitch roll yaw, elbow, wrist roll pitch yaw")
ap.add_argument("--hold_cradle", type=float, default=0.2, help="s in the spawn pose before the arm moves")
ap.add_argument("--blend", type=float, default=0.0, help="s to go from the spawn pose to the carry pose")
a = ap.parse_args()
src, dst = pathlib.Path(a.clip), pathlib.Path(a.out)
dst.mkdir(parents=True, exist_ok=True)
for f in src.iterdir():
    if f.name != "motion.npz":
        shutil.copy(f, dst / f.name)
d = dict(np.load(src / "motion.npz")); m = json.loads((src / "meta.json").read_text())
jn, bn, fps = m["joint_names"], m["body_names"], float(m["fps"])
T = len(d["joint_pos"])
t = np.arange(T) / fps
# --blend 0: the carry pose from the first frame. The clip starts close enough to the ladder that the
# spawn pose's hand (37 cm ahead of the pelvis) is already 41 mm inside tread 3, so the arm has to be
# in its carry pose before the clip begins; getting there from the spawn pose is the runner's job.
u = np.ones(T) if a.blend <= 0 else np.clip((t - a.hold_cradle) / a.blend, 0.0, 1.0)
w = (10 * u**3 - 15 * u**4 + 6 * u**5)[:, None]                 # minimum jerk
arm = (1 - w) * np.array(CRADLE) + w * np.array(a.carry)
q = d["joint_pos"].copy()
for k, n in enumerate(K.ARM):
    q[:, jn.index(n.format(s="right"))] = arm[:, k]
pel = bn.index("pelvis")
allb = K.fk_all(d["body_pos_w"][:, pel], d["body_quat_w"][:, pel], q, jn)
bp, bq = d["body_pos_w"].copy(), d["body_quat_w"].copy()
chg = 0.0
for i, b in enumerate(bn):
    p_, q_ = allb[b]
    if not b.startswith("right_") or not any(s in b for s in ("shoulder", "elbow", "wrist")):
        chg = max(chg, float(np.abs(p_ - bp[:, i]).max()))          # everything else must come back unchanged
    bp[:, i], bq[:, i] = p_, q_
dq_ = K.qmul(np.roll(bq, -1, axis=0), K.qconj(bq)); dq_[-1] = dq_[-2]
dq_ = dq_ * np.sign(dq_[..., :1] + 1e-12)
oc = d["object_contact"].copy(); held = int((oc[:, 1] > 0).sum()); oc[:, 1] = 0
np.savez(dst / "motion.npz", joint_pos=q, joint_vel=np.gradient(q, 1.0 / fps, axis=0), body_pos_w=bp, body_quat_w=bq,
         body_lin_vel_w=np.gradient(bp, 1.0 / fps, axis=0), body_ang_vel_w=2.0 * dq_[..., 1:] * fps, object_contact=oc)
hj = dst / "holds.json"
if hj.exists():
    h = json.loads(hj.read_text()); h["bulb_carry"] = dict(right_arm=list(a.carry), cradle=CRADLE, hold_cradle=a.hold_cradle, blend=a.blend)
    h.get("schedule", {})["right"] = []; hj.write_text(json.dumps(h, indent=1))
print(("right arm: carry pose from the first frame; " if a.blend <= 0
       else f"right arm: cradle for {a.hold_cradle:.1f} s, carry pose from {a.hold_cradle + a.blend:.1f} s; ")
      + f"peak arm speed {np.abs(np.gradient(arm, 1.0 / fps, axis=0)).max():.2f} rad/s")
print(f"right-hand hold labels cleared: {held} steps; other bodies moved by at most {chg*1000:.2f} mm (rebuild check)")
