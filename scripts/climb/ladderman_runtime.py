"""Compute LadderMan's four observation groups from a LIVE fiatlux env.

The teacher was trained inside LadderMan's own environment, against a procedurally generated
ladder, with a command manager driving a reference clip. None of that exists in S02/S10/S04/S12.
This module rebuilds all of it from the outside: it loads the same reference, keeps its own
phase clock and history buffers, places the motion into the benchmark's world, and emits the
four tensors ``Seq(encoder_priv, actor)`` expects.

``ladderman_obs.py`` fixes the LAYOUT (which terms, in which order, how wide) and checks it
against the checkpoint's own VecNorm statistics. This file fixes the VALUES. Keep them together:
a width error raises, a value error does not.

Three things here are the difference between working and silently wrong:

* **Joint order.** The motion's ``meta.json`` lists LadderMan's 29 joints in URDF order; the
  fiatlux articulation reports Isaac Lab's breadth-first order AND carries extra hand joints.
  Every index is resolved BY NAME, and a missing name raises rather than shifting a column.
* **The default pose.** ``joint_pos_history`` is relative to the action manager's offset, which
  for the teacher is LADDERMAN's default pose -- a deep crouch that differs from ours by up to
  27 deg at the knee. Using our default shifts every leg column with nothing raised.
* **The ladder frame.** LadderMan's rungs are at ``base_position + i*spacing*[cos,0,sin]`` with
  their orientations WRITTEN with a -90 deg z yaw, so ``ladder_heading_b`` measures the angle
  to a rung's LONG AXIS, not to the climb direction. See RUNG_YAW_OFFSET below. In the benchmark the climb direction is
  the ladder's local +y, and rung 0 is the first tread's centre -- both measured, see
  ``scripts/teleop/fullbody/ladder_treads.py`` and ``scripts/climb/measure_climb_geometry.py``.

Known approximation, stated rather than hidden: LadderMan averages ``joint_pos`` over the last
two PHYSICS substeps inside the step loop. A policy callable only sees the env between steps, so
we use the post-step value. At 200 Hz against a 50 Hz policy the two differ by one 5 ms sample of
joint motion; it is a real difference, not a zero, and it belongs on the list of things to check
if a loaded policy behaves worse here than it did in training.
"""

from __future__ import annotations

import json
import math
import os
import pathlib

import numpy as np
import torch
from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply_inverse,
    quat_conjugate,
    quat_mul,
    wrap_to_pi,
    yaw_quat,
)

from ladderman_contract import ACTION_SCALING, UNACTUATED  # noqa: E402

# Written by LadderMan's own trainer next to every run (`asset_meta.json`). It carries the
# articulation's joint order and default pose as the trainer actually saw them, which is the
# authority for both: `MotionDataset.create_from_path` REORDERS the clip's joints from the
# file's URDF order into this Isaac order, so every downstream term -- joint_pos_history,
# applied_torque, the tracking-joint slice, the 23-wide action -- speaks Isaac order. Rebuilding
# the default pose from regex patterns instead (an earlier version of this file) also silently
# dropped the joints whose default is 0.
# LadderMan's ladder_heading_b reads the RUNG's OWN quaternion
# (observations.py: object_quat_w[:, rung_id]), not the climb direction, and ladder_env.py
# builds every rung through `z_rot = (cos(-pi/4), 0, 0, sin(-pi/4))` -- a half-angle, so a -90
# deg yaw -- with the comment "rotate about z so that the rungs run along the world y axis".
# The rung's reported yaw is therefore its LONG AXIS, 90 deg off the climb direction.
#
# This file previously asserted the opposite ("their orientations never written, so
# ladder_heading_b measures the angle to the CLIMB DIRECTION"). That was wrong, and it fed the
# policy cos(dyaw)=+1.000 where every training sample had cos ~ 0: the checkpoint's own VecNorm
# running mean for that element is -0.0499 +/- 0.1223, putting our value at z = +8.58.
# Overridable only so the fix can be A/B'd against the old (wrong) 0.0 without editing code.
RUNG_YAW_OFFSET = float(os.environ.get("FX_RUNG_YAW_OFFSET", -math.pi / 2))

ASSET_META = pathlib.Path(__file__).with_name("ladderman_asset_meta.json")

# LadderMan cfg/task/base/ladder.yaml
FUTURE_STEPS = (1, 2, 8, 16, 32)
TARGET_FPS = 50
TRACKING_BODY_PATTERNS = (
    "{s}_hip_pitch_link", "{s}_hip_yaw_link", "{s}_knee_link", "{s}_ankle_roll_link",
    "{s}_shoulder_pitch_link", "{s}_elbow_link", "{s}_wrist_yaw_link",
)
TRACKING_BODY_SINGLETONS = ("pelvis", "torso_link")
TRACKING_JOINT_SUFFIXES = (
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "{s}_hip_pitch_joint", "{s}_hip_roll_joint", "{s}_hip_yaw_joint",
    "{s}_knee_joint", "{s}_ankle_pitch_joint", "{s}_ankle_roll_joint",
    "{s}_shoulder_pitch_joint", "{s}_shoulder_roll_joint", "{s}_shoulder_yaw_joint",
    "{s}_elbow_joint",
)
POLICY_JOINT_HISTORY = (0, 1, 2, 3, 4, 8)
PRIV_HISTORY = (0, 1, 2, 3, 4, 5, 6, 7, 8)
PREV_ACTION_STEPS = 3

# The ladder LadderMan thinks it is climbing -- must match lm_train_*.sbatch.
LM_BASE = (-0.58, 0.0, 0.196)
LM_INCL_DEG = 70.3
LM_SPACING = 0.2492


def _ordered_tracking_bodies() -> list[str]:
    """The 16 tracking bodies, in the order isaaclab's find_bodies returns them.

    find_bodies sorts by the asset's own body index, so the ORDER here is only a candidate set;
    the real order is resolved against the articulation in __init__. Listing them explicitly means a
    renamed link raises instead of silently dropping a body and shrinking the group.
    """
    names = list(TRACKING_BODY_SINGLETONS)
    for pat in TRACKING_BODY_PATTERNS:
        for side in ("left", "right"):
            names.append(pat.format(s=side))
    return names


def _ordered_tracking_joints() -> list[str]:
    names = []
    for pat in TRACKING_JOINT_SUFFIXES:
        if "{s}" in pat:
            names.extend(pat.format(s=side) for side in ("left", "right"))
        else:
            names.append(pat)
    return names


def _resample(motion: dict, src_fps: float, dst_fps: int) -> dict:
    """LadderMan's own resampling, reproduced so the reference we drive matches the one it
    trained on frame for frame (utils/motion.py::interpolate)."""
    from scipy.spatial.transform import Rotation as sRot
    from scipy.spatial.transform import Slerp

    T = motion["joint_pos"].shape[0]
    end_t = T / src_fps
    ts_src = np.arange(0, end_t, 1.0 / src_fps)
    ts_dst = np.arange(0, end_t, 1.0 / dst_fps)
    if ts_dst[-1] > ts_src[-1]:
        ts_dst = ts_dst[:-1]
    if len(ts_src) != T:
        raise ValueError(
            f"this clip has {T} frames but np.arange gives {len(ts_src)} source samples -- "
            "rebuild it through build_topout_motion.fit_to_loader()"
        )

    def lerp(x):
        flat = x.reshape(T, -1)
        out = np.stack([np.interp(ts_dst, ts_src, flat[:, i]) for i in range(flat.shape[1])], -1)
        return out.reshape(len(ts_dst), *x.shape[1:])

    out = {k: lerp(motion[k]) for k in
           ("body_pos_w", "body_lin_vel_w", "body_ang_vel_w", "joint_pos", "joint_vel")}
    q = motion["body_quat_w"]
    nb = q.shape[1]
    oq = np.empty((len(ts_dst), nb, 4))
    for i in range(nb):
        s = Slerp(ts_src, sRot.from_quat(q[:, i, [1, 2, 3, 0]]))
        oq[:, i, :] = s(ts_dst).as_quat()[..., [3, 0, 1, 2]]
    out["body_quat_w"] = oq
    out["object_contact"] = motion["object_contact"][: len(ts_dst)]
    return out


class LadderManObs:
    """Drives a LadderMan teacher inside a fiatlux subtask env."""

    def __init__(self, env, motion_dir: str, ladder_asset: str = "ladder",
                 rung0_local=(0.0, -0.3804, 0.196), climb_dir_local=(0.0, 1.0, 0.0),
                 asset_meta: str | None = None):
        self.env = env
        self.device = env.device
        self.N = env.num_envs
        self.robot = env.scene["robot"]
        self.ladder = env.scene[ladder_asset]

        am = json.loads(pathlib.Path(asset_meta or ASSET_META).read_text())
        self.lm_joint_names: list[str] = list(am["joint_names_isaac"])     # 29, Isaac order
        self.lm_body_names_isaac: list[str] = list(am["body_names_isaac"])

        d = pathlib.Path(motion_dir)
        meta = json.loads((d / "meta.json").read_text())
        raw = dict(np.load(d / "motion.npz"))
        file_joints: list[str] = list(meta["joint_names"])                 # URDF order
        self.lm_body_names: list[str] = list(meta["body_names"])
        m = _resample(raw, float(meta["fps"]), TARGET_FPS)
        self.motion_len = int(m["joint_pos"].shape[0])

        # reorder the clip's joint tracks into Isaac order, exactly as create_from_path does
        src = [file_joints.index(n) for n in self.lm_joint_names]
        m["joint_pos"] = m["joint_pos"][:, src]
        m["joint_vel"] = m["joint_vel"][:, src]

        # ---- index maps, all BY NAME ---------------------------------------------------------
        fx_j = list(self.robot.joint_names)
        fx_b = list(self.robot.body_names)
        missing = [n for n in self.lm_joint_names if n not in fx_j]
        if missing:
            raise KeyError(f"fiatlux robot is missing LadderMan joints: {missing}")
        self.j_fx = torch.tensor([fx_j.index(n) for n in self.lm_joint_names], device=self.device)

        tb = [n for n in _ordered_tracking_bodies() if n in fx_b]
        if len(tb) != 16:
            raise KeyError(f"resolved {len(tb)} tracking bodies, expected 16: {tb}")
        tb.sort(key=self.lm_body_names_isaac.index)  # LadderMan's asset order, not ours
        self.track_b_fx = torch.tensor([fx_b.index(n) for n in tb], device=self.device)
        self.track_b_mo = torch.tensor([self.lm_body_names.index(n) for n in tb], device=self.device)
        self.track_bodies = tb

        tj = _ordered_tracking_joints()
        tj.sort(key=self.lm_joint_names.index)       # Isaac order, as find_joints returns
        self.track_j_mo = torch.tensor([self.lm_joint_names.index(n) for n in tj], device=self.device)
        self.track_joints = tj

        self.root_mo = self.lm_body_names.index("pelvis")

        # ---- LadderMan's default pose, which joint_pos_history is measured against ------------
        self.lm_default = torch.tensor(am["default_joint_pos"], dtype=torch.float32,
                                       device=self.device)            # [29], Isaac order
        # The actuator model the policy's residual was learned under. Ours is a different robot
        # with much stiffer, much less damped joints (waist 8.8x, shoulders 7.0x; LadderMan runs
        # a uniform kd ~ 0.064*kp where ours is 0.013-0.02), so the same position target produces
        # several times the torque here. See LadderManPolicy.to_fiatlux(match_torque=True).
        self.lm_kp = torch.tensor(am["stiffness"], dtype=torch.float32, device=self.device)
        self.lm_kd = torch.tensor(am["damping"], dtype=torch.float32, device=self.device)

        # ---- the 23 actuated joints, in the action manager's order ---------------------------
        act_names = [n for n in self.lm_joint_names if not any(u in n for u in UNACTUATED)]
        if len(act_names) != 23:
            raise KeyError(f"resolved {len(act_names)} actuated joints, expected 23")
        self.act_names = act_names
        self.act_mo = torch.tensor([self.lm_joint_names.index(n) for n in act_names],
                                   device=self.device)
        scal = []
        for n in act_names:
            hit = [v for k, v in ACTION_SCALING.items() if k in n]
            if len(hit) != 1:
                raise KeyError(f"{n}: matched {len(hit)} action_scaling entries, expected 1")
            scal.append(hit[0])
        self.act_scaling = torch.tensor(scal, dtype=torch.float32, device=self.device)

        # ---- place the motion into this world ------------------------------------------------
        self._set_frame(rung0_local, climb_dir_local)
        t = torch.as_tensor
        self.ref_body_pos = self._to_world_p(t(m["body_pos_w"], dtype=torch.float32, device=self.device))
        self.ref_body_quat = self._to_world_q(t(m["body_quat_w"], dtype=torch.float32, device=self.device))
        self.ref_body_linvel = self._rot_only(t(m["body_lin_vel_w"], dtype=torch.float32, device=self.device))
        self.ref_body_angvel = self._rot_only(t(m["body_ang_vel_w"], dtype=torch.float32, device=self.device))
        self.ref_joint_pos = t(m["joint_pos"], dtype=torch.float32, device=self.device)

        # ---- buffers -------------------------------------------------------------------------
        nj = len(self.lm_joint_names)
        self.t = torch.zeros(self.N, dtype=torch.long, device=self.device)
        self.buf_angvel = torch.zeros(self.N, 9, 3, device=self.device)
        self.buf_gravity = torch.zeros(self.N, 9, 3, device=self.device)
        self.buf_jpos = torch.zeros(self.N, 9, nj, device=self.device)
        self.buf_action = torch.zeros(self.N, PREV_ACTION_STEPS, 23, device=self.device)
        self.last_action = torch.zeros(self.N, 23, device=self.device)
        self.reset()

    # -- frame -------------------------------------------------------------------------------
    def _set_frame(self, rung0_local, climb_dir_local) -> None:
        """Rigid yaw+offset taking the motion's world into this env's world.

        The motion was authored with rung 0 at LM_BASE and the climb along +x. Here rung 0 is a
        tread centre in the ladder's local frame and the climb runs along the ladder's local +y,
        so the map is a yaw by the difference plus the offset between the two rung-0 points.
        """
        lp = torch.as_tensor(self.ladder.data.root_pos_w, device=self.device)[:1]       # [1,3]
        lq = torch.as_tensor(self.ladder.data.root_quat_w, device=self.device)[:1]
        r0 = torch.tensor(rung0_local, device=self.device, dtype=torch.float32)[None]
        cd = torch.tensor(climb_dir_local, device=self.device, dtype=torch.float32)[None]
        R = matrix_from_quat(lq)                                                        # [1,3,3]
        self.rung0_w = lp + (R @ r0[..., None]).squeeze(-1)
        climb_w = (R @ cd[..., None]).squeeze(-1)
        self.climb_yaw = torch.atan2(climb_w[:, 1], climb_w[:, 0])                      # [1]
        dyaw = self.climb_yaw - 0.0                                                     # motion climbs +x
        h = dyaw / 2
        self.dq = torch.stack([torch.cos(h), torch.zeros_like(h), torch.zeros_like(h),
                               torch.sin(h)], -1)                                       # [1,4]
        base = torch.tensor(LM_BASE, device=self.device, dtype=torch.float32)[None]
        self.dz = float((self.rung0_w[0, 2] - base[0, 2]).item())
        self.base = base

    def _to_world_p(self, p: torch.Tensor) -> torch.Tensor:
        flat = p.reshape(-1, 3) - self.base
        q = self.dq.expand(flat.shape[0], 4)
        rot = quat_mul(q, torch.cat([torch.zeros_like(flat[:, :1]), flat], -1))
        rot = quat_mul(rot, quat_conjugate(q))[:, 1:]
        return (rot + self.rung0_w).reshape(p.shape)

    def _to_world_q(self, q: torch.Tensor) -> torch.Tensor:
        flat = q.reshape(-1, 4)
        return quat_mul(self.dq.expand(flat.shape[0], 4), flat).reshape(q.shape)

    def _rot_only(self, v: torch.Tensor) -> torch.Tensor:
        flat = v.reshape(-1, 3)
        q = self.dq.expand(flat.shape[0], 4)
        rot = quat_mul(q, torch.cat([torch.zeros_like(flat[:, :1]), flat], -1))
        return quat_mul(rot, quat_conjugate(q))[:, 1:].reshape(v.shape)

    # -- lifecycle ---------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.t[ids] = 0
        self.buf_angvel[ids] = self.robot.data.root_ang_vel_b[ids].unsqueeze(1)
        self.buf_gravity[ids] = self.robot.data.projected_gravity_b[ids].unsqueeze(1)
        self.buf_jpos[ids] = self.robot.data.joint_pos[:, self.j_fx][ids].unsqueeze(1)
        self.buf_action[ids] = 0.0
        self.last_action[ids] = 0.0

    def step(self, action_lm: torch.Tensor) -> None:
        """Record the action we just applied and advance the reference clock by one policy step."""
        self.last_action = action_lm
        self.buf_action = self.buf_action.roll(1, dims=1)
        self.buf_action[:, 0] = action_lm
        self.t = torch.clamp(self.t + 1, max=self.motion_len - 1)

    def update(self) -> None:
        self.buf_angvel = self.buf_angvel.roll(1, 1)
        self.buf_angvel[:, 0] = self.robot.data.root_ang_vel_b
        self.buf_gravity = self.buf_gravity.roll(1, 1)
        self.buf_gravity[:, 0] = self.robot.data.projected_gravity_b
        self.buf_jpos = self.buf_jpos.roll(1, 1)
        self.buf_jpos[:, 0] = self.robot.data.joint_pos[:, self.j_fx]

    # -- reference slices --------------------------------------------------------------------
    def _future_idx(self) -> torch.Tensor:
        f = torch.tensor(FUTURE_STEPS, device=self.device)
        return torch.clamp(self.t[:, None] + f[None], max=self.motion_len - 1)   # [N,F]

    # -- groups ------------------------------------------------------------------------------
    def command(self) -> torch.Tensor:
        fi = self._future_idx()
        rb = self.ref_body_pos[fi][:, :, self.track_b_mo]                # [N,F,B,3]
        rr_p = self.ref_body_pos[self.t][:, self.root_mo].clone()        # [N,3]
        rr_q = yaw_quat(self.ref_body_quat[self.t][:, self.root_mo])     # [N,4]
        rr_p[:, 2] = 0.0
        local = quat_apply_inverse(
            rr_q[:, None, None].expand(-1, rb.shape[1], rb.shape[2], -1),
            rb - rr_p[:, None, None],
        )
        rj = self.ref_joint_pos[fi][:, :, self.track_j_mo]               # [N,F,J]
        phase = (self.t.float() / self.motion_len)[:, None]
        return torch.cat([local.reshape(self.N, -1), rj.reshape(self.N, -1), phase], -1)

    def policy(self) -> torch.Tensor:
        jp = (self.buf_jpos - self.lm_default[None, None])[:, list(POLICY_JOINT_HISTORY)]
        return torch.cat([
            self.buf_angvel[:, [0]].reshape(self.N, -1),
            self.buf_gravity[:, [0]].reshape(self.N, -1),
            jp.reshape(self.N, -1),
            self.buf_action.reshape(self.N, -1),
        ], -1)

    def object(self) -> torch.Tensor:
        rq = yaw_quat(self.robot.data.root_quat_w)
        xy = quat_apply_inverse(rq, self.rung0_w.expand(self.N, 3) - self.robot.data.root_pos_w)[:, :2]
        robot_yaw = torch.atan2(
            2 * (rq[:, 0] * rq[:, 3] + rq[:, 1] * rq[:, 2]),
            1 - 2 * (rq[:, 2] ** 2 + rq[:, 3] ** 2),
        )
        dyaw = wrap_to_pi(self.climb_yaw.expand(self.N) + RUNG_YAW_OFFSET - robot_yaw)[:, None]
        return torch.cat([xy, torch.cos(dyaw), torch.sin(dyaw)], -1)

    def priv(self) -> torch.Tensor:
        fi = self._future_idx()
        F, B = fi.shape[1], len(self.track_bodies)
        rq_root = self.robot.data.root_quat_w
        rp_root = self.robot.data.root_pos_w

        ref_root_p = self.ref_body_pos[fi][:, :, self.root_mo]                       # [N,F,3]
        ref_root_q = self.ref_body_quat[fi][:, :, self.root_mo]
        rrp = quat_apply_inverse(rq_root[:, None].expand(-1, F, -1), ref_root_p - rp_root[:, None])
        rrq = quat_mul(quat_conjugate(rq_root[:, None]).expand(-1, F, -1), ref_root_q)
        rro = matrix_from_quat(rrq.reshape(-1, 4)).reshape(self.N, F, 3, 3)[:, :, :2, :]

        # reference, in the MOTION root's yaw frame
        mr_q = yaw_quat(self.ref_body_quat[self.t][:, self.root_mo])[:, None, None]
        mr_p = self.ref_body_pos[self.t][:, self.root_mo].clone()
        mr_p[:, 2] = 0.0
        e = (self.N, F, B, -1)
        ref_p = quat_apply_inverse(mr_q.expand(*e[:3], 4),
                                   self.ref_body_pos[fi][:, :, self.track_b_mo] - mr_p[:, None, None])
        ref_v = quat_apply_inverse(mr_q.expand(*e[:3], 4), self.ref_body_linvel[fi][:, :, self.track_b_mo])
        ref_w = quat_apply_inverse(mr_q.expand(*e[:3], 4), self.ref_body_angvel[fi][:, :, self.track_b_mo])
        ref_q = quat_mul(quat_conjugate(mr_q).expand(*e[:3], 4),
                         self.ref_body_quat[fi][:, :, self.track_b_mo])

        # robot, in its own yaw frame
        br_q = yaw_quat(rq_root)[:, None]
        bp = self.robot.data.body_pos_w[:, self.track_b_fx]
        brp = rp_root.clone(); brp[:, 2] = 0.0
        rob_p = quat_apply_inverse(br_q.expand(self.N, B, 4), bp - brp[:, None])
        rob_v = quat_apply_inverse(br_q.expand(self.N, B, 4), self.robot.data.body_lin_vel_w[:, self.track_b_fx])
        rob_w = quat_apply_inverse(br_q.expand(self.N, B, 4), self.robot.data.body_ang_vel_w[:, self.track_b_fx])
        rob_q = quat_mul(quat_conjugate(br_q).expand(self.N, B, 4),
                         self.robot.data.body_quat_w[:, self.track_b_fx])

        dq = quat_mul(quat_conjugate(rob_q[:, None]).expand(self.N, F, B, 4), ref_q)
        d_ori = matrix_from_quat(dq.reshape(-1, 4)).reshape(self.N, F, B, 3, 3)[..., :2, :]

        ank = [i for i, n in enumerate(self.track_bodies) if n.endswith("ankle_roll_link")]
        hb_names = [n for n in self.track_bodies if n.endswith("ankle_roll_link")] + ["pelvis", "torso_link"]
        hb = torch.tensor([list(self.robot.body_names).index(n) for n in hb_names], device=self.device)

        jp = (self.buf_jpos - self.lm_default[None, None])[:, list(PRIV_HISTORY)]
        return torch.cat([
            self.buf_angvel[:, list(PRIV_HISTORY)].reshape(self.N, -1),
            self.buf_gravity[:, list(PRIV_HISTORY)].reshape(self.N, -1),
            jp.reshape(self.N, -1),
            rrp.reshape(self.N, -1),
            rro.reshape(self.N, -1),
            (ref_p - rob_p[:, None]).reshape(self.N, -1),
            d_ori.reshape(self.N, -1),
            (ref_v - rob_v[:, None]).reshape(self.N, -1),
            (ref_w - rob_w[:, None]).reshape(self.N, -1),
            quat_apply_inverse(rq_root, self.robot.data.root_lin_vel_w).reshape(self.N, -1),
            rob_p[:, ank].reshape(self.N, -1),
            rob_v[:, ank].reshape(self.N, -1),
            self.robot.data.body_pos_w[:, hb, 2].reshape(self.N, -1),
            self.last_action.reshape(self.N, -1),
            self.robot.data.applied_torque[:, self.j_fx].reshape(self.N, -1),
        ], -1)

    def ref_joint_action(self) -> torch.Tensor:
        """``(ref_joint_pos - default) / action_scaling`` over the 23 actuated joints.

        The actor's 23 outputs are added to THIS, not used directly -- see RefJointPos in
        LadderMan's common.py. Substituting through the action manager, the absolute target is
        ``ref_joint_pos + net_out * action_scaling``.
        """
        ref = self.ref_joint_pos[self.t][:, self.act_mo]
        return (ref - self.lm_default[self.act_mo][None]) / self.act_scaling[None]

    def groups(self) -> dict[str, torch.Tensor]:
        return {"command": self.command(), "policy": self.policy(),
                "object": self.object(), "priv": self.priv(),
                "ref_joint_pos_": self.ref_joint_action()}
