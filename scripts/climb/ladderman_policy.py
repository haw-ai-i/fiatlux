"""Load a LadderMan teacher checkpoint and turn it into fiatlux joint-position actions.

Three stages, each reproduced from the trained artifact rather than from the paper:

1. **Normalize.** The checkpoint stores four ``ObservationNorm`` transforms (one per VecNorm
   group) with ``standard_normal=True``, so each group is ``(x - loc) / scale``. Their ``loc``
   equals the running mean to 0.0 and ``scale`` is the running std floored at 0.01 -- we use the
   STORED tensors, not a re-derivation, because those are the numbers the network was fed.
2. **Run the two networks.**  ``encoder_priv([priv, object]) -> feature(256)`` then
   ``actor([command, policy, feature]) -> 23``.  Concatenation order is the ``in_keys`` order from
   ``ppo_teacher.build_teacher_networks``, with ``CatTensors(..., sort=False)`` -- the order is
   NOT alphabetical and getting it wrong changes nothing observable except the robot's behaviour.
3. **Add the residual and convert.**  ``residual_action=True`` means the 23 outputs are an offset
   on the reference pose in action units, so the absolute LadderMan target is
   ``ref_joint_pos + net_out * action_scaling`` (see ladderman_contract). The six wrists are held
   at LadderMan's default. That absolute target is then re-expressed in OUR action units, which
   differ in both offset (their default pose is a deeper crouch) and gain (ours is a flat 0.5).

The networks are rebuilt by hand because LadderMan's are tensordict modules that would drag in
its whole package; the shapes below are asserted against the checkpoint, so a changed
architecture fails loudly here instead of quietly producing a 23-vector of nonsense.
"""

from __future__ import annotations

import ladderman_obs
import torch
from ladderman_contract import FIATLUX_ACTION_SCALE
from torch import nn

# make_mlp(num_units, activation=nn.Mish, norm="before") -> [Linear, LayerNorm, Mish] per unit
_ACT = nn.Mish


def _mlp(in_dim: int, units: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for n in units:
        layers += [nn.Linear(d, n), nn.LayerNorm(n), _ACT()]
        d = n
    return nn.Sequential(*layers)


class LadderManPolicy(nn.Module):
    def __init__(self, checkpoint_path: str, device: torch.device | str = "cuda"):
        super().__init__()
        ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.device = torch.device(device)

        # -- normalizers, matched to groups by width ------------------------------------------
        env = ck["env"]
        by_width = {}
        for i in (2, 3, 4, 5):
            loc, scale = env[f"transforms.{i}.loc"], env[f"transforms.{i}.scale"]
            if not bool(env[f"transforms.{i}.standard_normal"]):
                raise ValueError(f"transform {i} is not standard_normal; the formula differs")
            by_width[loc.shape[0]] = (loc.to(self.device), scale.to(self.device))
        self.norm = {}
        for g in ladderman_obs.VECNORM_GROUPS:
            w = ladderman_obs.width(g)
            if w not in by_width:
                raise KeyError(f"no stored normalizer of width {w} for group '{g}'")
            self.norm[g] = by_width[w]

        # -- networks --------------------------------------------------------------------------
        enc_in = ladderman_obs.width("priv") + ladderman_obs.width("object")
        act_in = ladderman_obs.width("command") + ladderman_obs.width("policy") + 256
        self.encoder = nn.Sequential(_mlp(enc_in, [256]), nn.Linear(256, 256))
        self.trunk = _mlp(act_in, [512, 256, 256])
        self.head = nn.Linear(256, ladderman_obs.NUM_ACTUATED)

        e, a = ck["policy"]["encoder_priv"], ck["policy"]["actor"]
        self._load(self.encoder[0][0], e, "module.1.module.0.0")
        self._load(self.encoder[0][1], e, "module.1.module.0.1")
        self._load(self.encoder[1], e, "module.1.module.1")
        for dst, src in ((0, 0), (1, 1), (3, 3), (4, 4), (6, 6), (7, 7)):
            self._load(self.trunk[dst], a, f"module.0.module.1.module.{src}")
        self._load(self.head, a, "module.0.module.2.module.actor_mean")
        self.to(self.device).eval()

    @staticmethod
    def _load(mod: nn.Module, sd: dict, prefix: str) -> None:
        w, b = sd[f"{prefix}.weight"], sd[f"{prefix}.bias"]
        if tuple(w.shape) != tuple(mod.weight.shape):
            raise ValueError(f"{prefix}: checkpoint {tuple(w.shape)} vs built {tuple(mod.weight.shape)}")
        mod.weight.data.copy_(w)
        mod.bias.data.copy_(b)

    def _n(self, g: str, x: torch.Tensor) -> torch.Tensor:
        loc, scale = self.norm[g]
        return (x - loc) / scale

    @torch.no_grad()
    def act(self, groups: dict[str, torch.Tensor]) -> torch.Tensor:
        """23 LadderMan action units, reference residual already added."""
        feat = self.encoder(torch.cat([self._n("priv", groups["priv"]), self._n("object", groups["object"])], -1))
        out = self.head(
            self.trunk(
                torch.cat([self._n("command", groups["command"]), self._n("policy", groups["policy"]), feat], -1)
            )
        )
        return groups["ref_joint_pos_"] + out  # RefJointPos: ref_jpos + action

    @torch.no_grad()
    def to_fiatlux(
        self, action_lm: torch.Tensor, obs, match_torque: bool = False, wrist_from_ref: bool = False
    ) -> torch.Tensor:
        """Re-express LadderMan's absolute joint target in our action units.

        theirs: pos_target = lm_default + applied_action * action_scaling   (23; wrists held)
        ours:   pos_target = fx_default + FIATLUX_ACTION_SCALE * action     (every joint)
        """
        robot = obs.robot
        n_fx = robot.data.joint_pos.shape[1]
        target = robot.data.default_joint_pos.clone()  # [N, n_fx]
        lm_target = obs.lm_default[None].expand(action_lm.shape[0], -1).clone()
        lm_target[:, obs.act_mo] = obs.lm_default[obs.act_mo][None] + action_lm * obs.act_scaling[None]
        if wrist_from_ref:
            # Teachers trained with JointPositionRefWrist: the wrists' PD target is the clip's
            # wrist angle at the current phase, not the default pose.
            lm_target[:, obs.wrist_mo] = obs.ref_joint_pos[obs.t][:, obs.wrist_mo]
        if match_torque:
            # Both sides run Isaac Lab's implicit PD: tau = kp*(q* - q) - kd*qd. The policy chose
            # q* for THEIR gains, so applying it under ours multiplies the torque. Solve for the
            # target that makes OUR actuator produce THEIR torque:
            #     q*_fx = q + [ kp_lm*(q*_lm - q) + (kd_fx - kd_lm)*qd ] / kp_fx
            # Exact for the PD term; it cannot fix the armature, link masses or effort limits,
            # which differ because this is a different robot, not just different gains.
            q = robot.data.joint_pos[:, obs.j_fx]
            qd = robot.data.joint_vel[:, obs.j_fx]
            kp_fx = robot.data.joint_stiffness[:, obs.j_fx]
            kd_fx = robot.data.joint_damping[:, obs.j_fx]
            kp_lm, kd_lm = obs.lm_kp[None], obs.lm_kd[None]
            safe = kp_fx.clamp_min(1e-6)
            lm_target = q + (kp_lm * (lm_target - q) + (kd_fx - kd_lm) * qd) / safe
        target[:, obs.j_fx] = lm_target
        a = (target - robot.data.default_joint_pos) / FIATLUX_ACTION_SCALE
        if a.shape[1] != n_fx:
            raise ValueError(f"action is {a.shape[1]} wide, env expects {n_fx}")
        return a
