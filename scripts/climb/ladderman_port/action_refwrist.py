"""LadderMan's joint-position action, with the WRISTS driven from the reference clip.

Deployed to LadderMan as ``active_adaptation/envs/mdp/action_refwrist.py`` and selected with
``task.action._target_=active_adaptation.envs.mdp.action_refwrist.JointPositionRefWrist``.

Why. LadderMan's policy has 23 outputs; the six wrist joints are not among them and are held at
their default (straight) by the PD controller. That is fine for a mitt hooked over a rung. Our
clip's holds are not that: a Dex3 has to lie flat on a tread edge, pitch up onto its finger pads,
roll over a rim -- wrist angles up to 90 deg, on a schedule. Held straight, the hands arrive
edge-on, and the episode ends 25 steps in on the wrist links' orientation error.

So the wrists follow the clip open-loop: each step their PD target is the reference's wrist
angle at the current phase. The policy still outputs 23 numbers and nothing about its
observations or network changes; whoever runs the trained policy elsewhere must drive the wrists
the same way (the reference is available there too).
"""

from __future__ import annotations

import torch

from .action import JointPosition


class JointPositionRefWrist(JointPosition):
    def __init__(self, env, ref_joint_names=(".*_wrist_(roll|pitch|yaw)_joint",), **kwargs):
        super().__init__(env, **kwargs)
        ids, names = self.asset.find_joints(list(ref_joint_names))
        clash = sorted(set(ids) & set(self.joint_ids))
        if clash:
            raise ValueError(
                f"reference-driven joints are also policy-actuated: {[self.asset.joint_names[i] for i in clash]}"
            )
        self.ref_ids = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        print(f"[action] wrists follow the reference: {names}")

    def __call__(self, action, substep: int):
        cmd = getattr(self.env, "command_manager", None)
        ref = getattr(cmd, "future_ref_motion", None)
        if ref is not None:
            q = ref.joint_pos[:, 0][:, cmd.asset_joint_idx_motion]  # [N, all joints], asset order
            self.offset[:, self.ref_ids] = q[:, self.ref_ids] - self.default_joint_pos[:, self.ref_ids]
        return super().__call__(action, substep)
