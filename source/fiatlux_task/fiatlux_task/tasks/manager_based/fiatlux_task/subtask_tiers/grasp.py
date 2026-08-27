# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tier for the grasp mode -- S08 (grab-the-bulb), its only member since the ladder legs folded.

A grasp subtask ends by taking an object's weight: a reach-progress channel toward the object, a
contact-force bootstrap so the policy can find "close the hand" before the sparse gate is
reachable, a grip-force penalty on the same filtered channel, and a single-target contact sensor
the leaf's gate reads for the force conjunct. The family's usual ``sustained`` window is
shortened here (0.5 s, not the 1.0 s ``place`` tier uses): a grasp only needs to prove the hold
isn't a glancing contact, not that the object has settled.

Kept as a tier at one member rather than inlined into the leaf: the split it encodes -- which
terms exist here, which numbers live in the leaf -- is what keeps routine retuning out of shared
code, and that argument does not depend on the member count.
"""

from collections.abc import Callable

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from .. import mdp
from ..mdp import grasp_terms
from ..scene_cfg import G1ReplaceSceneCfg
from ..subtask_env_cfg import SubtaskEnvCfg, SubtaskRewardsCfg

# Shorter than ``place``'s 1.0 s: proving a hold isn't a glance, not that the object has settled.
GRASP_SUSTAIN_SECONDS = 0.5

# Dense bootstrap saturates well under the leaf's fragility threshold (the 50 N bulb-glass limit)
# so it never fights the grip-force penalty on the way to a real grasp -- it only has to signal
# "the hand is pressing", not "the hand is holding". PROVISIONAL.
GRASP_BOOTSTRAP_SATURATION_N = 10.0


def add_grasp_contact_sensor(scene: G1ReplaceSceneCfg, target_prim_path: str) -> None:
    """Hand-only contact sensor filtered to ONE target prim, for the grasp gate's force conjunct.

    Mirrors ``subtask_tiers.place.add_release_contact_sensor``: the scene's shared
    ``hand_contact`` filters two prims in the replace preset (Bulb + OldBulb), so its columns
    cannot be attributed to a single one of them, and ``force_matrix_w`` reads zero unless the
    filtered target is a rigid body.
    """
    scene.grasp_contact = ContactSensorCfg(
        prim_path=scene.hand_contact.prim_path,
        filter_prim_paths_expr=[target_prim_path],
        history_length=1,
        track_air_time=False,
    )


@configclass
class GraspRewardsCfg(SubtaskRewardsCfg):
    """Reach + bootstrap + grip-force discipline; ``distance_fn`` comes from the leaf."""

    reach_progress = RewTerm(
        func=mdp.distance_progress, weight=500.0, params={"distance_fn": grasp_terms.hand_bulb_distance}
    )
    contact_bootstrap = RewTerm(
        func=grasp_terms.grasp_contact_bootstrap,
        weight=20.0,
        params={"sensor_cfg": SceneEntityCfg("grasp_contact"), "saturation_force": GRASP_BOOTSTRAP_SATURATION_N},
    )
    grip_force_penalty = RewTerm(
        func=mdp.hand_contact_force_l2, weight=-1.0e-4, params={"sensor_cfg": SceneEntityCfg("grasp_contact")}
    )


@configclass
class GraspSubtaskCfg(SubtaskEnvCfg):
    """A subtask whose job is to grasp an object and take its weight."""

    progress_distance_fn: Callable | None = None
    rewards: GraspRewardsCfg = GraspRewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.progress_distance_fn is None:
            raise ValueError(f"{type(self).__name__} must set progress_distance_fn")
        self.rewards.reach_progress.params["distance_fn"] = self.progress_distance_fn
