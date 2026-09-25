# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Argparse args shared between ``record_run.py`` and ``verify_scene.py``'s CLIs.

Deliberately its own module outside the ``fiatlux_task`` package, not just constants in
``fiatlux_task/policy.py``: importing anything under ``fiatlux_task`` runs
``fiatlux_task/__init__.py``'s ``from .tasks import *``, which -- even though it silently
no-ops pre-Kit (``tasks/__init__.py`` swallows the failure) -- still leaves ``torch`` fully
imported and cached in ``sys.modules`` along the way (confirmed: a bare ``import
fiatlux_task.policy``, even with ``policy.py``'s own ``torch`` import made lazy, still leaves
``torch`` in ``sys.modules``). Both scripts import this module before ``AppLauncher`` runs
specifically to avoid paying that cost on the ``--help``/bad-argument fast path; this module
imports only ``argparse`` (already required pre-launch by both callers anyway), so there is
nothing heavy left to pull in early.

``add_policy_cli_args`` adds the full ``type``/``default``/``choices``/``help`` for each flag,
not just the help strings, so a future second caller can't drift on those either
(``verify_scene.py`` currently only imports ``ROBOT_CHOICES`` from this module).
"""

import argparse

POLICY_SPEC_HELP = (
    "Policy spec: zero | random | basic_standard | basic_cheatcode | wbc_stand | "
    "sonic_stand | groot[:<host:port>] | rsl_rl[:<ckpt>] | <path>.pt (or jit:<path>). "
    "See fiatlux_task/policy.py."
)
CHECKPOINT_HELP = "Checkpoint path for rsl_rl policies."
INSTRUCTION_HELP = "Language instruction for VLA policies (groot); default: the task's canonical sentence."
ROBOT_CHOICES = ("inspire", "dex3")
ROBOT_HELP = "G1 hand variant. dex3 matches GR00T's REAL_G1 embodiment."


def add_policy_cli_args(parser: argparse.ArgumentParser) -> None:
    """Add ``--policy``/``--checkpoint``/``--instruction``/``--robot``, identically, to ``parser``."""
    parser.add_argument("--policy", type=str, default="zero", help=POLICY_SPEC_HELP)
    parser.add_argument("--checkpoint", type=str, default=None, help=CHECKPOINT_HELP)
    parser.add_argument("--instruction", type=str, default=None, help=INSTRUCTION_HELP)
    parser.add_argument("--robot", type=str, default="inspire", choices=ROBOT_CHOICES, help=ROBOT_HELP)
