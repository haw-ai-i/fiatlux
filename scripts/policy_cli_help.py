# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``--policy``/``--checkpoint`` argparse help text, shared by ``eval.py`` and ``record_run.py``.

Deliberately its own module outside the ``fiatlux_task`` package, not just a constant in
``fiatlux_task/policy.py``: importing anything under ``fiatlux_task`` runs
``fiatlux_task/__init__.py``'s ``from .tasks import *``, which -- even though it silently
no-ops pre-Kit (``tasks/__init__.py`` swallows the failure) -- still leaves ``torch`` fully
imported and cached in ``sys.modules`` along the way (confirmed: a bare ``import
fiatlux_task.policy``, even with ``policy.py``'s own ``torch`` import made lazy, still leaves
``torch`` in ``sys.modules``). Both scripts import this module before ``AppLauncher`` runs
specifically to avoid paying that cost on the ``--help``/bad-argument fast path; this module
has zero imports of its own so there is nothing left to pull in early.
"""

POLICY_SPEC_HELP = (
    "Policy spec: zero | random | basic_standard | basic_cheatcode | wbc_stand | "
    "sonic_stand | groot[:<host:port>] | rsl_rl[:<ckpt>] | <path>.pt (or jit:<path>). "
    "See fiatlux_task/policy.py."
)
CHECKPOINT_HELP = "Checkpoint path for rsl_rl policies."
