# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One module per subtask mode: the reward channels and gate shape its leaves share.

Modes sit between ``SubtaskEnvCfg`` and the leaves. A mode declares WHICH terms exist and how they
are wired; a leaf declares the numbers and the predicates. Routine retuning therefore never touches
shared code, while a change here is honestly a change to every leaf under it.
"""
