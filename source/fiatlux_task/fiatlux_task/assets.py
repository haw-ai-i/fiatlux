# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Central asset locations for the Fiatlux benchmark.

Assets are synced into the repo-root ``assets/`` dir by ``assets/download_assets.sh``.
Override the root with the ``FIATLUX_ASSETS_DIR`` env var if they live elsewhere.

Every env and script should import paths from here rather than re-deriving the
repo root, so there is a single place to fix when the asset layout changes.
"""

import os

# ``source/fiatlux_task/fiatlux_task/assets.py`` -> repo root is three levels up.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir, os.pardir))

FIATLUX_ASSETS_DIR = os.environ.get(
    "FIATLUX_ASSETS_DIR", os.path.join(_REPO_ROOT, "assets")
)

# Unitree's official pre-assembled G1 USDs (mirrored into the GCS bucket from
# unitreerobotics/unitree_sim_isaaclab_usds). The legged (wholebody) Inspire-hand
# variant is the default; the Dex3-hand variant is staged for an easy future swap.
G1_USD = os.path.join(
    FIATLUX_ASSETS_DIR, "unitree_g1", "wholebody_inspire", "g1_29dof_with_inspire_rev_1_0.usd"
)
G1_DEX3_USD = os.path.join(
    FIATLUX_ASSETS_DIR, "unitree_g1", "wholebody_dex3", "g1_29dof_with_dex3_rev_1_0.usd"
)

# Bulb + fixture are BEHAVIOR-1K objects (synced under ``behavior1k_*/`` by
# ``download_assets.sh``). Each object id is its own dir; swap the id/variant here.
# - the bulb is the graspable rigid body (``ymomhw`` is a clean single-body variant),
# - the lamp stands in as the socket/fixture the bulb is seated into. Lamps are
#   multi-body BEHAVIOR-1K objects; the env spawns this one as a single rigid body by
#   deactivating its ``meta__*`` helper links (see ``g1_bulb_env_cfg`` / issue #14).
BULB_USD = os.path.join(FIATLUX_ASSETS_DIR, "behavior1k_bulb", "ymomhw", "usd", "ymomhw.usd")
SOCKET_USD = os.path.join(FIATLUX_ASSETS_DIR, "behavior1k_lamp", "bbentu", "usd", "bbentu.usd")

# Room dressing: table, warehouse backdrop, clutter, and an HDRI sky, mirrored once
# from Isaac Sim's own Nucleus content library (Isaac/Props, Isaac/Environments) into
# our own bucket so a clean `uv sync` + `download_assets.sh` never touches NVIDIA's CDN.
TABLE_USD = os.path.join(FIATLUX_ASSETS_DIR, "isaac_packing_table", "packing_table.usd")
# The warehouse USD is nested to match its original Nucleus depth (Isaac/Environments/
# Simple_Warehouse/warehouse.usd) because it references shared props two directories up
# (Isaac/Props/KLT_Bin/...) via a relative path -- flattening it breaks that reference.
WAREHOUSE_USD = os.path.join(
    FIATLUX_ASSETS_DIR, "isaac_warehouse", "Environments", "Simple_Warehouse", "warehouse.usd"
)
CLUTTER_BOX_USD = os.path.join(
    FIATLUX_ASSETS_DIR,
    "isaac_warehouse",
    "Environments",
    "Simple_Warehouse",
    "Props",
    "SM_CardBoxB_01_681.usd",
)
SKY_HDRI = os.path.join(FIATLUX_ASSETS_DIR, "isaac_skies", "kloofendal_43d_clear_puresky_4k.hdr")
