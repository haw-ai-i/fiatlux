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
# ``kfmkwd`` (bulb) and ``ehjsdz`` (lamp) are BEHAVIOR's canonical ``changing_light_bulbs``
# mating pair: the bulb carries a ``bulblampM`` male-plug metalink that seats into the
# lamp's ``bulblampF`` female-socket metalink (manifest ``has_socket=True``, usage
# ``task_asset``). The earlier ``ymomhw``/``bbentu`` pair had no attachment metalinks --
# ``bbentu`` is a closed decorative dressing lamp with no socket, which made the seated
# pose physically unattainable (issue #29). The env spawns each as a single rigid body by
# deactivating its ``meta__*`` helper links (``spawn_b1k_single_body``), so the metalink
# transforms below are baked as constants rather than read at runtime.
BULB_USD = os.path.join(FIATLUX_ASSETS_DIR, "behavior1k_bulb", "kfmkwd", "kfmkwd.usd")
SOCKET_USD = os.path.join(FIATLUX_ASSETS_DIR, "behavior1k_lamp", "ehjsdz", "ehjsdz.usd")

# Curated Omniverse LightBulb split into a graspable bulb + its socket base (Y-up, cm-authored ->
# spawn scale + an X rotation). Clean-rendering (unlike the flat-texpath B1K lamp); used by the
# teleop bench (FIATLUX-Insert-Teleop-v0).
OMNI_BULB_USD = os.path.join(FIATLUX_ASSETS_DIR, "omniverse_bulb", "LightBulb_bulb.usda")
OMNI_SOCKET_USD = os.path.join(FIATLUX_ASSETS_DIR, "omniverse_bulb", "LightBulb_socket.usda")

# Attachment-metalink offsets (metres, in each object's base_link frame; both assets are
# Z-up, metersPerUnit=1.0). The bulb is "seated" when the bulb's plug point coincides with
# the lamp's socket point. Read once from the ``meta__*_attachment_*_joint`` localPos0 in
# the source USDs (see issue #29 fix).
SOCKET_SEAT_OFFSET = (0.0, 0.0, 0.0326)  # bulblampF, relative to the lamp base_link origin
BULB_PLUG_OFFSET = (0.0635, 0.0, -0.0225)  # bulblampM, relative to the bulb base_link origin

# Ladder for the climb-family tasks: ``shfvtl`` is the tall upright BEHAVIOR-1K
# ladder (bbox_z=1.67 m) marked "primary ladder for G1 to climb" in
# ``assets/behavior1k_uploaded_manifest.csv``. Synced with the other task assets
# by ``download_assets.sh``; 97 more Omniverse ladders are staged for variety.
LADDER_USD = os.path.join(FIATLUX_ASSETS_DIR, "behavior1k_ladder", "shfvtl", "usd", "shfvtl.usd")

# Work-site step ladder: a deployed, free-standing A-frame from the Omniverse SimReady
# pack (probe: 0.68 x 1.11 x 1.75 m deployed, cm-authored -> spawn scale 0.01, base at
# z=0). The ``_collision`` variant carries BOTH the visible render meshes and the authored
# PhysX colliders; the plain variant has no colliders. 27 more designs are local for
# variety/domain randomization (see assets/omniverse_ladder/).
STEP_LADDER_USD = os.path.join(
    FIATLUX_ASSETS_DIR,
    "omniverse_ladder",
    "HeavyDutyFRPStep_A",
    "HeavyDutyFiberglassStepLadder_A01_PR_NVD_01_collision.usd",
)

# The step ladder's top-platform point in its base_link frame (metres, after the 0.01 spawn
# scale; the asset is 1.75 m tall with its base authored at z=0, the top step sits slightly
# below the rail top). Used by the replace task's ladder-progress scoring.
STEP_LADDER_TOP_OFFSET = (0.0, 0.0, 1.70)

# Elevated fixture for the at-height task presets (climb/descend/remove/install): a
# BEHAVIOR-1K ceiling chandelier standing in as the socket the ladder leads to. One
# deterministic model (``qghfol``); 3 more chandelier ids are local for variety.
ELEVATED_SOCKET_USD = os.path.join(
    FIATLUX_ASSETS_DIR, "behavior1k_chandelier", "qghfol", "usd", "qghfol.usd"
)

# Parts crate (the Install preset's bulb bin), from the packing-table prop set; the
# ``_physics`` variant ships collision geometry so the bulb can rest inside it.
CRATE_USD = os.path.join(
    FIATLUX_ASSETS_DIR,
    "isaac_packing_table",
    "props",
    "SM_Crate_A07_Yellow_01",
    "SM_Crate_A07_Yellow_01_physics.usd",
)

# Room dressing: table, room backdrop, and an HDRI sky, mirrored once from Isaac
# Sim's own Nucleus content library (Isaac/Props, Isaac/Environments) into our own
# bucket so a clean `uv sync` + `download_assets.sh` never touches NVIDIA's CDN.
TABLE_USD = os.path.join(FIATLUX_ASSETS_DIR, "isaac_packing_table", "packing_table.usd")
# Simple_Room, not the much heavier Simple_Warehouse: the warehouse's ~100+ unique
# MDL materials each need a one-time (single-threaded) shader compile on first use,
# which made a single video render take hours. Simple_Room has ~11 materials and
# still gives real walls/floor/windows instead of a bare plane. Nested to match its
# original Nucleus depth in case of relative references, same precaution as before.
ROOM_USD = os.path.join(
    FIATLUX_ASSETS_DIR, "isaac_room", "Environments", "Simple_Room", "simple_room.usd"
)
SKY_HDRI = os.path.join(FIATLUX_ASSETS_DIR, "isaac_skies", "kloofendal_43d_clear_puresky_4k.hdr")
