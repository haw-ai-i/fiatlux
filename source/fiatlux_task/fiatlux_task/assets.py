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

FIATLUX_ASSETS_DIR = os.environ.get("FIATLUX_ASSETS_DIR", os.path.join(_REPO_ROOT, "assets"))

# Unitree's official pre-assembled G1 USDs (mirrored into the GCS bucket from
# unitreerobotics/unitree_sim_isaaclab_usds). The legged (wholebody) Inspire-hand
# variant is the default; the Dex3-hand variant is staged for an easy future swap.
G1_USD = os.path.join(FIATLUX_ASSETS_DIR, "unitree_g1", "wholebody_inspire", "g1_29dof_with_inspire_rev_1_0.usd")
G1_DEX3_USD = os.path.join(FIATLUX_ASSETS_DIR, "unitree_g1", "wholebody_dex3", "g1_29dof_with_dex3_rev_1_0.usd")

# Bulb + fixture: the two halves of the Omniverse Sample-Scenes ``LightBulb``, the only
# TASK-READY bulb-in-a-socket in any Omniverse pack. It is not the only *separable* one --
# a prim-level survey of all 14 packs found five geometric pairs (see
# ``assets/omniverse_uploaded_manifest.csv``; ``chandelier_A`` even has an open ~1 mm
# clearance bore) -- but every ArchVis lamp ships without colliders, and this is the pair
# whose socket has usable ones. The socket's colliders are an
# exact triangle mesh (``physics:approximation = "none"``) so the screw hole stays OPEN -- a
# bulb lowered in nests and rests. Keep it: a convex hull closes the hole, and an exact mesh
# is illegal on a dynamic body, so the socket can never be anything but static/kinematic.
# ``scripts/omniverse/omniverse_bulb_rigid.py`` authors these two spawn-ready wrappers from
# Yujin Chen's ``_bulb``/``_socket`` layers, converting Y-up centimetres to Z-up metres so the
# env cfgs spawn them with an identity rotation.
BULB_USD = os.path.join(FIATLUX_ASSETS_DIR, "omniverse_bulb", "LightBulb_bulb_z_rigid.usda")
SOCKET_USD = os.path.join(FIATLUX_ASSETS_DIR, "omniverse_bulb", "LightBulb_socket_z_static.usda")

# Curated Omniverse LightBulb split into a graspable bulb + its socket base (Y-up, cm-authored ->
# spawn scale + an X rotation). Clean-rendering (unlike the flat-texpath B1K lamp); used by the
# teleop bench (FIATLUX-Insert-Teleop-v0).
OMNI_BULB_USD = os.path.join(FIATLUX_ASSETS_DIR, "omniverse_bulb", "LightBulb_bulb.usda")
OMNI_SOCKET_USD = os.path.join(FIATLUX_ASSETS_DIR, "omniverse_bulb", "LightBulb_socket.usda")

# Ladder for the climb-family tasks: ``shfvtl`` is the tall upright BEHAVIOR-1K
# ladder (bbox_z=1.67 m) marked "primary ladder for G1 to climb" in
# ``assets/behavior1k_uploaded_manifest.csv``. Synced with the other task assets
# by ``download_assets.sh``; 97 more Omniverse ladders are staged for variety.
LADDER_USD = os.path.join(FIATLUX_ASSETS_DIR, "behavior1k_ladder", "shfvtl", "usd", "shfvtl.usd")

# Mating geometry (metres, in each object's own root frame, Z-up). Both halves are authored
# ASSEMBLED AT IDENTITY, so the mating point -- the screw cap's bottom rim -- is the SAME
# point in both frames, and "seated" is simply *bulb root pose == socket root pose*.
# Measured off the source geometry (cap bottom y=3.6259 cm, glass radius 3.9561 cm, fixture
# top y=7.0485 cm); ``verify_scene`` re-measures these at runtime so they cannot drift.
SOCKET_SEAT_OFFSET = (0.0, 0.0, 0.036259)
BULB_PLUG_OFFSET = (0.0, 0.0, 0.036259)
# Mating axis in each object's own root frame (out of the hole / cap -> glass). Seating is
# scored on alignment about THIS axis, so rotation about it -- the screwing motion -- is free.
BULB_PLUG_AXIS = (0.0, 0.0, 1.0)
SOCKET_SEAT_AXIS = (0.0, 0.0, 1.0)
# Placement helpers: a bulb standing on its cap has root z = surface_z - BULB_STAND_Z_OFFSET;
# lying on its glass, root z = surface_z + BULB_LIE_Z_OFFSET. The fixture's origin IS its
# floor-contact plane, so it rests on a surface at exactly the surface height.
BULB_STAND_Z_OFFSET = 0.036259
BULB_LIE_Z_OFFSET = 0.039561
SOCKET_BASE_Z_OFFSET = 0.0
# Fixture insertion depth (cap bottom to fixture top) -- how far the bulb sinks when seated.
SOCKET_INSERTION_DEPTH = 0.034226

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
# The same ladder's ``_collision_rigid`` overlay: it sublayers the ``_collision`` variant
# (render meshes + colliders) and pre-applies a single DYNAMIC ``RigidBodyAPI`` + ``MassAPI``
# (authored mass ~7.3 kg). Used by FIATLUX-Carry-v0 so the graspable ladder does not need the
# rigid body stamped in code -- the spawner only tunes props + binds the grip material.
STEP_LADDER_RIGID_USD = os.path.join(
    FIATLUX_ASSETS_DIR,
    "omniverse_ladder",
    "HeavyDutyFRPStep_A",
    "HeavyDutyFiberglassStepLadder_A01_PR_NVD_01_collision_rigid.usd",
)

# The step ladder's top-platform point in its base_link frame (metres, after the 0.01 spawn
# scale; the asset is 1.75 m tall with its base authored at z=0, the top step sits slightly
# below the rail top). Used by the replace task's ladder-progress scoring.
STEP_LADDER_TOP_OFFSET = (0.0, 0.0, 1.70)

# Fingertip height above whatever surface the feet are on, arm raised straight up. MEASURED
# by FK (right shoulder pitch at its -3.089 rad limit, all other arm joints 0): pelvis 0.790,
# highest finger body 1.3738. Bounds how high a fixture may be mounted and still be worked on.
G1_OVERHEAD_REACH = 1.3738

# Elevated fixture for the at-height presets: the SAME socket half as the bench tasks, just
# mounted inverted. One socket everywhere means one set of mating constants and one seating
# rule; a decorative chandelier has no matching socket and cannot be seated into at all.
ELEVATED_SOCKET_USD = SOCKET_USD

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
ROOM_USD = os.path.join(FIATLUX_ASSETS_DIR, "isaac_room", "Environments", "Simple_Room", "simple_room.usd")
SKY_HDRI = os.path.join(FIATLUX_ASSETS_DIR, "isaac_skies", "kloofendal_43d_clear_puresky_4k.hdr")
