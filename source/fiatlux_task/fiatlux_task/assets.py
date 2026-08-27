# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Central asset locations for the Fiatlux benchmark.

Assets are synced into the repo-root ``assets/`` dir by ``assets/download_assets.sh``.
Override the root with the ``FIATLUX_ASSETS_DIR`` env var if they live elsewhere.
"""

import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir, os.pardir))

FIATLUX_ASSETS_DIR = os.environ.get("FIATLUX_ASSETS_DIR", os.path.join(_REPO_ROOT, "assets"))

G1_USD = os.path.join(FIATLUX_ASSETS_DIR, "unitree_g1", "wholebody_inspire", "g1_29dof_with_inspire_rev_1_0.usd")
G1_DEX3_USD = os.path.join(FIATLUX_ASSETS_DIR, "unitree_g1", "wholebody_dex3", "g1_29dof_with_dex3_rev_1_0.usd")

# The socket's colliders are an exact triangle mesh (``physics:approximation = "none"``), which
# is illegal on a dynamic body: the socket must stay static or kinematic.
BULB_USD = os.path.join(FIATLUX_ASSETS_DIR, "omniverse_bulb", "LightBulb_bulb_z_rigid.usda")
SOCKET_USD = os.path.join(FIATLUX_ASSETS_DIR, "omniverse_bulb", "LightBulb_socket_z_static.usda")

# Curated Omniverse LightBulb split into a graspable bulb + its socket base (Y-up, cm-authored ->
# spawn scale + an X rotation). Clean-rendering (unlike the flat-texpath B1K lamp); used by the
# teleop bench (FIATLUX-Insert-Teleop-v0).
OMNI_BULB_USD = os.path.join(FIATLUX_ASSETS_DIR, "omniverse_bulb", "LightBulb_bulb.usda")
OMNI_SOCKET_USD = os.path.join(FIATLUX_ASSETS_DIR, "omniverse_bulb", "LightBulb_socket.usda")

# Metres, in each object's own root frame, Z-up. Both halves are authored assembled at
# identity, so "seated" is *bulb root pose == socket root pose*.
SOCKET_SEAT_OFFSET = (0.0, 0.0, 0.036259)
BULB_PLUG_OFFSET = (0.0, 0.0, 0.036259)
# Mating axis in each object's own root frame; rotation about it is free.
BULB_PLUG_AXIS = (0.0, 0.0, 1.0)
SOCKET_SEAT_AXIS = (0.0, 0.0, 1.0)
# A bulb standing on its cap has root z = surface_z - BULB_STAND_Z_OFFSET; lying on its glass,
# root z = surface_z + BULB_LIE_Z_OFFSET. The fixture's origin is its floor-contact plane.
BULB_STAND_Z_OFFSET = 0.036259
BULB_LIE_Z_OFFSET = 0.039561
SOCKET_BASE_Z_OFFSET = 0.0
# Cap bottom to fixture top: how far the bulb sinks when seated.
SOCKET_INSERTION_DEPTH = 0.034226

# AlumStep_D: 0.608 x 0.979 x 1.861 m, cm-authored -> spawn scale 0.01, base at z=0. The
# ``_collision`` variant carries the render meshes and the authored PhysX colliders; the plain
# variant has none.
STEP_LADDER_USD = os.path.join(
    FIATLUX_ASSETS_DIR,
    "omniverse_ladder",
    "AlumStep_D",
    "AluminumStepLadder_D01_PR_NVD_01_collision.usd",
)
# Sublayers the ``_collision`` variant and pre-applies a dynamic ``RigidBodyAPI`` + ``MassAPI``.
# Mass is stamped in code via ``LADDER_MASS_KG``, not read from the asset.
STEP_LADDER_RIGID_USD = os.path.join(
    FIATLUX_ASSETS_DIR,
    "omniverse_ladder",
    "AlumStep_D",
    "AluminumStepLadder_D01_PR_NVD_01_collision_rigid.usd",
)

# The standing point on the platform, in the ladder's base_link frame (metres, after the 0.01
# spawn scale). The tread is 48 x 40 cm, spanning x=[-0.24,0.24], y=[-0.10,0.30], topping out at
# z=1.18; steps ascend below it on local -y (z=0.47 / 0.71 / 0.94). The stance sits at y=0, in
# the strip in FRONT of the guardrail (which occupies y=[0.11,0.27] from z=1.46 up) -- centred
# on the tread instead, the rail passes through the robot's thighs.
#
# The tread exists in the render mesh but NOT in the convex-decomposed collider, so
# scripts/omniverse/omniverse_ladder_platform.py authors an explicit box collider for it into
# the ``_collision`` overlay. Without that box a robot placed here falls through the ladder.
STEP_LADDER_TOP_OFFSET = (0.0, 0.00, 1.18)

# Fingertip height above the surface the feet are on, arm raised straight up.
G1_OVERHEAD_REACH = 1.3738

# Horizontal fingertip offset from the pelvis, arm forward at rail height.
G1_HORIZONTAL_REACH = 0.5045

ELEVATED_SOCKET_USD = SOCKET_USD

# The ``_physics`` variant ships collision geometry.
CRATE_USD = os.path.join(
    FIATLUX_ASSETS_DIR,
    "isaac_packing_table",
    "props",
    "SM_Crate_A07_Yellow_01",
    "SM_Crate_A07_Yellow_01_physics.usd",
)

# Simple_Room (~11 materials), not Simple_Warehouse (~100+ MDL materials, each needing a
# single-threaded shader compile on first use). Nested to match its original Nucleus depth so
# relative references resolve.
TABLE_USD = os.path.join(FIATLUX_ASSETS_DIR, "isaac_packing_table", "packing_table.usd")
ROOM_USD = os.path.join(FIATLUX_ASSETS_DIR, "isaac_room", "Environments", "Simple_Room", "simple_room.usd")
SKY_HDRI = os.path.join(FIATLUX_ASSETS_DIR, "isaac_skies", "kloofendal_43d_clear_puresky_4k.hdr")
