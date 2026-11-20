"""The benchmark's STEP ladder, as LadderMan kinematic cuboids.

Deployed to LadderMan as ``active_adaptation/envs/ladder_fiatlux.py``; ``ladder_env.py`` calls
``create_ladder_cfg_dispatch`` instead of ``create_ladder_cfg`` (a one-line change -- with no
``kind`` in ``ladder_config`` it falls straight through to LadderMan's own builders).

Why. LadderMan's ladder is rungs on two rails. Ours is a step ladder: four treads, then a
PLATFORM, then a tool-tray HANDRAIL 0.7 m above it, and the reference clip puts the hands on
exactly those -- tread 4, the platform's front edge, the handrail's front rim. A rung ladder has
none of the last two, so a teacher trained on one learns to lean on air.

Geometry, measured off the benchmark ladder's collision mesh in the clip's frame (x toward the
ladder, y left, z up; the same frame LadderMan's ``base_position`` is in):

    treads     top z 0.2397 / 0.4742 / 0.7092 / 0.9441, 68 mm deep, each 85.7 mm further in
    platform   top z 1.1517, x -0.3036 .. +0.0762
    handrail   a tray: floor z 1.8419, front rim x -0.036..-0.028 up to 1.8609 with a chamfered
               nose reaching x -0.042, back rim x 0.060..0.070
    stringers  40 x 76 mm at y +-0.225, 70.1 deg, floor to the handrail
    rear legs  26 x 73 mm at y +-0.225, 80.0 deg the other way, floor to z 1.5

Object order (positional -- it is what ``object_contact`` and ``rung_id`` index):

    0..3    treads 1-4          (0 is "rung 0": the ladder observation reads it)
    4       platform
    5       handrail tray
    6,7     handrail front rim, lower and upper course (the chamfer, in two steps)
    8       handrail back rim
    9,10    stringers            11,12   rear legs
    13      a parked dummy       <- contact label 0 ("no contact") lands here, masked
    14..19  hand ANCHORS for contact labels 1..6: no collision, they only mark where a hand
            that the clip says is holding should be. Use ``contact_idx=13``.

Why anchors. LadderMan's contact target is ``object_pos + a per-hand offset`` -- one offset for
every object. That works when every hold is a rung centre. Here the hand rests on a different
part of each object (an edge, a rim), so no single offset fits; an anchor per hold does. They
are read from the clip's own ``anchors.json`` so the targets cannot drift from the clip.
"""
from __future__ import annotations

import json
import math

TREAD_TOPS = (0.2397, 0.4742, 0.7092, 0.9441)
TREAD_X4 = (-0.3650, -0.2973)          # tread 4's top face; each lower tread is 85.7 mm nearer
TREAD_DX = 0.0857
TREAD_THICK = 0.0356
HALF_W = 0.200                          # treads, platform, tray between the stringers
PLATFORM = dict(x=(-0.3036, 0.0762), top=1.1517, thick=0.0428)
TRAY = dict(x=(-0.0421, 0.0701), floor=1.8419, bottom=1.8105, rim=1.8609, half_w=0.247)
N_PHYS = 13


def _q_yaw(deg):
    h = math.radians(deg) / 2.0
    return (math.cos(h), 0.0, 0.0, math.sin(h))


def _q_pitch(deg):
    h = math.radians(deg) / 2.0
    return (math.cos(h), 0.0, math.sin(h), 0.0)


def create_fiatlux_step_ladder_cfg(prim_path: str = "{ENV_REGEX_NS}/Ladder", anchors_file: str | None = None,
                                   anchors=None, mass: float = 2.0, **_unused):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg

    def cuboid(size, pos, rot, path, collide=True, color=(0.10, 0.10, 0.10)):
        return RigidObjectCfg(
            prim_path=path,
            spawn=sim_utils.CuboidCfg(
                size=size,
                activate_contact_sensors=False,
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, rigid_body_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=collide),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
        )

    def slab(x0, x1, half_w, top, thick):
        """A level box given by its x range, half width and top height. Rotated -90 deg about z
        like a LadderMan rung (local x = the long, across-the-ladder axis), so the heading the
        ladder observation reads off object 0 means what it means there."""
        return ((2 * half_w, x1 - x0, thick), (0.5 * (x0 + x1), 0.0, top - 0.5 * thick), _q_yaw(-90.0))

    objs = []
    for k, top in enumerate(TREAD_TOPS):
        dx = -(3 - k) * TREAD_DX
        size, pos, rot = slab(TREAD_X4[0] + dx, TREAD_X4[1] + dx, HALF_W, top, TREAD_THICK)
        objs.append(cuboid(size, pos, rot, f"{prim_path}_rung_{len(objs)}"))
    size, pos, rot = slab(*PLATFORM["x"], HALF_W, PLATFORM["top"], PLATFORM["thick"])
    objs.append(cuboid(size, pos, rot, f"{prim_path}_rung_{len(objs)}"))
    t = TRAY
    for x0, x1, top, bottom in ((t["x"][0], t["x"][1], t["floor"], t["bottom"]),       # tray body
                                (-0.0395, -0.0280, 1.8510, t["floor"]),                # rim, lower course
                                (-0.0358, -0.0280, t["rim"], 1.8510),                  # rim, upper course
                                (0.0600, t["x"][1], t["rim"], t["floor"])):            # back rim
        size, pos, rot = slab(x0, x1, t["half_w"], top, top - bottom)
        objs.append(cuboid(size, pos, rot, f"{prim_path}_rung_{len(objs)}", color=(0.55, 0.05, 0.05)))
    # stringers: front edge x = -0.598 at z 0.30, rising 0.3623 m in x per m of z; 81 mm wide in x
    ang, zc = 70.08, 0.915
    xc = -0.598 + (zc - 0.30) * 0.3623 + 0.0405
    length = 1.83 / math.sin(math.radians(ang))
    for sy in (1.0, -1.0):
        objs.append(cuboid((length, 0.040, 0.0762), (xc, 0.225 * sy, zc), _q_pitch(-ang),
                           f"{prim_path}_rung_{len(objs)}"))
    # rear legs: front edge x = 0.165 at z 0.30, falling 0.1755 m in x per m of z; 74 mm wide
    ang, zc = 80.03, 0.75
    xc = 0.165 - (zc - 0.30) * 0.1755 + 0.037
    length = 1.50 / math.sin(math.radians(ang))
    for sy in (1.0, -1.0):
        objs.append(cuboid((length, 0.026, 0.0729), (xc, 0.225 * sy, zc), _q_pitch(-(180.0 - ang)),
                           f"{prim_path}_rung_{len(objs)}"))
    assert len(objs) == N_PHYS, len(objs)
    objs.append(cuboid((0.02, 0.02, 0.02), (0.0, 0.0, -5.0), (1.0, 0.0, 0.0, 0.0),
                       f"{prim_path}_rung_{len(objs)}", collide=False))               # label 0 parks here
    if anchors is None:
        if anchors_file is None:
            raise ValueError("fiatlux_step ladder needs anchors_file (the clip's anchors.json)")
        import pathlib
        f = pathlib.Path(anchors_file)
        if not f.is_absolute():                            # hydra changes cwd: resolve against the repo
            f = pathlib.Path(__file__).resolve().parents[2] / f
        anchors = json.loads(f.read_text())["anchors"]
    assert len(anchors) == 6, "one anchor per contact label 1..6"
    for xyz in anchors:
        objs.append(cuboid((0.01, 0.01, 0.01), tuple(float(v) for v in xyz), (1.0, 0.0, 0.0, 0.0),
                           f"{prim_path}_rung_{len(objs)}", collide=False, color=(0.0, 0.8, 0.0)))
    return RigidObjectCollectionCfg(rigid_objects={f"rung_{i}": c for i, c in enumerate(objs)})


def create_ladder_cfg_dispatch(**ladder_config):
    """``kind: fiatlux_step`` -> this file's ladder; anything else -> LadderMan's, unchanged."""
    kind = ladder_config.pop("kind", None)
    if kind == "fiatlux_step":
        return create_fiatlux_step_ladder_cfg(**ladder_config)
    try:
        from .ladder_rails import create_ladder_cfg_maybe_railsegs as _stock
    except ImportError:
        from .ladder_env import create_ladder_cfg as _stock
    return _stock(**ladder_config)
