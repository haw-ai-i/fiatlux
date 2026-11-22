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


# ---------------------------------------------------------------------------------------------
# FREE-STANDING ladder (``ladder_config.free: true``)
#
# The benchmark's ladder is not fixed: it is one free rigid body of 10.9 kg standing on the floor
# (fiatlux ``LADDER_MASS_KG``; friction 1.5 / 1.2; centre of mass and inertia as the simulator
# reports them for that asset, measured 2026-11-21: com (0, -0.012, 0.826) in the ladder's own
# frame = (-0.2116, 0, 0.826) here; inertia about the com 3.57 / 3.927 / 1.318 kg m^2 about the
# climb, cross and vertical axes). A teacher trained on fixed boxes has never had a hold or a
# tread move under load, and loses most benchmark episodes in the mount.
#
# How. LadderMan reads the ladder as a COLLECTION of objects (``env.ladder.data.object_pos_w``):
# object 0 for the ladder observation, the anchor objects for the hand targets. That interface is
# kept. With ``free`` the collection's boxes lose their collision and become markers; ONE dynamic
# body carrying the same thirteen boxes as its colliders (``{ENV_REGEX_NS}/LadderBody``) is what
# the robot touches, and after every physics step each marker is written to the body's pose
# composed with its own offset (``install_free_ladder``). The stringers and rear legs stop 45 mm
# short of the floor and stand on four flat feet, so the body rests where the fixed ladder was
# drawn instead of starting 13 mm inside the ground on the tilted boxes' corners.
# ---------------------------------------------------------------------------------------------
FREE_MASS_KG = 10.9
FREE_COM = (-0.2116, 0.0, 0.826)
FREE_INERTIA = (3.57, 3.927, 1.318)
FREE_FRICTION = (1.5, 1.2)
FOOT = (0.10, 0.040, 0.030)             # x, y, z of each flat foot
LEG_BOTTOM = 0.045                      # the tilted members' axes stop here


def _members(z_bottom: float):
    """(size, pos, rot) of the two stringers and two rear legs, with their axes starting at z_bottom."""
    out = []
    for ang, x_at_030, slope, half_x, thick, z_top in ((70.08, -0.598, 0.3623, 0.0405, (0.040, 0.0762), 1.83),
                                                      (80.03, 0.165, -0.1755, 0.037, (0.026, 0.0729), 1.50)):
        zc = 0.5 * (z_top + z_bottom)
        xc = x_at_030 + (zc - 0.30) * slope + half_x
        length = (z_top - z_bottom) / math.sin(math.radians(ang))
        pitch = -ang if slope > 0 else -(180.0 - ang)
        for sy in (1.0, -1.0):
            out.append(((length, thick[0], thick[1]), (xc, 0.225 * sy, zc), _q_pitch(pitch)))
    return out


def _feet():
    out = []
    for x_at_030, slope, half_x in ((-0.598, 0.3623, 0.0405), (0.165, -0.1755, 0.037)):
        xc = x_at_030 + (0.5 * FOOT[2] - 0.30) * slope + half_x
        for sy in (1.0, -1.0):
            out.append((FOOT, (xc, 0.225 * sy, 0.5 * FOOT[2]), (1.0, 0.0, 0.0, 0.0)))
    return out


def free_body_boxes():
    """Every collider of the free ladder as (size, pos, rot) in the ladder's own (= the clip's) frame."""
    boxes = []
    for k, top in enumerate(TREAD_TOPS):
        dx = -(3 - k) * TREAD_DX
        x0, x1 = TREAD_X4[0] + dx, TREAD_X4[1] + dx
        boxes.append(((2 * HALF_W, x1 - x0, TREAD_THICK), (0.5 * (x0 + x1), 0.0, top - 0.5 * TREAD_THICK), _q_yaw(-90.0)))
    x0, x1 = PLATFORM["x"]
    boxes.append(((2 * HALF_W, x1 - x0, PLATFORM["thick"]), (0.5 * (x0 + x1), 0.0, PLATFORM["top"] - 0.5 * PLATFORM["thick"]), _q_yaw(-90.0)))
    t = TRAY
    for x0, x1, top, bottom in ((t["x"][0], t["x"][1], t["floor"], t["bottom"]), (-0.0395, -0.0280, 1.8510, t["floor"]),
                                (-0.0358, -0.0280, t["rim"], 1.8510), (0.0600, t["x"][1], t["rim"], t["floor"])):
        boxes.append(((2 * t["half_w"], x1 - x0, top - bottom), (0.5 * (x0 + x1), 0.0, top - 0.5 * (top - bottom)), _q_yaw(-90.0)))
    return boxes + _members(LEG_BOTTOM) + _feet()


def _spawn_free_ladder(prim_path, cfg, translation=None, orientation=None, **kwargs):
    import isaacsim.core.utils.prims as prim_utils
    from isaaclab.sim import schemas
    from isaaclab.sim.utils import bind_physics_material
    from pxr import Gf, UsdPhysics

    prim = prim_utils.create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation)
    mat_path = f"{prim_path}/physicsMaterial"
    cfg.physics_material.func(mat_path, cfg.physics_material)
    for i, (size, pos, rot) in enumerate(cfg.boxes):
        path = f"{prim_path}/box_{i}"
        prim_utils.create_prim(path, "Cube", translation=pos, orientation=rot, scale=tuple(0.5 * v for v in size),
                               attributes={"size": 2.0})
        schemas.define_collision_properties(path, cfg.collision_props)
        bind_physics_material(path, mat_path)
    schemas.define_mass_properties(prim_path, cfg.mass_props)
    schemas.define_rigid_body_properties(prim_path, cfg.rigid_props)
    m = UsdPhysics.MassAPI(prim)
    m.CreateCenterOfMassAttr().Set(Gf.Vec3f(*cfg.com))
    m.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*cfg.inertia))
    m.CreatePrincipalAxesAttr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    return prim


def create_fiatlux_free_ladder_body_cfg(prim_path: str = "{ENV_REGEX_NS}/LadderBody"):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass

    @configclass
    class _FreeLadderSpawnCfg(sim_utils.RigidObjectSpawnerCfg):
        func: callable = clone(_spawn_free_ladder)
        boxes: list = None
        com: tuple = FREE_COM
        inertia: tuple = FREE_INERTIA
        physics_material: sim_utils.RigidBodyMaterialCfg = None

    spawn = _FreeLadderSpawnCfg(
        boxes=free_body_boxes(),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=FREE_FRICTION[0], dynamic_friction=FREE_FRICTION[1], restitution=0.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=FREE_MASS_KG),
        # as the benchmark sets them on its ladder (scene_cfg.apply_position_preset)
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False, rigid_body_enabled=True,
                                                     solver_position_iteration_count=16, solver_velocity_iteration_count=1,
                                                     max_depenetration_velocity=1.0, stabilization_threshold=0.001),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        activate_contact_sensors=False,
    )
    return RigidObjectCfg(prim_path=prim_path, spawn=spawn, init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)))


def install_free_ladder(env) -> None:
    """Make the collection's markers follow the free body, and put the body back at every reset."""
    import torch
    from isaaclab.utils.math import quat_apply, quat_mul

    body, coll = env.scene.rigid_objects["ladder_body"], env.ladder
    n = coll.num_objects
    # each marker's pose in the body's frame = where it was authored (the body starts at the env origin, unrotated)
    loc_p = (coll.data.default_object_state[:, :, :3]).clone()                  # [N, n, 3], env-local
    loc_q = (coll.data.default_object_state[:, :, 3:7]).clone()
    keep = torch.zeros(n, dtype=torch.bool, device=env.device); keep[N_PHYS] = True      # the parked dummy stays parked

    def sync(_substep=None):
        bp, bq = body.data.root_link_pos_w, body.data.root_link_quat_w                  # [N, 3], [N, 4]
        q = quat_mul(bq[:, None].expand(-1, n, -1).reshape(-1, 4), loc_q.reshape(-1, 4)).reshape(-1, n, 4)
        p = bp[:, None] + quat_apply(bq[:, None].expand(-1, n, -1).reshape(-1, 4), loc_p.reshape(-1, 3)).reshape(-1, n, 3)
        p[:, keep] = env.scene.env_origins[:, None] + loc_p[:, keep]
        coll.write_object_link_pose_to_sim(torch.cat([p, q], dim=-1))

    def reset(env_ids):
        st = body.data.default_root_state[env_ids].clone()
        st[:, :3] += env.scene.env_origins[env_ids]
        st[:, 7:] = 0.0
        body.write_root_state_to_sim(st, env_ids=env_ids)
        sync()

    env._reset_callbacks.insert(0, reset)
    env._post_step_callbacks.append(sync)
    env.ladder_body = body
    print(f"[ladder] FREE-STANDING: one {FREE_MASS_KG} kg body with {len(free_body_boxes())} box colliders; "
          f"{n} markers follow it after every physics step")


def create_fiatlux_step_ladder_cfg(prim_path: str = "{ENV_REGEX_NS}/Ladder", anchors_file: str | None = None,
                                   anchors=None, mass: float = 2.0, free: bool = False, **_unused):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg

    def cuboid(size, pos, rot, path, collide=True, color=(0.10, 0.10, 0.10)):
        collide = collide and not free                 # free: the boxes are markers, the body collides
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
