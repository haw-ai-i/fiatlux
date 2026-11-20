"""A LadderMan ladder whose RAILS are addressable as contact targets.

Why this exists. LadderMan's `create_ladder_cfg` builds each rail as ONE long cuboid spanning the
whole ladder, so `object_pos_w[rail_idx]` is a single fixed point at the ladder's midpoint. The
contact machinery reads exactly that point (`command.py`: `object_pos_w[env_ids, obj_idx]`), so a
rail cannot be a hand target -- the target would never move as the robot climbs. Rungs work only
because there is one object per height.

That matters for this benchmark because our ladder is a STEP ladder. Measured off the asset's own
mesh by plane intersection (not vertex sampling -- a prismatic member has no vertices mid-span):

    climbing-side rail   40.0 x 74.5 mm, centres at +/-0.225 m, continuous from the floor to
                         z ~= 1.35 m, then NOTHING on the climbing face up to the 1.861 m top
    treads               4 of them, tops 0.236 / 0.470 / 0.706 / 0.940, plus a platform at 1.152

So the only handhold our ladder offers above the treads is the rail, and the previous reference
never addressed one: 51% of its 939 steps commanded a hand onto a rung at 1.409 or 1.644 m (empty
air) or onto the rear top cap at 1.878 m.

This builder keeps rungs for the feet -- they land on our treads to within 0.8 mm -- and adds
short rail SEGMENTS, one object per height, so a hand target rises with the climb.

Collection order (positional, which is what `object_contact` indexes):

    0 .. num_rungs-1              rungs (our treads, then the platform)
    num_rungs + 2k                left  rail segment k
    num_rungs + 2k + 1            right rail segment k

`create_ladder_cfg_maybe_railsegs` dispatches: with `rail_segments=0` it calls LadderMan's own
function unchanged, so the original behaviour is always one config key away.
"""
from __future__ import annotations

import math


def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def create_ladder_cfg_railsegs(
    prim_path: str = "{ENV_REGEX_NS}/Ladder",
    num_rungs: int = 5,
    rung_length: float = 0.4946,
    rung_width: float = 0.0738,
    rung_height: float = 0.080,
    rung_offset: float = 0.0,
    rung_mass: float = 0.5,
    base_position: tuple[float, float, float] = (-0.58, 0.0, 0.196),
    rung_spacing: float = 0.2492,
    inclination_deg: float = 70.3,
    align_rungs_to_incline: bool = False,
    contact_idx: int = 0,            # consumed by the tracking command, not here
    rail_segments: int = 6,
    rail_lateral: float = 0.040,     # MEASURED cross-section across the ladder's width
    rail_depth: float = 0.0745,      # MEASURED cross-section front-to-back
    rail_offset: float = 0.225,      # MEASURED centre offset from the ladder centreline
    rail_z_lo: float = 0.25,
    rail_z_hi: float = 1.35,         # MEASURED: the climbing-side rail ends here
    top_cap: bool = True,
    top_cap_z: float = 1.845,        # MEASURED apex height
    top_cap_lateral: float = 0.063,  # MEASURED across
    top_cap_depth: float = 0.107,    # MEASURED fore-aft
    top_cap_len: float = 0.090,      # along the ladder axis
    top_cap_offset: float = 0.220,   # MEASURED centre offset
):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg

    angle = math.radians(inclination_deg)
    direction = (math.cos(angle), 0.0, math.sin(angle))
    z_rot = (math.cos(-math.pi * 0.25), 0.0, 0.0, math.sin(-math.pi * 0.25))

    rung_rotation = (
        (math.cos(angle / 2.0), 0.0, math.sin(angle / 2.0), 0.0)
        if align_rungs_to_incline else (1.0, 0.0, 0.0, 0.0)
    )
    rung_rotation = _quat_mul(z_rot, rung_rotation)
    rail_rotation = _quat_mul(z_rot, (math.cos(angle / 2.0), 0.0, math.sin(angle / 2.0), 0.0))

    def cuboid(size, mass, pos, rot, path, colour):
        return RigidObjectCfg(
            prim_path=path,
            spawn=sim_utils.CuboidCfg(
                size=size,
                activate_contact_sensors=False,
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True, rigid_body_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=colour),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
        )

    objs = []
    for i in range(num_rungs):
        s = i * rung_spacing
        pos = (base_position[0] + direction[0] * s + i * rung_offset,
               base_position[1] + direction[1] * s,
               base_position[2] + direction[2] * s)
        objs.append(cuboid((rung_length, rung_width, rung_height), rung_mass, pos,
                           rung_rotation, f"{prim_path}_rung_{i}", (0.4, 0.2, 0.0)))

    # Rail segments, one object per height so a hand target climbs. Placed by WORLD z and then
    # projected back onto the ladder axis, because the measurements above are heights off the
    # floor, not distances along the incline.
    if rail_segments > 0:
        sin_a = math.sin(angle)
        s_lo = (rail_z_lo - base_position[2]) / sin_a
        s_hi = (rail_z_hi - base_position[2]) / sin_a
        seg_len = (s_hi - s_lo) / rail_segments
        for k in range(rail_segments):
            s = s_lo + (k + 0.5) * seg_len
            cx = base_position[0] + direction[0] * s
            cz = base_position[2] + direction[2] * s
            for sign in (-1.0, +1.0):            # left first, then right -> N+2k, N+2k+1
                objs.append(cuboid(
                    (seg_len, rail_lateral, rail_depth), rung_mass,
                    (cx, base_position[1] + sign * rail_offset, cz),
                    rail_rotation,
                    f"{prim_path}_rail_{'L' if sign < 0 else 'R'}_{k}",
                    (0.0, 0.35, 0.45)))

    # The TOP CAP, where the front rails and back legs meet at the apex. MEASURED: a pair of
    # members at x = +/-0.22, 60-66 mm x ~107 mm, spanning y +0.158..+0.265 around z 1.84-1.86.
    # It matters because extrapolating the tread line to z=1.845 gives fore-aft +0.158 -- the
    # cap's own near edge -- so it sits exactly where an 8th rung would be, and it is the only
    # real handhold above the rails. A person topping out a step ladder holds this.
    if top_cap:
        s_cap = (top_cap_z - base_position[2]) / math.sin(angle)
        cx = base_position[0] + direction[0] * s_cap
        cz = base_position[2] + direction[2] * s_cap
        for sign in (-1.0, +1.0):
            objs.append(cuboid(
                (top_cap_len, top_cap_lateral, top_cap_depth), rung_mass,
                (cx, base_position[1] + sign * top_cap_offset, cz),
                rail_rotation,
                f"{prim_path}_cap_{'L' if sign < 0 else 'R'}",
                (0.45, 0.2, 0.0)))

    return RigidObjectCollectionCfg(
        rigid_objects={f"rung_{i}": cfg for i, cfg in enumerate(objs)})


def create_ladder_cfg_maybe_railsegs(**kw):
    """Dispatch: `rail_segments` > 0 gets segmented rails, otherwise LadderMan's own builder."""
    if int(kw.get("rail_segments", 0) or 0) > 0:
        return create_ladder_cfg_railsegs(**kw)
    kw.pop("rail_segments", None)
    for k in ("rail_lateral", "rail_depth", "rail_offset", "rail_z_lo", "rail_z_hi",
              "top_cap", "top_cap_z", "top_cap_lateral", "top_cap_depth", "top_cap_len",
              "top_cap_offset"):
        kw.pop(k, None)
    from .ladder_env import create_ladder_cfg
    return create_ladder_cfg(**kw)


def describe(num_rungs: int, rail_segments: int, **kw) -> str:
    """The index map, so a clip author can see which label addresses what."""
    import math as _m
    angle = _m.radians(kw.get("inclination_deg", 70.3))
    base = kw.get("base_position", (-0.58, 0.0, 0.196))
    sp = kw.get("rung_spacing", 0.2492)
    rh = kw.get("rung_height", 0.080)
    lines = [f"{'idx':>4}  {'object':<16} {'top z':>8}"]
    for i in range(num_rungs):
        lines.append(f"{i:>4}  {'rung ' + str(i):<16} {base[2] + i*sp*_m.sin(angle) + rh/2:>8.3f}")
    lo, hi = kw.get("rail_z_lo", 0.25), kw.get("rail_z_hi", 1.35)
    s_lo = (lo - base[2]) / _m.sin(angle); s_hi = (hi - base[2]) / _m.sin(angle)
    seg = (s_hi - s_lo) / max(rail_segments, 1)
    for k in range(rail_segments):
        z = base[2] + (s_lo + (k + 0.5) * seg) * _m.sin(angle)
        lines.append(f"{num_rungs+2*k:>4}  {'rail L seg ' + str(k):<16} {z:>8.3f}")
        lines.append(f"{num_rungs+2*k+1:>4}  {'rail R seg ' + str(k):<16} {z:>8.3f}")
    if kw.get("top_cap", True):
        b = num_rungs + 2 * rail_segments
        lines.append(f"{b:>4}  {'top cap L':<16} {kw.get('top_cap_z', 1.845):>8.3f}")
        lines.append(f"{b+1:>4}  {'top cap R':<16} {kw.get('top_cap_z', 1.845):>8.3f}")
    return "\n".join(lines)
