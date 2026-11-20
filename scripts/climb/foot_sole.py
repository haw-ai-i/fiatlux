"""Give the simulated G1 foot the flat sole the real foot has.

The robot USD collides each foot with four 5 mm spheres at the sole's corners (x -0.05 / +0.12,
y +-0.03) and NOTHING under the mid-foot. On flat ground that is a fine stand-in. On a ladder it
is not: a tread here is 76 mm deep and the pads are 170 mm apart, so a foot centred on a tread
has no collider on it at all -- the tread passes between the heel and toe pads and the robot
drops through its own footing. The real foot has a continuous rubber sole there (the visual mesh
is flat at z -0.035 over x -0.066..+0.142, y +-0.038).

This adds that missing surface: one thin box per foot, inside the visual sole's outline and flush
with the bottoms of the existing spheres, so standing on flat ground is unchanged. It adds
contact; it filters or disables none. The link's authored mass and inertia are untouched.

Applied from the climb scripts by wrapping the robot's spawn function, so the benchmark's own
robot config is not edited; moving it into ``robots/g1.py`` is the team-level version of this.
"""
from __future__ import annotations

SOLE_X = (-0.060, 0.135)
SOLE_HALF_Y = 0.033
SOLE_Z = (-0.035, -0.025)


def add_foot_sole(cfg) -> None:
    """Wrap ``cfg.scene.robot.spawn.func`` so both ankle_roll links get a sole collider."""
    from isaaclab.sim.utils import clone

    orig = cfg.scene.robot.spawn.func
    inner = getattr(orig, "__wrapped__", orig)

    @clone
    def _spawn(prim_path, spawn_cfg, translation=None, orientation=None):
        from pxr import Gf, UsdGeom, UsdPhysics

        prim = inner(prim_path, spawn_cfg, translation, orientation)
        stage = prim.GetStage()
        for side in ("left", "right"):
            link = stage.GetPrimAtPath(f"{prim_path}/{side}_ankle_roll_link")
            if not link.IsValid():
                raise RuntimeError(f"no {side}_ankle_roll_link under {prim_path}")
            cube = UsdGeom.Cube.Define(stage, f"{link.GetPath()}/sole_collider")
            cube.CreateSizeAttr(1.0)
            xf = UsdGeom.Xformable(cube.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(0.5 * (SOLE_X[0] + SOLE_X[1]), 0.0, 0.5 * (SOLE_Z[0] + SOLE_Z[1])))
            xf.AddScaleOp().Set(Gf.Vec3f(SOLE_X[1] - SOLE_X[0], 2 * SOLE_HALF_Y, SOLE_Z[1] - SOLE_Z[0]))
            cube.CreatePurposeAttr(UsdGeom.Tokens.guide)        # collide, do not render
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        return prim

    cfg.scene.robot.spawn = cfg.scene.robot.spawn.replace(func=_spawn)
