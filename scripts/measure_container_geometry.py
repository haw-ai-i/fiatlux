# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""How wide is the disposal crate inside, at each height? How big is the bench top? (issues #204, #207)

Both gates that read these are currently written against a SINGLE number, and both are wrong in
the same way -- a container is not a box of constant width:

* ``place_terms.CRATE_INTERIOR_HALF_EXTENT`` is one half-extent for the crate's whole height.
  Near the rim the real interior is wider, so a bulb resting against an inner wall up there
  reads as poking through the wall and ``old_bulb_in_bin`` fails a genuine disposal (#204).
* S08's lift gate tested the bulb's height against the tabletop plus a clearance, which only
  recognises a lift that goes UP -- a bulb held below the bench is off it too (#207). Replacing
  that with a footprint test needs the bench's real top surface and extent.

Method follows ``measure_bore_geometry.py``: pull each mesh's authored points, push them
through the prim's own local-to-world transform, and profile them by height slice. For the
crate the statistic per slice is the INNER FACE of each wall -- measured on the wall it
belongs to, which is the surface a bulb can rest against. For the bench it is the top surface height and the full
footprint.

Run with `uv run python` from the repo root, not a bare .venv/bin/python.

Example
-------
    uv run python scripts/measure_container_geometry.py --headless
"""

import argparse
import sys

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seed", type=int, default=3, help="Layout seed (the crate/bench move with it).")
parser.add_argument("--slice", type=float, default=0.01, help="Height slice thickness (m).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
simulation_app = AppLauncher(args_cli).app

import fiatlux_task.tasks  # noqa: E402, F401
import gymnasium as gym  # noqa: E402
from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import set_layout_seed  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def build_cfg():
    set_layout_seed(args_cli.seed)
    cfg = parse_env_cfg("FIATLUX-Replace-v0", device=args_cli.device, num_envs=1)
    cfg.seed = args_cli.seed
    for camera in ("ego_camera", "torso_camera", "wrist_camera"):
        if getattr(cfg.scene, camera, None) is not None:
            setattr(cfg.scene, camera, None)
    for group_name in ("policy", "privileged"):
        group = getattr(cfg.observations, group_name, None)
        for term in ("ego_rgb", "torso_rgb", "wrist_rgb"):
            if group is not None and getattr(group, term, None) is not None:
                setattr(group, term, None)
    return cfg


def all_meshes(stage, prefix: str) -> list[tuple[str, np.ndarray]]:
    """Every mesh under ``prefix``, as (path, world-space points)."""
    from pxr import Gf, UsdGeom

    out = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not (path.startswith(prefix) and prim.IsA(UsdGeom.Mesh)):
            continue
        points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if not points:
            continue
        xform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0)
        out.append((path, np.array([xform.Transform(Gf.Vec3d(*p)) for p in points], dtype=np.float64)))
    return out


def interior_profile(pts: np.ndarray, origin: np.ndarray, slice_m: float) -> list[tuple[float, float, float]]:
    """(height above origin, interior half-extent x, half-extent y) per slice.

    The interior half-extent at a height is the SMALLEST |x| (and |y|) reached by wall points in
    that slice -- the inner face. Taking the max would measure the crate's outer shell, which is
    what a bulb can never rest against.
    """
    rel = pts - origin
    rows = []
    z = rel[:, 2]
    lo, hi = float(z.min()), float(z.max())
    step = slice_m
    h = lo
    while h < hi:
        sel = rel[(z >= h) & (z < h + step)]
        if len(sel) >= 8:
            # Each wall measured on the wall it belongs to. Filtering both axes with one
            # combined `0.5 * edge` threshold and taking the minimum returned the THRESHOLD
            # (0.1504), not a wall, putting the inner x face at 0.155 when it is at 0.2873.
            x_wall = sel[np.abs(sel[:, 1]) < 0.6 * np.abs(sel[:, 1]).max()]
            y_wall = sel[np.abs(sel[:, 0]) < 0.6 * np.abs(sel[:, 0]).max()]
            if len(x_wall) and len(y_wall):
                x_shell = np.abs(x_wall[:, 0])
                y_shell = np.abs(y_wall[:, 1])
                inner_x = x_shell[x_shell > 0.8 * x_shell.max()]
                inner_y = y_shell[y_shell > 0.8 * y_shell.max()]
                if len(inner_x) and len(inner_y):
                    rows.append((h + step / 2.0, float(inner_x.min()), float(inner_y.min())))
        h += step
    return rows


def main() -> int:
    env = gym.make("FIATLUX-Replace-v0", cfg=build_cfg()).unwrapped
    env.reset()
    import omni.usd

    stage = omni.usd.get_context().get_stage()

    crate = env.scene["bin"]
    crate_origin = crate.data.root_pos_w[0].cpu().numpy()
    meshes = all_meshes(stage, "/World/envs/env_0/Bin")
    if not meshes:
        print("[measure] no crate meshes found under /World/envs/env_0/Bin")
        return 1
    pts = np.concatenate([p for _, p in meshes])
    print(f"\n[measure] crate: {len(meshes)} meshes, {len(pts)} points, origin {np.round(crate_origin, 4).tolist()}")
    print(f"{'height above origin':>20s} {'half_x':>9s} {'half_y':>9s}")
    rows = interior_profile(pts, crate_origin, args_cli.slice)
    for h, hx, hy in rows:
        print(f"{h:20.4f} {hx:9.4f} {hy:9.4f}")
    if rows:
        print(f"\n[measure] narrowest interior: x {min(r[1] for r in rows):.4f}  y {min(r[2] for r in rows):.4f}")
        print(f"[measure] widest interior:    x {max(r[1] for r in rows):.4f}  y {max(r[2] for r in rows):.4f}")
        print("[measure] CRATE_INTERIOR_HALF_EXTENT currently assumes one value for all heights")

    table_meshes = all_meshes(stage, "/World/envs/env_0/Table")
    if table_meshes:
        tp = np.concatenate([p for _, p in table_meshes])
        print(f"\n[measure] bench: {len(table_meshes)} meshes, {len(tp)} points")
        print(f"[measure] top surface z   {tp[:, 2].max():.4f}")
        print(f"[measure] x span          {tp[:, 0].min():.4f} .. {tp[:, 0].max():.4f}")
        print(f"[measure] y span          {tp[:, 1].min():.4f} .. {tp[:, 1].max():.4f}")
        print("[measure] per-mesh top z (the bench prim also carries the crates on it, #176):")
        for path, pm in sorted(table_meshes, key=lambda kv: -kv[1][:, 2].max())[:10]:
            print(f"    {pm[:, 2].max():8.4f}  {len(pm):7d} pts  {path.split('/Table/')[-1][:60]}")
        print(f"[measure] half extent     x {(tp[:, 0].max() - tp[:, 0].min()) / 2:.4f}  "
              f"y {(tp[:, 1].max() - tp[:, 1].min()) / 2:.4f}")
        print(f"[measure] centre          {((tp[:, 0].max() + tp[:, 0].min()) / 2):.4f}, "
              f"{((tp[:, 1].max() + tp[:, 1].min()) / 2):.4f}")

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
