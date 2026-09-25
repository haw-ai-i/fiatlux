# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""How deep is the socket bore, and how far into it does the plug reach? (issue #171)

The retention wrench's axial term is supposed to model a magnetic snap that acts only while
the plug is actually inside the bore, and its gain has to be sized against how far the plug can
withdraw before it is out. Both of those need a real measured engagement depth, not an
assumption -- ``assets/omniverse_bulb/CHANGES.md`` records the bore RADIUS (a constant 20.2 mm)
from an earlier point-cloud measurement, but never its depth.

This measures both, in the SEAT FRAME the wrench works in (axial = along the socket's seat
axis, zero at the seat; radial = distance from the seat axis line), off the running stage
rather than offline USD inspection, so it reports whatever Isaac Sim actually resolved.

Method: pull each mesh's authored points, push them through the prim's own
local-to-world transform, and project into the seat frame. For the socket the interesting
statistic per axial slice is the MINIMUM radius -- that is the inner wall of the bore, the
surface the plug rides against. For the plug it is the MAXIMUM radius, its widest point at that
height. The bore is open where its min radius is near 20.2 mm and closed where the slice fills
in toward the axis.

Run with `uv run python` from the repo root, not a bare .venv/bin/python -- see
verify_common.py's docstring for why.

Example
-------
    uv run python scripts/measure_bore_geometry.py --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Measure the socket bore depth and plug engagement in the seat frame.")
parser.add_argument("--seed", type=int, default=3, help="Env seed (only affects layout, not the geometry).")
parser.add_argument("--slice_mm", type=float, default=2.0, help="Axial bin width for the profile, in mm.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True if args_cli.headless is None else args_cli.headless

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Everything else follows."""

import importlib

import fiatlux_task.tasks  # noqa: F401  -- registers the FIATLUX Gym environments
import gymnasium as gym
import numpy as np
import torch
import verify_common
from fiatlux_task.assets import BULB_PLUG_OFFSET, SOCKET_SEAT_AXIS, SOCKET_SEAT_OFFSET
from fiatlux_task.tasks.manager_based.fiatlux_task.scene_cfg import set_layout_seed

from isaaclab.utils.math import quat_apply

from isaaclab_tasks.utils import parse_env_cfg


def build_cfg():
    set_layout_seed(args_cli.seed)
    cfg = parse_env_cfg("FIATLUX-Replace-v0", device=args_cli.device, num_envs=1)
    verify_common.assert_right_checkout(cfg, "old_bulb")
    cfg.seed = args_cli.seed
    verify_common.strip_visual_obs(cfg)
    return cfg


def profile(label: str, pts_seat: np.ndarray, slice_m: float, statistic: str) -> None:
    """Print a per-axial-slice radius profile: min radius for a bore, max for a plug."""
    axial, radial = pts_seat[:, 0], pts_seat[:, 1]
    lo, hi = axial.min(), axial.max()
    print(f"  {label}: {len(pts_seat)} points, axial extent {1000 * lo:+.1f} .. {1000 * hi:+.1f} mm", flush=True)
    print(f"    {'axial slice (mm)':>20s} {statistic + ' radius (mm)':>22s} {'n':>7s}", flush=True)
    edges = np.arange(lo, hi + slice_m, slice_m)
    for a, b in zip(edges, edges[1:]):
        sel = (axial >= a) & (axial < b)
        if sel.sum() == 0:
            continue
        r = radial[sel].min() if statistic == "min" else radial[sel].max()
        print(f"    {1000 * a:+8.1f}..{1000 * b:+8.1f} {1000 * r:22.2f} {int(sel.sum()):7d}", flush=True)


def main() -> int:
    cfg = build_cfg()
    spec = gym.spec("FIATLUX-Replace-v0")
    module_name, class_name = spec.entry_point.split(":")
    env = getattr(importlib.import_module(module_name), class_name)(cfg=cfg)
    env.reset(seed=args_cli.seed)

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    socket = env.scene["socket"]
    dev = env.device
    seat_axis = torch.tensor(SOCKET_SEAT_AXIS, device=dev).unsqueeze(0)
    seat_off = torch.tensor(SOCKET_SEAT_OFFSET, device=dev).unsqueeze(0)

    axis_w = quat_apply(socket.data.root_quat_w, seat_axis)[0].cpu().numpy().astype(np.float64)
    axis_w /= np.linalg.norm(axis_w)
    seat_w = (
        (socket.data.root_pos_w + quat_apply(socket.data.root_quat_w, seat_off))[0].cpu().numpy().astype(np.float64)
    )

    def to_seat_frame(pts: np.ndarray) -> np.ndarray:
        """(axial, radial) per point: signed distance along the seat axis, and off-axis radius.

        Axial is positive OUTWARD (the withdrawal direction), matching attach.py's sign
        convention, so a plug fully home reads ~0 and a withdrawing one reads positive.
        """
        d = pts - seat_w
        axial = d @ axis_w
        radial = np.linalg.norm(d - axial[:, None] * axis_w[None, :], axis=1)
        return np.stack([axial, radial], axis=1)

    print(f"\n=== SEAT FRAME (socket pose from the live stage, seed {args_cli.seed}) ===", flush=True)
    print(f"  seat point (world) = {np.round(seat_w, 5)}", flush=True)
    print(f"  seat axis  (world) = {np.round(axis_w, 5)}  (axial > 0 is the withdrawal direction)", flush=True)

    slice_m = args_cli.slice_mm / 1000.0
    bore_r = 0.0202  # the authored bore radius, from assets/omniverse_bulb/CHANGES.md
    near_bore = 1.5 * bore_r  # "this slice has geometry close enough to the axis to be a bore"

    socket_path_prefix = socket.root_physx_view.prim_paths[0]
    bulb = env.scene["old_bulb"] if "old_bulb" in env.scene.keys() else env.scene["fresh_bulb"]
    bulb_path_prefix = bulb.root_physx_view.prim_paths[0]

    print("\n=== SOCKET MESHES: which one carries the bore? ===", flush=True)
    print(f"  under {socket_path_prefix}", flush=True)
    bore_pts = None
    bore_name = None
    # Enumerating rather than guessing one mesh name: the socket half of this asset is 8
    # separate base/switch collider meshes (assets/omniverse_bulb/CHANGES.md), and which of
    # them carries the cylindrical bore is exactly what needs finding rather than assuming.
    for path, pts in verify_common.all_meshes(stage, socket_path_prefix):
        seat_pts = to_seat_frame(pts)
        inner = seat_pts[seat_pts[:, 1] < near_bore]
        # The bore is simply whichever mesh has the most surface near the seat axis. Testing the
        # max near-axis radius against the authored 20.2 mm was too strict: the same mesh also
        # carries the cavity above the bore and the mount cap on the axis, so its near-axis
        # radii span 0 to ~29 mm and only a BAND of it is the cylinder. Find the mesh here; find
        # the band from its profile below.
        tag = ""
        if len(inner) and (bore_pts is None or len(inner) > len(bore_pts)):
            bore_pts, bore_name, tag = inner, path, "  <-- most near-axis surface"
        span = (
            f"axial {1000 * inner[:, 0].min():+7.1f}..{1000 * inner[:, 0].max():+7.1f} mm  "
            f"radius {1000 * inner[:, 1].min():5.2f}..{1000 * inner[:, 1].max():5.2f} mm"
            if len(inner)
            else "no points within 1.5x the bore radius of the axis"
        )
        print(f"  {path.split('/')[-1]:16s} {len(pts):5d} pts, {len(inner):5d} near axis: {span}{tag}", flush=True)
    if bore_pts is None:
        print("  FATAL no socket mesh has any surface near the seat axis", flush=True)
        return 1

    print(f"\n=== BORE PROFILE ({bore_name.split('/')[-1]}, min radius per slice) ===", flush=True)
    profile("bore", bore_pts, slice_m, "min")

    # The cylindrical bore is the band of slices whose min radius sits at the authored radius:
    # inside it the plug is guided, outside it the mesh has opened into the lamp cavity (large
    # radius) or closed onto the mount cap (radius -> 0). Take the widest CONTIGUOUS such band,
    # so a stray slice elsewhere in the mesh cannot stretch the answer.
    axial, radial = bore_pts[:, 0], bore_pts[:, 1]
    edges = np.arange(axial.min(), axial.max() + slice_m, slice_m)
    cylindrical = []
    for a, b in zip(edges, edges[1:]):
        sel = (axial >= a) & (axial < b)
        cylindrical.append(bool(sel.sum()) and abs(radial[sel].min() - bore_r) < 0.0015)
    best_lo = best_hi = run_lo = None
    for i, is_cyl in enumerate([*cylindrical, False]):
        if is_cyl and run_lo is None:
            run_lo = i
        elif not is_cyl and run_lo is not None:
            if best_lo is None or (i - run_lo) > (best_hi - best_lo):
                best_lo, best_hi = run_lo, i
            run_lo = None
    if best_lo is None:
        print(f"  FATAL no contiguous band at the authored {1000 * bore_r:.1f} mm radius", flush=True)
        return 1
    bore_lo_m, bore_hi_m = edges[best_lo], edges[best_hi]
    print(
        f"  cylindrical band at r={1000 * bore_r:.1f} mm: axial {1000 * bore_lo_m:+.1f} .. "
        f"{1000 * bore_hi_m:+.1f} mm  ({1000 * (bore_hi_m - bore_lo_m):.1f} mm long)",
        flush=True,
    )
    bore_pts = bore_pts[(axial >= bore_lo_m) & (axial <= bore_hi_m)]

    print("\n=== PLUG PROFILE (max radius per slice, placed fully home) ===", flush=True)
    plug_meshes = [
        (p, pts) for p, pts in verify_common.all_meshes(stage, bulb_path_prefix) if p.endswith("BulbGrp/Base")
    ]
    if not plug_meshes:
        print(f"  FATAL no BulbGrp/Base mesh under {bulb_path_prefix}", flush=True)
        return 1
    plug_path, plug_pts = plug_meshes[0]
    # Re-express the plug's points relative to ITS OWN plug reference point, then put that point
    # at the seat -- so the profile describes a fully-home plug wherever the bulb happens to have
    # spawned. Must come from the SAME bulb as the mesh, or the two disagree by metres.
    plug_ref = (
        (
            bulb.data.root_pos_w
            + quat_apply(bulb.data.root_quat_w, torch.tensor(BULB_PLUG_OFFSET, device=dev).unsqueeze(0))
        )[0]
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    plug_seat = to_seat_frame(plug_pts - plug_ref + seat_w)
    print(f"  prim: {plug_path}", flush=True)
    profile("plug", plug_seat, slice_m, "max")

    # Engagement: the overlap of the two axial spans is how much plug is inside how much bore
    # when fully home, and therefore how far it can withdraw before it is out.
    bore_lo, bore_hi = bore_pts[:, 0].min(), bore_pts[:, 0].max()
    plug_lo, plug_hi = plug_seat[:, 0].min(), plug_seat[:, 0].max()
    overlap = max(0.0, min(bore_hi, plug_hi) - max(bore_lo, plug_lo))
    print("\n=== ENGAGEMENT (all in the seat frame; axial > 0 is withdrawal) ===", flush=True)
    print(f"  bore spans   axial {1000 * bore_lo:+.1f} .. {1000 * bore_hi:+.1f} mm", flush=True)
    print(f"  plug spans   axial {1000 * plug_lo:+.1f} .. {1000 * plug_hi:+.1f} mm  (fully home)", flush=True)
    print(f"  overlap      {1000 * overlap:.1f} mm of plug inside bore when home", flush=True)
    print(
        f"  => withdrawing {1000 * overlap:.1f} mm takes the plug out of the bore entirely. A snap "
        "force\n     that is supposed to act only while inside the bore must be gone by then.",
        flush=True,
    )
    env.close()
    return 0


if __name__ == "__main__":
    verify_common.run_verify_main(main, simulation_app)
