#!/usr/bin/env python3
"""Isaac Sim playground for the BEHAVIOR-1K lighting assets.

Lays out every behavior1k light / lamp / bulb in a grid so you can eyeball their materials
(useful for spotting the red-fallback / missing-texture problems tracked in issue 18).

The assets bind OmniGibson's ``OmniGibsonVRayMtl`` material, so its mdls must be on the MDL
search path or everything renders red. Point at that folder with ``--mdl-path`` or the
``FIATLUX_B1K_MDL`` env var; it defaults to ``assets/behavior1k_materials/`` (once the three
OmniGibson mdls are bundled there). Requires a camera-capable Isaac build (env_isaaclab).

Usage:
  python scripts/behavior1k_playground.py                       # GUI window on $DISPLAY
  python scripts/behavior1k_playground.py --render overview.png # headless overview capture
  python scripts/behavior1k_playground.py --mdl-path <dir>      # folder holding the vray mdls
"""

import argparse
import glob
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSETS = os.path.join(REPO, "assets")
GROUPS = [
    "behavior1k_bulb",
    "behavior1k_bulb_broken",
    "behavior1k_lamp",
    "behavior1k_floor_lamp",
    "behavior1k_chandelier",
    "behavior1k_lampshade",
    "behavior1k_paper_lantern",
    "behavior1k_downlight",
    "behavior1k_spotlight",
    "behavior1k_square_light",
    "behavior1k_rectangular_light",
    "behavior1k_track_light",
    "behavior1k_wall_mounted_light",
    "behavior1k_room_light",
]
COLS = 14
SP = 1.5


def collect(assets):
    """One primary USD per model folder across every lighting group."""
    items = []
    for g in GROUPS:
        for m in sorted(glob.glob(os.path.join(assets, g, "*"))):
            if not os.path.isdir(m):
                continue
            name = os.path.basename(m)
            cands = [
                f for f in glob.glob(os.path.join(m, "**", f"{name}.usd"), recursive=True) if "_collision" not in f
            ]
            if cands:
                items.append(cands[0])
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--render", metavar="PNG", default=None, help="headless: capture an overview PNG and exit")
    ap.add_argument(
        "--mdl-path",
        default=os.environ.get("FIATLUX_B1K_MDL", os.path.join(ASSETS, "behavior1k_materials")),
        help="folder holding omnigibson_vray_mtl.mdl (+ vray_maps/vray_materials)",
    )
    args = ap.parse_args()

    # Must be set before the renderer/MDL system starts.
    if os.path.isdir(args.mdl_path):
        os.environ["MDL_USER_PATH"] = os.path.abspath(args.mdl_path)
        print(f"[MDL_USER_PATH = {os.environ['MDL_USER_PATH']}]", flush=True)
    else:
        print(
            f"[warn] mdl path not found: {args.mdl_path} -> assets will render RED "
            "until the OmniGibson vray mdls are on the path",
            flush=True,
        )

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": args.render is not None})

    import omni.usd  # noqa: E402
    from pxr import Gf, Usd, UsdGeom, UsdLux  # noqa: E402

    items = collect(ASSETS)
    print(f"[{len(items)} behavior1k lighting assets]", flush=True)

    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(1200)
    UsdLux.DistantLight.Define(stage, "/World/Sun").CreateIntensityAttr(3000)
    g = UsdGeom.Cube.Define(stage, "/World/Ground")
    g.CreateSizeAttr(1.0)
    g.AddXformOp(UsdGeom.XformOp.TypeTranslate).Set(Gf.Vec3d(0, 0, -0.05))
    g.AddXformOp(UsdGeom.XformOp.TypeScale).Set(Gf.Vec3f(120, 120, 0.1))
    g.CreateDisplayColorAttr([(0.35, 0.37, 0.4)])

    for i, f in enumerate(items):
        mpu = UsdGeom.GetStageMetersPerUnit(Usd.Stage.Open(f))
        slot = UsdGeom.Xform.Define(stage, f"/World/a_{i}")
        slot.AddXformOp(UsdGeom.XformOp.TypeTranslate).Set(Gf.Vec3d((i % COLS) * SP, (i // COLS) * SP, 0))
        slot.AddXformOp(UsdGeom.XformOp.TypeScale).Set(Gf.Vec3f(mpu, mpu, mpu))
        UsdGeom.Xform.Define(stage, f"/World/a_{i}/r").GetPrim().GetReferences().AddReference(f)

    nrows = (len(items) + COLS - 1) // COLS
    cx, cy = (COLS - 1) * SP / 2.0, (nrows - 1) * SP / 2.0
    gd = (nrows - 1) * SP

    if args.render:
        import numpy as np

        import omni.replicator.core as rep

        cam = rep.create.camera(position=(cx, cy - (gd + 12), gd * 0.95 + 6), look_at=(cx, cy, 0.2))
        rp = rep.create.render_product(cam, (2600, 1500))
        annot = rep.AnnotatorRegistry.get_annotator("rgb")
        annot.attach(rp)
        for _ in range(150):
            app.update()
        for _ in range(8):
            rep.orchestrator.step(rt_subframes=16)
            app.update()
        import imageio.v2 as imageio

        imageio.imwrite(args.render, np.asarray(annot.get_data())[..., :3].astype(np.uint8))
        print(f"[saved -> {args.render}]", flush=True)
    else:
        try:
            from isaacsim.core.utils.viewports import set_camera_view

            set_camera_view(eye=[cx, cy - (gd + 10), gd * 0.9 + 6], target=[cx, cy, 0.2])
        except Exception as e:
            print(f"[camera preset skipped: {e}; press F to frame]", flush=True)
        print("[GUI open -- orbit=left-drag, zoom=scroll, F=frame. Close window to quit.]", flush=True)
        while app.is_running():
            app.update()
    app.close()


if __name__ == "__main__":
    main()
