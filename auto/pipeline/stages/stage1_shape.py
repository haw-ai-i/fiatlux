#!/usr/bin/env python3
"""Stage 1: single image -> textured mesh, via Hunyuan3D-2.1.

Runs inside ``.venvs/hunyuan``.  Must be launched with cwd = the Hunyuan3D-2.1
checkout: several config paths inside the paint pipeline are resolved relative
to the process working directory.

Modelled on the repo's ``model_worker.py`` rather than its ``demo.py``.  demo.py
passes ``output_mesh_path='demo_textured.glb'``, but the writer underneath is
``save_obj_mesh`` -- so it writes Wavefront OBJ text into a file named ``.glb``
and then asks Blender to import that as glTF.  model_worker.py takes the .obj
path and converts explicitly, which is what we do.

Two upstream quirks are worked around here rather than in the vendored repo:

* **Background removal never runs upstream.** Both demo.py:25-28 and
  model_worker.py:165-167 do ``image.convert("RGBA")`` and *then* test
  ``if image.mode == "RGB"`` -- a branch that can no longer be true. For a JPEG
  like ``ladder.jpg`` the alpha channel is therefore all-255, and the recentring
  preprocessor treats the whole frame as foreground, which wrecks the shape.
  ``gradio_app.py:289-291`` gets this right; we follow it and call rembg
  explicitly.
* **Two imports fail silently.** ``MeshRender.py`` wraps both the bpy IO import
  and the mesh_inpaint_processor import in bare ``try/except`` that only print.
  A missing build then surfaces much later as a ``NameError`` inside texture
  baking. We assert them up front instead.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time

REPO = os.getcwd()
# The repo root itself carries torchvision_fix.py and makes `hy3dpaint` importable
# as a namespace package; the two subdirs are needed because hy3dshape lives at
# hy3dshape/hy3dshape/, and hy3dpaint's internals use top-level imports
# (`from textureGenPipeline import ...`, `from hunyuanpaintpbr.unet...`).
# Python puts the *script's* directory on sys.path, not the CWD, so add all three.
sys.path.insert(0, os.path.join(REPO, "hy3dshape"))
sys.path.insert(0, os.path.join(REPO, "hy3dpaint"))
sys.path.insert(0, REPO)


@contextlib.contextmanager
def chdir(path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def preflight(texture: bool):
    """Fail loudly now rather than confusingly later."""
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("ERROR: CUDA is not available in the Hunyuan environment")
    print(f"  torch {torch.__version__}  device={torch.cuda.get_device_name(0)}")
    if not texture:
        return
    missing = []
    try:
        import custom_rasterizer  # noqa: F401
    except Exception as exc:
        missing.append(f"custom_rasterizer ({exc})")
    try:
        from DifferentiableRenderer import mesh_inpaint_processor  # noqa: F401
    except Exception as exc:
        missing.append(f"mesh_inpaint_processor ({exc})")
    try:
        import bpy  # noqa: F401
    except Exception as exc:
        missing.append(f"bpy ({exc})")
    if missing:
        raise SystemExit(
            "ERROR: the texture stage needs these, and they fail silently upstream:\n    "
            + "\n    ".join(missing)
            + "\n  Re-run setup/02_env_hunyuan.sh, or pass --no-texture."
        )
    print("  custom_rasterizer / mesh_inpaint_processor / bpy: OK")


def load_pipelines(args, texture):
    """Load rembg + shape (+ paint) once; batch runs amortise this ~2-3 min."""
    from hy3dshape.rembg import BackgroundRemover
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    print("  loading shape pipeline ...")
    shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        args.model_path, subfolder="hunyuan3d-dit-v2-1",
        use_safetensors=False, variant="fp16", device="cuda")
    paint = None
    if texture:
        from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

        print("  loading paint pipeline ...")
        conf = Hunyuan3DPaintConfig(args.paint_views, args.paint_resolution)  # positional; no defaults
        conf.realesrgan_ckpt_path = os.path.join(REPO, "hy3dpaint/ckpt/RealESRGAN_x4plus.pth")
        conf.multiview_cfg_path = os.path.join(REPO, "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml")
        conf.custom_pipeline = os.path.join(REPO, "hy3dpaint/hunyuanpaintpbr")
        paint = Hunyuan3DPaintPipeline(conf)
    return BackgroundRemover(), shape, paint


def process_one(image_path, name, out, rembg, shape, paint, args):
    """One image -> textured mesh, into `out`.  Returns the stage1 result dict."""
    import torch
    from PIL import Image

    t0 = time.time()
    os.makedirs(out, exist_ok=True)
    print(f"\n[stage1] {image_path} -> {out}")
    image = Image.open(image_path).convert("RGB")
    print(f"  input {image.size}")
    # Unconditional, unlike upstream -- see the module docstring.
    image = rembg(image)
    rgba_path = os.path.join(out, f"{name}_rgba.png")
    image.save(rgba_path)
    print(f"  background removed -> {os.path.basename(rgba_path)}")

    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    mesh = shape(image=image, num_inference_steps=args.steps,
                 guidance_scale=args.guidance, generator=gen,
                 octree_resolution=args.octree_resolution,
                 num_chunks=args.num_chunks, mc_algo="mc",   # 'dmc' would need the extra `diso` dep
                 output_type="trimesh")[0]
    print(f"  raw mesh: {len(mesh.vertices)} verts / {len(mesh.faces)} faces")

    from hy3dshape.postprocessors import FaceReducer, FloaterRemover, DegenerateFaceRemover
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh, max_facenum=args.max_faces)
    print(f"  cleaned:  {len(mesh.vertices)} verts / {len(mesh.faces)} faces")

    white_glb = os.path.join(out, f"{name}_white.glb")
    mesh.export(white_glb)
    result = {"white_glb": white_glb, "rgba": rgba_path, "textured": paint is not None}

    if paint is not None:
        from hy3dpaint.convert_utils import create_glb_with_pbr_materials

        # MUST end in .obj: the writer is save_obj_mesh regardless of extension,
        # and save_glb=True only does a `.replace(".obj", ".glb")` afterwards.
        obj_path = os.path.join(out, f"{name}.obj")
        obj_path = paint(mesh_path=white_glb, image_path=image,
                         output_mesh_path=obj_path, use_remesh=True,
                         save_glb=False)          # we convert ourselves, without Blender
        print(f"  painted -> {os.path.basename(obj_path)}")

        stem = obj_path[:-4]
        textures = {"albedo": stem + ".jpg",
                    "metallic": stem + "_metallic.jpg",
                    "roughness": stem + "_roughness.jpg"}
        if os.path.isfile(stem + "_normal.jpg"):
            textures["normal"] = stem + "_normal.jpg"
        textures = {k: v for k, v in textures.items() if os.path.isfile(v)}

        glb_path = os.path.join(out, f"{name}.glb")
        # create_glb_with_pbr_materials writes temp.glb and mr_combined.png into
        # the CWD; run it from the output dir so it cannot litter the vendored repo.
        with chdir(out):
            create_glb_with_pbr_materials(obj_path, dict(textures), glb_path)
        for junk in ("temp.glb", "mr_combined.png"):
            jp = os.path.join(out, junk)
            if os.path.exists(jp):
                os.remove(jp)

        # Canonical keys only: `mesh` (the glb later stages consume) and
        # `textured_obj` (the .obj whose stem locates the PBR maps).
        result.update(textures=textures, mesh=glb_path, textured_obj=obj_path)
        print(f"  glb     -> {os.path.basename(glb_path)}  "
              f"({len(textures)} PBR map(s): {', '.join(textures)})")
    else:
        result.update(mesh=white_glb, textures={})

    result["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(out, "stage1.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"[stage1] done in {result['seconds']}s -> {result['mesh']}")
    return result


def main():
    ap = argparse.ArgumentParser(description="Hunyuan3D-2.1: image -> textured mesh")
    ap.add_argument("--image", help="input image (single mode)")
    ap.add_argument("--out-dir", help="output directory (single mode)")
    ap.add_argument("--name", default="object")
    ap.add_argument("--batch", help="JSON file: [{image, name, out_dir}, ...] -- one "
                                    "process, models loaded once for all items")
    ap.add_argument("--no-texture", dest="texture", action="store_false", default=True)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--octree-resolution", type=int, default=384)
    ap.add_argument("--num-chunks", type=int, default=8000)
    ap.add_argument("--max-faces", type=int, default=40000)
    ap.add_argument("--paint-views", type=int, default=6)
    ap.add_argument("--paint-resolution", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--model-path", default="tencent/Hunyuan3D-2.1")
    args = ap.parse_args()

    if args.batch:
        with open(args.batch) as fh:
            items = json.load(fh)
    else:
        if not args.image or not args.out_dir:
            ap.error("--image and --out-dir are required (or use --batch)")
        items = [{"image": args.image, "name": args.name, "out_dir": args.out_dir}]

    preflight(args.texture)
    if args.texture and args.max_faces > 40000:
        # hy3dpaint's remesh step hardcodes target_count=40000
        # (hy3dpaint/utils/simplify_mesh_utils.py:23) and decimates anything
        # above it, so a larger --max-faces silently comes back out at 40k.
        print(f"  [WARN] --max-faces {args.max_faces} > 40000: the paint stage's "
              "internal remesh will clip the textured mesh back to 40000 faces")

    # basicsr (a RealESRGAN dep) imports torchvision.transforms.functional_tensor,
    # removed in torchvision >= 0.17. This shim must land before that import.
    from torchvision_fix import apply_fix
    apply_fix()

    rembg, shape, paint = load_pipelines(args, args.texture)
    for item in items:
        process_one(item["image"], item["name"], os.path.abspath(item["out_dir"]),
                    rembg, shape, paint, args)


if __name__ == "__main__":
    main()
