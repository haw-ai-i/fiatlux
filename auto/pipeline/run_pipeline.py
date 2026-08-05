#!/usr/bin/env python3
"""image -> mesh -> articulation -> USD, end to end.

    python3 run_pipeline.py --image ../images/ladder.jpg --name ladder

Stdlib only, and run with the *system* python on purpose: its whole job is to
launch each stage under a different interpreter.  The three stages have
irreconcilable pins -- Hunyuan3D wants numpy 1.24 / bpy / CPython 3.11, SimArt
wants torch 2.4.0+cu121, and the USD writer wants nothing but usd-core -- so
they live in separate venvs under ``.venvs/`` and hand off through files.

    1  01_shape/       Hunyuan3D-2.1   image        -> textured mesh (.glb + .obj + PBR maps)
    2  02_normalized/  SimArt prep     mesh         -> centred, unit-extent .glb (+ preview renders)
    3  03_articulate/  SimArt          mesh         -> URDF + per-part .obj + prediction JSON
    4  04_usd/         urdf_to_usd     URDF+JSON    -> articulated USD (+ rigid fallback), verified

Every stage writes into its own directory, and ``--from``/``--to`` re-run any
slice against existing intermediates -- which matters, because a cold full run
is 1-2 hours and the stages fail for very different reasons.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
AUTO = os.path.dirname(HERE)
HUNYUAN = os.path.join(AUTO, "Hunyuan3D-2.1")
SIMART = os.path.join(AUTO, "SimArt")
VENV = os.path.join(HERE, ".venvs")
TOOLS = os.path.join(HERE, ".tools")
BLENDER = os.path.join(TOOLS, "blender-3.0.1-linux-x64", "blender")

PY_HUNYUAN = os.path.join(VENV, "hunyuan", "bin", "python")
PY_SIMART = os.path.join(VENV, "simart", "bin", "python")
PY_USD = os.path.join(VENV, "usd", "bin", "python")

STAGE_DIRS = {1: "01_shape", 2: "02_normalized", 3: "03_articulate", 4: "04_usd"}

# Stage-1 presets.  `high` mirrors what Hunyuan3D's own gradio_app ships
# (8 views / 768 paint resolution); `fast` trades fidelity for wall-clock.
QUALITY = {
    "fast":    {"steps": 30, "octree_resolution": 256, "paint_views": 6, "paint_resolution": 512},
    "default": {"steps": 50, "octree_resolution": 384, "paint_views": 6, "paint_resolution": 512},
    "high":    {"steps": 50, "octree_resolution": 512, "paint_views": 8, "paint_resolution": 768},
}

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


class Runner:
    def __init__(self, run_dir, name):
        self.run_dir = run_dir
        self.name = name
        self.log_path = os.path.join(run_dir, "pipeline.log")
        self.manifest_path = os.path.join(run_dir, "manifest.json")
        self.manifest = {}
        if os.path.isfile(self.manifest_path):
            with open(self.manifest_path) as fh:
                self.manifest = json.load(fh)

    def log(self, msg):
        line = f"{datetime.datetime.now():%H:%M:%S}  {msg}"
        print(line, flush=True)
        with open(self.log_path, "a") as fh:
            fh.write(line + "\n")

    def save(self):
        with open(self.manifest_path, "w") as fh:
            json.dump(self.manifest, fh, indent=2)

    def run(self, stage, argv, cwd, env=None, tail=40):
        """Run one stage, streaming into the log; abort the pipeline on failure."""
        e = dict(os.environ)
        # subprocess(cwd=...) changes the real working directory but leaves the
        # inherited PWD pointing at ours.  Blender resolves relative script paths
        # from PWD, so `utils/render_utils.py:70`'s relative
        # './blender_script/blender_render.py' would be looked up under this
        # directory instead of SimArt's -- and the failure is silent, because
        # Python's own os.path.exists() check on the same string passes.
        e["PWD"] = os.path.abspath(cwd)
        e.update(env or {})
        self.log(f"--- stage {stage} ---")
        self.log("    " + " ".join(argv))
        t0 = time.time()
        with open(self.log_path, "a") as fh:
            proc = subprocess.Popen(argv, cwd=cwd, env=e, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            lines = []
            for line in proc.stdout:
                fh.write(line)
                lines.append(line)
                sys.stdout.write(line)
                sys.stdout.flush()
            rc = proc.wait()
        dt = round(time.time() - t0, 1)
        if rc != 0:
            self.log(f"    stage {stage} FAILED (exit {rc}) after {dt}s")
            sys.stderr.write("\n".join(lines[-tail:]) + "\n")
            raise SystemExit(f"pipeline aborted in stage {stage}; see {self.log_path}")
        self.log(f"    stage {stage} ok in {dt}s")
        return dt


def git_sha(repo):
    try:
        return subprocess.check_output(["git", "-C", repo, "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def require(path, what):
    if not os.path.exists(path):
        raise SystemExit(f"ERROR: {what} not found at {path}\n"
                         f"       run the setup/ scripts first (see README.md)")
    return path


def ensure_repos(clone=True):
    """Clone the two upstream model repos before anything else runs.

    They are not vendored into this repo -- they carry their own .git and ~700 MB
    of history -- so a fresh checkout of the pipeline has nothing to drive until
    this runs.  setup/00_clone_repos.sh pins both to validated commits and
    applies the SimArt patch, and is idempotent, so calling it here costs a
    couple of `git rev-parse` calls once the repos exist.
    """
    missing = [p for p in (HUNYUAN, SIMART) if not os.path.isdir(os.path.join(p, ".git"))]
    if not missing:
        return
    names = ", ".join(os.path.basename(p) for p in missing)
    if not clone:
        raise SystemExit(f"ERROR: missing model repo(s): {names}\n"
                         f"       run setup/00_clone_repos.sh, or drop --no-clone")
    print(f"==> model repo(s) missing ({names}); cloning first")
    script = os.path.join(HERE, "setup", "00_clone_repos.sh")
    subprocess.check_call(["bash", script])
    still = [p for p in (HUNYUAN, SIMART) if not os.path.isdir(os.path.join(p, ".git"))]
    if still:
        raise SystemExit("ERROR: clone did not produce "
                         + ", ".join(os.path.basename(p) for p in still))


def link_blender_for_simart():
    """Point SimArt's hardcoded /tmp Blender path at the copy we already have.

    ``scripts/process_raw_objects.py:28-29`` hardcodes
    ``/tmp/blender-3.0.1-linux-x64/blender`` with no override, and downloads
    ~200 MB there if it is missing.  A symlink is cheaper than a patch.
    """
    if not os.path.isfile(BLENDER):
        return
    target = "/tmp/blender-3.0.1-linux-x64"
    if not os.path.exists(target):
        try:
            os.symlink(os.path.dirname(BLENDER), target)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser(
        description="image -> Hunyuan3D mesh -> SimArt articulation -> Isaac-Sim USD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--image", help="input image (required unless --from > 1)")
    ap.add_argument("--images", help="directory of images: batch mode -- stage 1 loads "
                                     "its models once for all of them, then stages 2-4 "
                                     "run per object under runs/<stem>/")
    ap.add_argument("--name", default=None, help="run name; defaults to the image stem")
    ap.add_argument("--runs-dir", default=os.path.join(HERE, "runs"))
    ap.add_argument("--from", dest="from_stage", type=int, default=1, choices=[1, 2, 3, 4])
    ap.add_argument("--to", dest="to_stage", type=int, default=4, choices=[1, 2, 3, 4])
    # stage 1
    ap.add_argument("--no-texture", dest="texture", action="store_false", default=True,
                    help="skip PBR texture painting (much faster, untextured USD)")
    ap.add_argument("--quality", default="default", choices=sorted(QUALITY),
                    help="stage-1 preset; individual flags below still override it")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--octree-resolution", type=int, default=None)
    ap.add_argument("--num-chunks", type=int, default=8000)
    ap.add_argument("--max-faces", type=int, default=40000)
    ap.add_argument("--model-path", default="tencent/Hunyuan3D-2.1")
    ap.add_argument("--paint-views", type=int, default=None)
    ap.add_argument("--paint-resolution", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1234)
    # stage 2
    # Zero is correct, but for a different reason than SimArt's README gives.
    # The README says assets must be "+Z up"; its own reference assets are not.
    # Measured: chair_00.glb and fridge_00.glb both have their height along Y
    # (glTF's Y-up convention), and infer.py:428 / mesh_utils.py:126 apply
    # R_x(+90) internally, which is exactly what turns Y-up into the Z-up frame
    # the model was trained in.  Hunyuan3D also exports Y-up glTF, so the two
    # already agree and no correction belongs here.  Always confirm against
    # 02_normalized/*_renders/ before trusting a new source.
    ap.add_argument("--rot-x", type=float, default=0.0)
    ap.add_argument("--rot-y", type=float, default=0.0)
    ap.add_argument("--rot-z", type=float, default=0.0)
    ap.add_argument("--no-preview", dest="preview", action="store_false", default=True,
                    help="skip the stage-2 orientation preview renders")
    # stage 3
    ap.add_argument("--attn", default="sdpa", help="SIMART_ATTN_IMPL (sdpa | flash_attention_2)")
    ap.add_argument("--max-new-tokens", type=int, default=24000)
    ap.add_argument("--attempts", type=int, default=1,
                    help="SimArt samples its predictions (temperature 0.7), so N > 1 "
                         "runs stage 3 N times with seeds seed..seed+N-1, scores each "
                         "(moving joints, usable parts) and keeps the best")
    # stage 4
    ap.add_argument("--scale-m", type=float, default=None)
    ap.add_argument("--scale-unit", default="auto")
    ap.add_argument("--limit-convention", default="spec", choices=["spec", "urdf"])
    ap.add_argument("--collision", default="convexDecomposition")
    ap.add_argument("--fixed-base", action="store_true")
    ap.add_argument("--total-mass", type=float, default=None,
                    help="pin the object's real mass in kg (the predicted per-part "
                         "density is often the weakest number SimArt returns)")
    ap.add_argument("--no-rigid", dest="rigid", action="store_false", default=True,
                    help="skip the single-body fallback USD")
    ap.add_argument("--rigid-only", action="store_true",
                    help="stage 1 then a rigid USD, skipping SimArt entirely")
    ap.add_argument("--expect-size", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                    help="plausible real-world extent in metres, checked by verify_usd.py")
    ap.add_argument("--no-clone", dest="clone", action="store_false", default=True,
                    help="fail instead of cloning the model repos if they are absent")
    ap.add_argument("--keep-intermediates", dest="clean", action="store_false", default=True,
                    help="keep the ~40 MB of per-run duplicates the stages leave behind")
    args = ap.parse_args()

    for k, v in QUALITY[args.quality].items():
        if getattr(args, k) is None:
            setattr(args, k, v)

    # Step zero, before any stage: the model repos must exist to be driven.
    ensure_repos(args.clone)

    if args.images:
        return run_batch(args)

    if not args.name:
        if not args.image:
            ap.error("--name is required when --image is omitted")
        args.name = os.path.splitext(os.path.basename(args.image))[0]

    run_dir = os.path.join(args.runs_dir, args.name)
    os.makedirs(run_dir, exist_ok=True)
    d = {k: os.path.join(run_dir, v) for k, v in STAGE_DIRS.items()}
    needed = {1, 4} if args.rigid_only else set(range(args.from_stage, args.to_stage + 1))
    for k in sorted(needed):
        os.makedirs(d[k], exist_ok=True)

    r = Runner(run_dir, args.name)
    r.manifest.setdefault("name", args.name)
    r.manifest["started"] = datetime.datetime.now().isoformat(timespec="seconds")
    r.manifest["repos"] = {"Hunyuan3D-2.1": git_sha(HUNYUAN), "SimArt": git_sha(SIMART)}
    r.manifest["args"] = vars(args)
    r.log(f"=== run '{args.name}'  stages {args.from_stage}..{args.to_stage} -> {run_dir}")

    # ---------------------------------------------------------------- stage 1
    if args.from_stage <= 1 <= args.to_stage:
        if not args.image:
            ap.error("--image is required for stage 1")
        require(PY_HUNYUAN, "the Hunyuan venv")
        src = os.path.join(run_dir, "00_input", os.path.basename(args.image))
        os.makedirs(os.path.dirname(src), exist_ok=True)
        if os.path.abspath(args.image) != src:
            shutil.copyfile(args.image, src)
        argv = [PY_HUNYUAN, os.path.join(HERE, "stages", "stage1_shape.py"),
                "--image", src, "--out-dir", d[1], "--name", args.name,
                "--steps", str(args.steps), "--guidance", str(args.guidance),
                "--octree-resolution", str(args.octree_resolution),
                "--num-chunks", str(args.num_chunks),
                "--max-faces", str(args.max_faces), "--paint-views", str(args.paint_views),
                "--paint-resolution", str(args.paint_resolution), "--seed", str(args.seed),
                "--model-path", args.model_path]
        if not args.texture:
            argv.append("--no-texture")
        # cwd MUST be the repo root: paint-pipeline config paths are relative to it.
        dt = r.run(1, argv, cwd=HUNYUAN, env={"PYTHONUNBUFFERED": "1"})
        with open(os.path.join(d[1], "stage1.json")) as fh:
            r.manifest["stage1"] = json.load(fh)
        r.manifest["stage1"]["seconds"] = dt
        if args.clean:
            cleanup_stage(1, d, args.name, r)
        r.save()

    s1 = r.manifest.get("stage1", {})
    mesh = s1.get("mesh")

    # ------------------------------------------------- rigid-only short circuit
    if args.rigid_only:
        require(PY_USD, "the USD venv")
        if not mesh:
            raise SystemExit("ERROR: --rigid-only needs stage 1 output (mesh missing from manifest)")
        out = os.path.join(d[4], f"{args.name}_rigid.usda")
        # rx90, not none: this path feeds Hunyuan's raw Y-up glTF/OBJ directly,
        # bypassing SimArt's internal R_x(+90) that justifies `none` elsewhere.
        argv = [PY_USD, os.path.join(HERE, "usd", "urdf_to_usd.py"), "--rigid",
                "--mesh", s1.get("textured_obj") or mesh, "--out", out,
                "--name", args.name, "--frame-rot", "rx90", "--collision", args.collision]
        if args.scale_m:
            argv += ["--scale-m", str(args.scale_m)]
        if args.total_mass:
            argv += ["--total-mass", str(args.total_mass)]
        dt = r.run(4, argv, cwd=HERE)
        verify(r, out, 1, 0, args)
        # setdefault-merge: a --rigid-only refresh must not erase the record of
        # an articulated USD produced by an earlier full run of the same name.
        r.manifest.setdefault("stage4", {}).update(rigid_usd=out, rigid_seconds=dt)
        r.manifest["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
        r.save()
        r.log(f"=== done -> {out}")
        return

    # ---------------------------------------------------------------- stage 2
    normalized = os.path.join(d[2], f"{args.name}.glb")
    if args.from_stage <= 2 <= args.to_stage:
        require(PY_SIMART, "the SimArt venv")
        if not mesh or not os.path.isfile(mesh):
            raise SystemExit("ERROR: stage 2 needs stage 1's mesh; run with --from 1")
        link_blender_for_simart()
        # process_raw_objects.py writes <output>/<input basename>, so feed it a
        # copy already named after the run.
        staged = os.path.join(d[2], "_input", f"{args.name}.glb")
        os.makedirs(os.path.dirname(staged), exist_ok=True)
        shutil.copyfile(mesh, staged)
        argv = [PY_SIMART, os.path.join(SIMART, "scripts", "process_raw_objects.py"),
                "--input", staged, "--output", d[2],
                "--rot_x", str(args.rot_x), "--rot_y", str(args.rot_y), "--rot_z", str(args.rot_z)]
        if args.preview:
            argv.append("--render")
        dt = r.run(2, argv, cwd=SIMART, env={"PYTHONUNBUFFERED": "1"})
        if not os.path.isfile(normalized):
            raise SystemExit(f"ERROR: stage 2 produced no {normalized}")
        r.manifest["stage2"] = {"normalized": normalized, "seconds": dt,
                                "preview_dir": os.path.join(d[2], f"{args.name}_renders")}
        if args.clean:
            cleanup_stage(2, d, args.name, r)
        r.save()

    # ---------------------------------------------------------------- stage 3
    urdf = os.path.join(d[3], f"{args.name}.urdf")
    pred = os.path.join(d[3], f"{args.name}.json")
    if args.from_stage <= 3 <= args.to_stage:
        require(PY_SIMART, "the SimArt venv")
        require(os.path.join(SIMART, "checkpoints", "simart_vqvae", "vq.pt"), "SimArt checkpoints")
        src = r.manifest.get("stage2", {}).get("normalized", normalized)
        if not os.path.isfile(src):
            raise SystemExit(f"ERROR: stage 3 needs {src}; run with --from 2")
        t3 = time.time()
        attempts = []
        for i in range(max(1, args.attempts)):
            adir = os.path.join(d[3], "attempts", f"a{i}")
            os.makedirs(adir, exist_ok=True)
            if i > 0:
                # infer.py skips its ~30 s Blender render when the PNG already
                # exists (infer.py:224), and the render is seed-independent.
                prev = os.path.join(d[3], "attempts", "a0", f"{args.name}_scaled.png")
                if os.path.isfile(prev):
                    shutil.copyfile(prev, os.path.join(adir, f"{args.name}_scaled.png"))
            argv = [PY_SIMART, os.path.join(SIMART, "inference", "infer.py"),
                    "--object_path", src, "--output_path", adir, "--name", args.name,
                    "--max_new_tokens", str(args.max_new_tokens)]
            if os.path.isfile(BLENDER):
                argv += ["--blender_path", BLENDER]
            # PYTHONPATH and cwd are both required and neither is set by the
            # repo: infer.py:29 imports `vqvae` with its sys.path fix commented
            # out, and utils/render_utils.py:70 hardcodes a CWD-relative blender
            # script path.  SIMART_SEED pins the sampler (patched in) so runs
            # are reproducible and attempts genuinely differ.
            r.run(3, argv, cwd=SIMART,
                  env={"PYTHONPATH": SIMART, "PYTHONUNBUFFERED": "1",
                       "SIMART_ATTN_IMPL": args.attn,
                       "SIMART_SEED": str(args.seed + i)})
            a_urdf = os.path.join(adir, f"{args.name}.urdf")
            if not os.path.isfile(a_urdf):
                raise SystemExit(f"ERROR: stage 3 attempt {i} produced no {a_urdf}")
            score = json.loads(subprocess.check_output(
                [PY_USD, os.path.join(HERE, "usd", "urdf_to_usd.py"),
                 "--score", "--urdf", a_urdf], text=True))
            score["seed"] = args.seed + i
            attempts.append(score)
            r.log(f"    attempt {i} (seed {score['seed']}): score {score['score']} "
                  f"({score.get('moving_joints', 0)} moving joints, "
                  f"{score.get('usable_links', 0)}/{score.get('links', 0)} parts)")

        best = max(range(len(attempts)), key=lambda i: attempts[i]["score"])
        bdir = os.path.join(d[3], "attempts", f"a{best}")
        for entry in (f"{args.name}.urdf", f"{args.name}.json", f"{args.name}_objs",
                      f"{args.name}_scaled.png", f"{args.name}_scaled.glb"):
            src_p, dst_p = os.path.join(bdir, entry), os.path.join(d[3], entry)
            if os.path.isdir(dst_p):
                shutil.rmtree(dst_p)
            elif os.path.isfile(dst_p):
                os.remove(dst_p)
            if os.path.exists(src_p):
                shutil.move(src_p, dst_p)
        if len(attempts) > 1:
            r.log(f"    kept attempt {best} of {len(attempts)}")
        r.manifest["stage3"] = {"urdf": urdf, "prediction": pred,
                                "seconds": round(time.time() - t3, 1),
                                "parts_dir": os.path.join(d[3], f"{args.name}_objs"),
                                "attempts": attempts, "chosen": best}
        if args.clean:
            cleanup_stage(3, d, args.name, r)
        r.save()

    # ---------------------------------------------------------------- stage 4
    if args.from_stage <= 4 <= args.to_stage:
        require(PY_USD, "the USD venv")
        if not os.path.isfile(urdf):
            raise SystemExit(f"ERROR: stage 4 needs {urdf}; run with --from 3")
        out = os.path.join(d[4], f"{args.name}.usda")
        argv = [PY_USD, os.path.join(HERE, "usd", "urdf_to_usd.py"),
                "--urdf", urdf, "--out", out, "--name", args.name,
                "--limit-convention", args.limit_convention, "--collision", args.collision,
                "--scale-unit", args.scale_unit]
        if os.path.isfile(pred):
            argv += ["--json", pred]
        if args.scale_m:
            argv += ["--scale-m", str(args.scale_m)]
        if args.fixed_base:
            argv.append("--fixed-base")
        if args.total_mass:
            argv += ["--total-mass", str(args.total_mass)]
        # Point at Hunyuan's own textured .obj: SimArt's per-part export runs
        # PBRMaterial.to_simple(), which drops the metallic/roughness maps.
        if s1.get("textured_obj"):
            argv += ["--textures", s1["textured_obj"]]
        dt = r.run(4, argv, cwd=HERE)
        r.manifest["stage4"] = {"usd": out, "seconds": dt}

        rep = {}
        if os.path.isfile(out + ".report.json"):
            with open(out + ".report.json") as fh:
                rep = json.load(fh)
        verify(r, out, len(rep.get("links", [])) or None,
               len(rep.get("joints", [])) or None, args)

        rigid_src = s1.get("textured_obj") or s1.get("white_glb")
        if args.rigid and rigid_src:
            rout = os.path.join(d[4], f"{args.name}_rigid.usda")
            # rx90, not none: the rigid path feeds Hunyuan's raw Y-up mesh
            # directly, bypassing SimArt's internal R_x(+90) that justifies
            # `none` on the articulated path above.
            rargv = [PY_USD, os.path.join(HERE, "usd", "urdf_to_usd.py"), "--rigid",
                     "--mesh", rigid_src, "--out", rout, "--name", f"{args.name}_rigid",
                     "--frame-rot", "rx90", "--collision", args.collision,
                     "--scale-m", str(args.scale_m or rep.get("scale_m") or 1.0)]
            if args.total_mass:
                rargv += ["--total-mass", str(args.total_mass)]
            r.run(4, rargv, cwd=HERE)
            verify(r, rout, 1, 0, args)
            check_bbox_agreement(r, out + ".report.json", rout + ".report.json")
            r.manifest["stage4"]["rigid_usd"] = rout
        r.save()
        r.log(f"=== done -> {out}")

    r.manifest["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
    r.save()


def run_batch(args):
    """--images <dir>: batched stage 1, then stages 2-4 per object.

    Stage 1 gets all images in ONE process (its ~2-3 min of model loading
    amortises across the batch); stages 2-4 then run per object by re-invoking
    this script with --name <stem> --from 2, which reuses every code path above
    verbatim.  Fails fast on the first broken object, like everything else here.
    """
    imgs = sorted(os.path.join(args.images, f) for f in os.listdir(args.images)
                  if f.lower().endswith(IMAGE_EXTS))
    if not imgs:
        raise SystemExit(f"ERROR: no images ({', '.join(IMAGE_EXTS)}) in {args.images}")
    names = [os.path.splitext(os.path.basename(i))[0] for i in imgs]
    if len(set(names)) != len(names):
        raise SystemExit("ERROR: duplicate image stems in the batch")
    print(f"==> batch of {len(imgs)}: {', '.join(names)}")

    if args.from_stage <= 1:
        require(PY_HUNYUAN, "the Hunyuan venv")
        items = []
        for img, name in zip(imgs, names):
            run_dir = os.path.join(args.runs_dir, name)
            src = os.path.join(run_dir, "00_input", os.path.basename(img))
            os.makedirs(os.path.dirname(src), exist_ok=True)
            shutil.copyfile(img, src)
            items.append({"image": src, "name": name,
                          "out_dir": os.path.join(run_dir, STAGE_DIRS[1])})
        batch_json = os.path.join(args.runs_dir, "_batch_stage1.json")
        with open(batch_json, "w") as fh:
            json.dump(items, fh, indent=2)
        argv = [PY_HUNYUAN, os.path.join(HERE, "stages", "stage1_shape.py"),
                "--batch", batch_json,
                "--steps", str(args.steps), "--guidance", str(args.guidance),
                "--octree-resolution", str(args.octree_resolution),
                "--num-chunks", str(args.num_chunks),
                "--max-faces", str(args.max_faces), "--paint-views", str(args.paint_views),
                "--paint-resolution", str(args.paint_resolution), "--seed", str(args.seed),
                "--model-path", args.model_path]
        if not args.texture:
            argv.append("--no-texture")
        r = Runner(args.runs_dir, "batch")
        r.log_path = os.path.join(args.runs_dir, "_batch_stage1.log")
        r.manifest_path = os.path.join(args.runs_dir, "_batch_stage1.manifest.json")
        r.run(1, argv, cwd=HUNYUAN, env={"PYTHONUNBUFFERED": "1"})
        os.remove(batch_json)
        # Seed each object's own manifest so the per-object child (--from 2)
        # finds stage 1's outputs exactly where a single run would have put them.
        for item in items:
            rr = Runner(os.path.dirname(item["out_dir"]), item["name"])
            with open(os.path.join(item["out_dir"], "stage1.json")) as fh:
                rr.manifest["stage1"] = json.load(fh)
            rr.manifest.setdefault("name", item["name"])
            rr.save()
            if args.clean:
                d = {k: os.path.join(os.path.dirname(item["out_dir"]), v)
                     for k, v in STAGE_DIRS.items()}
                cleanup_stage(1, d, item["name"], rr)

    if args.to_stage < 2:
        return
    passthrough = ["--from", str(max(2, args.from_stage)), "--to", str(args.to_stage),
                   "--runs-dir", args.runs_dir, "--quality", args.quality,
                   "--rot-x", str(args.rot_x), "--rot-y", str(args.rot_y),
                   "--rot-z", str(args.rot_z), "--attn", args.attn,
                   "--max-new-tokens", str(args.max_new_tokens),
                   "--attempts", str(args.attempts), "--seed", str(args.seed),
                   "--scale-unit", args.scale_unit,
                   "--limit-convention", args.limit_convention,
                   "--collision", args.collision]
    for flag, on in (("--no-texture", not args.texture), ("--no-preview", not args.preview),
                     ("--fixed-base", args.fixed_base), ("--no-rigid", not args.rigid),
                     ("--keep-intermediates", not args.clean)):
        if on:
            passthrough.append(flag)
    if args.scale_m:
        passthrough += ["--scale-m", str(args.scale_m)]
    if args.total_mass:
        passthrough += ["--total-mass", str(args.total_mass)]
    if args.expect_size:
        passthrough += ["--expect-size", str(args.expect_size[0]), str(args.expect_size[1])]
    for name in names:
        print(f"\n==> [{name}] stages {max(2, args.from_stage)}..{args.to_stage}")
        subprocess.check_call([sys.executable, os.path.abspath(__file__),
                               "--name", name] + passthrough)
    print(f"\n==> batch done: {len(names)} object(s) under {args.runs_dir}")


def cleanup_stage(stage, d, name, r: Runner):
    """Drop per-run files that are provably redundant (md5-identical to a kept
    file, or written-then-never-read).  ~40 MB/run on the ladder; every removal
    here was verified against what later stages actually read.

    stage 1: white_mesh_remesh.obj -- hy3dpaint's remesh temp, written next to
             its input and never consumed again.
    stage 2: _input/ -- a staging copy of stage 1's glb made only to control the
             output filename.
    stage 3: <name>_scaled.glb -- infer.py re-normalises an already-normalised
             mesh, so this is byte-identical to stage 2's output; and the
             per-part material.mtl/material_0.png -- trimesh duplicates the full
             albedo atlas into every part dir, and stage 4 deliberately ignores
             them (it rebinds Hunyuan's original PBR maps).
    """
    doomed = []
    if stage == 1:
        doomed.append(os.path.join(d[1], "white_mesh_remesh.obj"))
    elif stage == 2:
        doomed.append(os.path.join(d[2], "_input"))
    elif stage == 3:
        doomed.append(os.path.join(d[3], "attempts"))
        doomed.append(os.path.join(d[3], f"{name}_scaled.glb"))
        parts = os.path.join(d[3], f"{name}_objs")
        if os.path.isdir(parts):
            for pid in os.listdir(parts):
                for f in ("material.mtl", "material_0.png"):
                    doomed.append(os.path.join(parts, pid, f))
    freed = 0
    for path in doomed:
        try:
            if os.path.isdir(path):
                freed += sum(os.path.getsize(os.path.join(dp, f))
                             for dp, _, fs in os.walk(path) for f in fs)
                shutil.rmtree(path)
            elif os.path.isfile(path):
                freed += os.path.getsize(path)
                os.remove(path)
        except OSError:
            pass
    if freed:
        r.log(f"    cleaned {freed / 1e6:.1f} MB of stage-{stage} intermediates")


def check_bbox_agreement(r: Runner, art_report, rigid_report, tol=0.10):
    """The articulated and rigid USDs describe the same object at the same scale,
    so their world bboxes must agree.  A frame-rotation mistake shows up here as
    an axis swap (~60% off) long before anyone opens Isaac Sim -- this exact
    check is what the original rigid-tier Y/Z bug would have tripped."""
    try:
        with open(art_report) as fh:
            a = json.load(fh)
        with open(rigid_report) as fh:
            b = json.load(fh)
        ea = [hi - lo for lo, hi in zip(a["bbox_world_min"], a["bbox_world_max"])]
        eb = [hi - lo for lo, hi in zip(b["bbox_world_min"], b["bbox_world_max"])]
    except (OSError, KeyError, ValueError) as exc:
        r.log(f"    [warn] bbox cross-check skipped: {exc}")
        return
    scale = max(max(ea), 1e-9)
    worst = max(abs(x - y) for x, y in zip(ea, eb)) / scale
    detail = f"articulated={[round(v, 3) for v in ea]} rigid={[round(v, 3) for v in eb]}"
    if worst > tol:
        r.log(f"    [FAIL] articulated/rigid bboxes disagree ({worst:.0%}): {detail}")
        raise SystemExit(
            "bbox cross-check failed -- the two USDs describe differently-oriented "
            "or differently-sized objects. Inspect the *.report.json files; "
            "use --no-rigid to skip the rigid tier if the mismatch is expected "
            "(e.g. SimArt dropped a part).")
    r.log(f"    bbox cross-check ok ({worst:.1%} worst-axis difference): {detail}")


def verify(r: Runner, usd, links, joints, args):
    argv = [PY_USD, os.path.join(HERE, "usd", "verify_usd.py"), usd,
            "--collision", args.collision, "--json", usd + ".checks.json"]
    if links is not None:
        argv += ["--expect-links", str(links)]
    if joints is not None:
        argv += ["--expect-joints", str(joints)]
    if args.expect_size:
        argv += ["--size-range", str(args.expect_size[0]), str(args.expect_size[1])]
    r.run(4, argv, cwd=HERE)


if __name__ == "__main__":
    main()
