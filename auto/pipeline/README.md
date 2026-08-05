# `auto/pipeline` — image → mesh → articulated USD

Turns a **single photo** into a **physics-ready USD** for NVIDIA Isaac Sim, by
chaining the two models vendored next door and then authoring the USD by hand.

```
images/ladder.jpg
      │
      │  ① Hunyuan3D-2.1            image → textured mesh
      ▼        .glb + .obj + albedo/metallic/roughness .jpg
      │  ② SimArt preprocessing     centre at origin, scale max extent to 1.0, +Z up
      ▼        normalised .glb + preview renders
      │  ③ SimArt (Qwen3-VL-8B)     mesh → part decomposition + kinematics
      ▼        .urdf + per-part .obj + prediction .json
      │  ④ urdf_to_usd.py           URDF + JSON → USD   ← written here; exists nowhere else
      ▼
runs/ladder/04_usd/ladder.usda      articulated, collidable, textured, Z-up, metres
```

**Why step ④ had to be written.** SimArt stops at URDF — a repo-wide grep for
`pxr|Usd|UsdPhysics` returns nothing. The usual answer, Isaac Sim's URDF
importer, cannot run on this box: an A100 has no RT cores. So the stage authors
USD directly with `usd-core`, following the conventions the parent `fiatlux`
benchmark already uses (`scripts/omniverse/omniverse_ladder_collision.py`,
`omniverse_bulb_rigid.py`).

---

## Setup (once, ~20 min + ~55 GB of downloads)

Only `pipeline/` is in git. The two model repos are **cloned**, and every model
weight is **downloaded** — none of it is tracked here (see *What is and isn't in
git* below).

```bash
cd auto/pipeline
bash setup/00_clone_repos.sh     # git clone Hunyuan3D-2.1 + SimArt at pinned commits, apply patch
bash setup/01_system_deps.sh     # sudo: CUDA 12.4 toolkit, gcc/g++-12, GL/EGL libs
bash setup/02_env_hunyuan.sh     # py3.11 venv, torch 2.5.1+cu124, 2 native builds, RealESRGAN
bash setup/03_env_simart.sh      # py3.10 venv, torch 2.4.0+cu121, 19 GB checkpoints, Blender 3.0.1
bash setup/04_env_usd.sh         # py3.11 venv, usd-core 25.5.1  (seconds; no GPU)
```

All five are idempotent — safe to re-run after a failure.

`run_pipeline.py` calls step 00 itself if either repo is missing, so a fresh
checkout can go straight to `--image …` and the clone happens first; the rest of
setup still has to be run once. `--no-clone` turns that into an error instead.

Both repos are pinned to the commits this pipeline was validated against
(`Hunyuan3D-2.1` `82920d6`, `SimArt` `c8d0829`) — the workarounds in `stages/`
and `usd/` cite specific upstream line numbers, so an unpinned clone would rot.
Track upstream with `HUNYUAN_REF=main SIMART_REF=main bash setup/00_clone_repos.sh`.

Three separate virtualenvs under `.venvs/` are not fussiness: Hunyuan3D needs
`numpy==1.24.4` + `bpy`, SimArt needs `torch==2.4.0+cu121` + `transformers==4.57`,
and `fiatlux` itself is `torch 2.7.0+cu128`. They cannot coexist, so the stages
hand off through **files** and the driver runs on the *system* python.

Hunyuan3D's own weights (~30 GB) download on the first stage-1 run, into
`~/.cache/hy3dgen` and `~/.cache/huggingface`.

## Run it

```bash
python3 run_pipeline.py --image ../images/ladder.jpg --name ladder --expect-size 0.5 3.0
```

Everything lands in `runs/ladder/`:

```
00_input/      ladder.jpg
01_shape/      ladder.glb  ladder.obj/.mtl  ladder{,_metallic,_roughness}.jpg  ladder_white.glb
               ladder_rgba.png            ← background-removed input; check this first if the shape is wrong
02_normalized/ ladder.glb  ladder_renders/render_00*.png   ← 5 previews: is it upright and facing forward?
03_articulate/ ladder.urdf  ladder.json  ladder_objs/<pid>/<pid>.obj
04_usd/        ladder.usda        ← the deliverable (articulated)
               ladder_rigid.usda  ← single-body fallback (+ _static tier)
               textures/albedo.jpg metallic.jpg roughness.jpg
               *.report.json  *.checks.json
manifest.json  pipeline.log
```

### Re-running one stage

Stages are independent, which matters because a cold run is 1–2 hours and each
stage fails for different reasons:

```bash
python3 run_pipeline.py --name ladder --from 4              # just re-author the USD
python3 run_pipeline.py --name ladder --from 3 --to 4       # re-predict articulation, then USD
python3 run_pipeline.py --image X.jpg --name x --to 1       # generate the mesh only
```

### Useful variants

```bash
# Fast, untextured: skips PBR painting entirely (minutes instead of ~1 h)
python3 run_pipeline.py --image X.jpg --name x --no-texture

# Skip SimArt: photo -> single rigid body with colliders. The safe fallback.
python3 run_pipeline.py --image X.jpg --name x --rigid-only

# The object is not upright in 02_normalized/*_renders/ -- fix the input frame
python3 run_pipeline.py --name x --from 2 --rot-x -90

# Override the predicted real-world size (metres, largest bbox extent)
python3 run_pipeline.py --name x --from 4 --scale-m 1.83

# Pin the base to the world (a static prop rather than a free body)
python3 run_pipeline.py --name x --from 4 --fixed-base
```

## Stage reference

| # | Runs in | CWD | Script |
|---|---|---|---|
| 1 | `.venvs/hunyuan` | `Hunyuan3D-2.1/` | `stages/stage1_shape.py` |
| 2 | `.venvs/simart` | `SimArt/` | `SimArt/scripts/process_raw_objects.py` |
| 3 | `.venvs/simart` | `SimArt/` | `SimArt/inference/infer.py` |
| 4 | `.venvs/usd` | anywhere | `usd/urdf_to_usd.py`, `usd/verify_usd.py` |

**Stage 1** `--steps 50 --octree-resolution 384 --max-faces 40000 --paint-views 6
--paint-resolution 512 --seed 1234 --no-texture`
**Stage 2** `--rot-x/--rot-y/--rot-z 0 --no-preview`
**Stage 3** `--attn sdpa --max-new-tokens 24000`
**Stage 4** `--scale-m --scale-unit auto --total-mass --limit-convention spec
--collision convexDecomposition --fixed-base --no-rigid --expect-size MIN MAX`

### Sanity-check the prediction — it is the weakest link

The pipeline converts faithfully whatever SimArt predicts, and SimArt's physical
numbers are guesses. On the ladder run it returned `scale: 50` (→ 0.5 m for an
object whose own label reads **6 ft**) and material `Metal` at 7800 kg/m³ — solid
steel, for a ~13 kg fibreglass ladder, which composes to a 251 kg asset that
would behave absurdly in simulation. Both are one-flag fixes, and stage 4 costs
about a second, so re-authoring is cheap:

```bash
python3 run_pipeline.py --name ladder --from 4 --scale-m 1.83 --total-mass 13.0
```

`--total-mass` divides the pinned mass by the summed mesh volume and authors that
one density on every link. Read `04_usd/*.report.json` for the per-link table,
the scale provenance, and the joint limits under both conventions.

`usd/urdf_to_usd.py` and `usd/verify_usd.py` are standalone CLIs; run
`--help` on either. Neither needs a GPU.

## What the USD contains

* `upAxis = Z`, `metersPerUnit = 1`, `kilogramsPerUnit = 1` — the fiatlux convention.
* `/[Name]` — `UsdPhysics.ArticulationRootAPI`, and **no xformOps at all** (see below).
* `/[Name]/l_<pid>` per part — `RigidBodyAPI` + `MassAPI` with **`physics:density`**,
  so PhysX derives mass *and* a correct inertia tensor from the cooked collider.
  Segmented parts are open shells whose analytic volume is unreliable, and a wrong
  authored inertia is worse than a computed one.
* `/[Name]/l_<pid>/mesh` — geometry + `primvars:st`, `CollisionAPI` +
  `MeshCollisionAPI(convexDecomposition)`, bound to a high-friction physics
  material (1.2 / 1.0 / 0.0). Convex *decomposition*, not a hull: a hull fills
  the gaps between rungs and destroys climbability.
* `/[Name]/Joints/j_<pid>` — `PhysicsRevoluteJoint` / `PrismaticJoint` / `FixedJoint`.
* `/[Name]/Looks/SimArtPBR` — one shared `UsdPreviewSurface` fed by the
  albedo/metallic/roughness maps in `textures/`, by relative asset path.

Verify any of it without a GPU:

```bash
.venvs/usd/bin/python usd/verify_usd.py runs/ladder/04_usd/ladder.usda \
    --expect-links 5 --expect-joints 4 --size-range 0.5 3.0
```

The load-bearing check is **`joint frames coincide`**: it recomposes each joint's
two body frames through `XformCache` and asserts they land on the same world
pose. That single assertion catches the whole absolute-vs-relative coordinate
bug class — which is exactly the trap SimArt's own URDF falls into (below).

## Things that will bite you

These are all upstream behaviours the pipeline works around. Recorded because
each one cost real debugging time and none is obvious from the source.

| Issue | Where | What we do |
|---|---|---|
| **Background removal never runs.** `image.convert("RGBA")` then `if image.mode == "RGB"` — a branch that can no longer be true. A JPEG gets an all-opaque alpha, the recentring preprocessor treats the whole frame as foreground, and the shape comes out wrong. | `demo.py:25`, `model_worker.py:165` | call `BackgroundRemover()` unconditionally; result saved as `*_rgba.png` |
| **Part OBJs are in object-global coordinates.** `trimesh.submesh` only re-indexes. SimArt compensates with `joint origin=+center` / `visual origin=−center`, which cancels **only at tree depth 1**; deeper parts are displaced by their parent's offset. | `utils/urdf_utils.py:431,450` | put each link's frame at its own *absolute* anchor and offset its points by `−anchor` — correct at any depth |
| **SimArt's assets are Y-up, not "+Z up" as its README states.** `chair_00.glb` and `fridge_00.glb` both carry their height along Y. `infer.py:428` then applies `R_x(+90)` internally, which is what actually produces the Z-up frame the model was trained in. Rotating the input to Z-up yourself — as the README tells you to — double-applies it and lays the object on its side. | `SimArt/README.md` vs `infer.py:428` | stage 2 uses `--rot-x 0`, stage 4 uses `--frame-rot none`; both are documented at their definitions |
| **`subprocess(cwd=…)` does not update `PWD`, and Blender resolves relative script paths from `PWD`.** So `render_utils.py`'s relative blender-script path was looked up under the *caller's* directory. The failure is silent and confusing: Python's own `os.path.exists()` on the identical string passes, so you get "Render failed, output not found" with no hint why. | `render_utils.py:70` | `Runner.run` sets `PWD` to the child's cwd for every stage |
| **Revolute limits are 2× too wide.** The writer computes `(val/100)·2π` rad; the model's own prompt says "for 'revolute', 100 represents 180 degrees". The prismatic branch beside it uses a bare `val/100` with no `2π`. | `urdf_utils.py:441` vs `infer.py:101` | `--limit-convention spec` (default) uses `·180°`; `urdf` reproduces SimArt. Both are printed per joint |
| **`scale` has no unit.** The prompt never specifies one, so the model emits bare numbers (`"scale": 30` for a box). 30 m would be a six-storey box. | `infer.py:103` | magnitude heuristic: ≤10 → m, ≤500 → cm, else mm — always logged; `--scale-m` / `--scale-unit` override |
| **Prismatic limits are in normalised units**, not metres. | `urdf_utils.py:446` | multiplied by `scale_m`, then clamped to the part's own extent |
| **`.obj` output path is mandatory.** The writer is `save_obj_mesh` regardless of extension, so `demo.py`'s `output_mesh_path='demo_textured.glb'` writes OBJ text into a `.glb`. `__call__` also returns the `.obj` path even when `save_glb=True`. | `textureGenPipeline.py:186` | always pass `.obj`, then convert via `create_glb_with_pbr_materials` (pure trimesh, no Blender) |
| **Two imports fail silently.** `bpy` and `mesh_inpaint_processor` are wrapped in bare `try/except` that only print; a missing build surfaces much later as a `NameError` in texture baking. | `MeshRender.py:31-39` | asserted in stage 1's preflight |
| **Per-part `.mtl` loses the PBR maps.** `export_obj` calls `PBRMaterial.to_simple()`, keeping only base colour. | trimesh `export.py:873` | ignore them; bind one shared material from Hunyuan3D's original maps |
| **`floating` joints** (SimArt's `FREE`) have no USD/PhysX equivalent. | `urdf_utils.py:423` | `--floating-joint fixed` (default) / `spherical` / `free-body`, always warned |
| **`vqvae` import fails** — the `sys.path` fix is commented out; and the Blender script path is CWD-relative and fails *silently*. | `infer.py:27`, `render_utils.py:70` | driver sets `PYTHONPATH` and `cwd=SimArt` |
| **`uv` inherits fiatlux's pins.** These venvs live inside the fiatlux tree, so `uv` finds its `pyproject.toml` and forces `torch==2.7.0` / py3.11 onto them. | `fiatlux/pyproject.toml:52` | every `uv` call passes `--no-config` |
| **`bpy==4.0` no longer exists on PyPI**, and every surviving release is cp311 or cp313 — there is no cp310 bpy. | — | the Hunyuan env is **CPython 3.11**, not the README's 3.10; `bpy==4.2.23` |
| **`gcc` is 12 but `g++` is 11** on this image, with no `cc1plus` for 12, so nvcc dies building `custom_rasterizer`. | — | `01_system_deps.sh` installs `g++-12`; `CC/CXX` pinned |

Two SimArt edits are applied in-tree; `patches/simart_infer.patch` documents both:
selectable attention backend (`SIMART_ATTN_IMPL`, default `sdpa`, so a run does
not hinge on a `flash_attn` wheel), and a guarded `json.loads` that dumps the raw
response instead of throwing away 24k tokens of finished generation.

### Why geometry is baked rather than transformed

The converter writes final world-space points into each mesh and gives the root
**no `xformOp` at all** — no scale, no transform. `omniverse_bulb_rigid.py` uses
a child Xform instead, but only because it *references* an immutable source
layer. Here we author the points, and a scale above a rigid body is genuinely
hazardous: PhysX scales joint `localPos0/1` by accumulated body scale, which
desynchronises the two joint frames, and it is the usual cause of "cooked
collider doesn't match the render mesh". It also means Isaac Lab's `create_prim()`,
which overwrites translate/orient/scale on the spawned prim, has nothing to
clobber. `verify_usd.py` asserts no scale op ever appears; the bake parameters
are recorded in `customData` on the root so it stays auditable.

## What is and isn't in git

Tracked: **only the ~12 authored files** under `auto/pipeline/` (plus
`auto/images/ladder.jpg`, the 96 KB demo input this README references).

Never tracked — cloned or downloaded by `setup/`:

| | Size | Where |
|---|---|---|
| `auto/Hunyuan3D-2.1/`, `auto/SimArt/` | ~700 MB | cloned by `setup/00_clone_repos.sh` |
| Hunyuan3D shape + paint + DINOv2 weights | ~30 GB | `~/.cache/hy3dgen`, `~/.cache/huggingface` |
| SimArt MLLM + VQ-VAE checkpoints | ~19 GB | `SimArt/checkpoints/` |
| RealESRGAN, Blender 3.0.1 | ~270 MB | `Hunyuan3D-2.1/hy3dpaint/ckpt/`, `pipeline/.tools/` |
| `.venvs/`, `runs/` | ~20 GB | `pipeline/` |

This is enforced in two places, because one is not enough:

* `fiatlux/.gitignore` ignores `/auto/*` and re-admits only `pipeline/` and
  `images/`. It has to be `/auto/*`, **not** `/auto/` — git cannot re-include
  anything beneath an excluded *directory*, so the `!` rules would be dead.
* Each clone's own `.git/info/exclude` (written by `setup/00_clone_repos.sh`)
  hides `checkpoints/`, `ckpt/`, `*.safetensors`, `*.pt`, `*.ckpt`, `*.so`.
  Without this, SimArt — which ships no `.gitignore` at all — shows its 19 GB
  `checkpoints/` as plain untracked files, one `git add .` away from disaster.
  `info/exclude` is used rather than editing `.gitignore` so the clones stay
  byte-identical to upstream apart from `patches/simart_infer.patch`.

## Next step — using it in Isaac Sim

Isaac Sim cannot run here (no RT cores), so this is untested on-device and is
the natural next task on an RTX machine:

```python
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import UsdFileCfg

LADDER = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Ladder",
    spawn=UsdFileCfg(usd_path=".../runs/ladder/04_usd/ladder.usda"),
)
```

Copy `04_usd/` wholesale — the `.usda` references `./textures/` by relative
path, so the directory has to travel together. For a static prop, use
`ladder_rigid.usda` (or re-author with `--fixed-base`).
