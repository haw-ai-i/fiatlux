# image → mesh → articulated USD

One photo in, one physics-ready **USD** for NVIDIA Isaac Sim out — by chaining
[Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) (image →
textured mesh) into [SimArt](https://github.com/ByteDance-Seed/SimArt)
(mesh → articulated parts + URDF), then authoring the USD directly with
`usd-core` — SimArt stops at URDF, and Isaac Sim's own importer can't run on a
GPU without RT cores, so that last step is implemented here.

```
images/ladder.jpg
   │ ① Hunyuan3D-2.1     image → textured mesh (.glb/.obj + PBR maps)
   │ ② SimArt prep       centre at origin, scale max extent to 1.0
   │ ③ SimArt            mesh → part decomposition + kinematics (URDF + JSON)
   │ ④ urdf_to_usd.py    URDF + JSON → articulated USD, verified GPU-free
   ▼
runs/ladder/04_usd/ladder.usda   (+ ladder_rigid*.usda single-body fallback)
```

The result is Z-up, in metres, with convex-decomposition colliders, PBR
textures, joint limits, and per-part density — checked by `usd/verify_usd.py`
(27 structural assertions, incl. joint-frame coincidence) plus an
articulated-vs-rigid bbox cross-check.

## Install

Once, ~20 min plus ~55 GB of downloads. Needs an NVIDIA GPU, `sudo`, and
[uv](https://docs.astral.sh/uv/).

```bash
cd auto/pipeline
bash setup/00_clone_repos.sh      # clone the two model repos at pinned commits, apply patch
bash setup/01_system_deps.sh      # sudo: CUDA 12.4 toolkit, gcc/g++-12, GL/EGL libs
bash setup/02_env_hunyuan.sh      # py3.11 venv: torch 2.5.1+cu124, 2 native builds
bash setup/03_env_simart.sh       # py3.10 venv: torch 2.4.0+cu121, 19 GB checkpoints, Blender
bash setup/04_env_usd.sh          # py3.11 venv: usd-core 25.5.1 (seconds, no GPU)
bash setup/05_fetch_weights.sh    # ~35 GB Hunyuan3D + DINOv2 + rembg, so runs never stall on downloads
```

Every script is idempotent — re-run freely after a failure. The three separate
virtualenvs are required: the stages' dependency pins conflict irreconcilably
(see [docs/NOTES.md](docs/NOTES.md)).

## Use

```bash
python3 run_pipeline.py --image ../images/ladder.jpg --name ladder \
    --scale-m 1.83 --total-mass 13.0 --expect-size 1.0 2.5
```

`--scale-m` (real height, metres) and `--total-mass` (kg) override SimArt's
predicted size/density, which are rough guesses; `--expect-size` makes the
verifier assert the result is in a plausible range. Outputs land in
`runs/<name>/`, one directory per stage, with `manifest.json` + `pipeline.log`.

Common variants:

```bash
# Re-run one stage against existing intermediates (stage 4 takes ~1 s)
python3 run_pipeline.py --name ladder --from 4 --scale-m 1.83 --total-mass 13.0

# SimArt samples its predictions -- try 5 seeds, keep the best articulation
python3 run_pipeline.py --name ladder --from 3 --attempts 5

# Speed/fidelity presets for stage 1 (default: 50 steps, octree 384, paint 6x512)
python3 run_pipeline.py --image X.jpg --quality fast     # or: high
python3 run_pipeline.py --image X.jpg --no-texture       # untextured, much faster

# Whole folder of images; stage-1 models load once for all of them
python3 run_pipeline.py --images ../images/

# Skip articulation entirely: photo -> single rigid body with colliders
python3 run_pipeline.py --image X.jpg --rigid-only
```

`python3 run_pipeline.py --help` lists every flag. Runs are seeded and
reproducible (`--seed`). If the object comes out mis-oriented, check the
preview renders in `02_normalized/*_renders/` and adjust `--rot-x/y/z`.

To consume the result in Isaac Sim, copy `runs/<name>/04_usd/` wholesale (the
USD references `./textures/` relatively): `ladder.usda` is the articulated
asset, `ladder_rigid_static.usda` a static prop, `ladder_rigid.usda` a dynamic
single body. Spawn-config snippet in [docs/NOTES.md](docs/NOTES.md).

## Repository structure

```
auto/
├── images/               demo input (ladder.jpg)
├── Hunyuan3D-2.1/        cloned by setup/00 (pinned 82920d6, not in git)
├── SimArt/               cloned by setup/00 (pinned c8d0829, not in git)
└── pipeline/             ← everything authored lives here
    ├── run_pipeline.py   stdlib-only driver: venv dispatch, batching, cleanup, verification
    ├── stages/stage1_shape.py    Hunyuan3D runner (background removal, shape, PBR paint)
    ├── usd/urdf_to_usd.py        URDF + prediction JSON → USD (pure pxr; also --rigid, --score)
    ├── usd/verify_usd.py         GPU-free structural checks on the produced USD
    ├── setup/00..05_*.sh         clone / system deps / three venvs / weight prefetch
    ├── patches/simart_infer.patch  3 documented SimArt fixes (attn backend, JSON guard, seeding)
    ├── docs/NOTES.md             engineering notes: every upstream gotcha and design decision
    └── runs/<name>/              per-run outputs (00_input … 04_usd), not in git
```

Model repos, weights (~55 GB), venvs and runs are never committed — enforced by
`fiatlux/.gitignore` and per-clone `.git/info/exclude` (details in the notes).

## Credits

This pipeline glues together, and would be nothing without:

* **[Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)** — Tencent
  Hunyuan3D Team. Image-to-mesh and PBR texture synthesis (stage 1). License:
  *Tencent Hunyuan 3D 2.1 Community License* (region restrictions apply — see
  their LICENSE). Pinned at `82920d6`.
* **[SimArt](https://github.com/ByteDance-Seed/SimArt)** — ByteDance Seed,
  SIGGRAPH 2026. Part decomposition and kinematic prediction (stages 2–3).
  Apache-2.0. Pinned at `c8d0829`.
* **[OpenUSD](https://openusd.org)** (`usd-core`) — Pixar; **[Blender](https://www.blender.org)**
  3.0.1 (SimArt's renderer); **[DINOv2](https://github.com/facebookresearch/dinov2)** — Meta;
  **[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)**; **[rembg](https://github.com/danielgatis/rembg)**;
  and NVIDIA **Isaac Sim / Isaac Lab**, whose asset conventions this follows.

```bibtex
@misc{hunyuan3d2025hunyuan3d,
    title={Hunyuan3D 2.1: From Images to High-Fidelity 3D Assets with Production-Ready PBR Material},
    author={Tencent Hunyuan3D Team},
    year={2025}, eprint={2506.15442}, archivePrefix={arXiv}, primaryClass={cs.CV}
}

@article{zhang2026simart,
    title={SIMART: Decomposing Monolithic Meshes into Sim-ready Articulated Assets via MLLM},
    author={Zhang, Chuanrui and Qin, Minghan and Wang, Yuang and Xie, Baifeng and Li, Hang and Wang, Ziwei},
    journal={arXiv preprint arXiv:2603.23386},
    year={2026}
}
```
