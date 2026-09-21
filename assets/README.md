# Fiatlux Assets

USD assets for the Fiatlux benchmark (Unitree G1 humanoid replacing a light bulb).
Assets are stored in the [`haw-ai-i/fiatlux-assets`](https://huggingface.co/datasets/haw-ai-i/fiatlux-assets)
HF dataset and synced locally via `download_assets.sh`. Binary USD files are git-ignored.

## Download

```bash
# Robot + task assets + room dressing (table, warehouse backdrop, HDRI sky)
./assets/download_assets.sh

# ... + the optional extras (omniverse_climb, omniverse_lamp)
./assets/download_assets.sh --scene-dressing
```

Room dressing (`isaac_*` paths below) is always synced -- it's what every recorded
run looks like by default. The `--scene-dressing` extras are opt-in and nothing in the task
scene loads them.

Requires `uv` (assets are fetched via `uvx --from huggingface_hub hf`) and access to the
`haw-ai-i` HF org. Override the source dataset repo with the `FIATLUX_ASSET_REPO` env var.

## HF Paths

### BEHAVIOR-1K -- scene dressing only

**No benchmark task loads these.** The bulb and socket are the Omniverse pair
(`BULB_USD` / `SOCKET_USD`) and the ladder is `omniverse_ladder/AlumStep_D`; no scene entity
references a `behavior1k_*` asset.

The eleven lighting-fixture categories are still synced by `./download_assets.sh
--scene-dressing`, so they are on hand if you want to dress a scene of your own or add a
fixture to an existing one. `behavior1k_materials/` syncs with them and is **required** --
every B1K asset binds OmniGibson's shared `OmniGibsonVRayMtl` by relative path into that
bundle, and without it they render red rather than failing loudly (issue 18).

The task-asset groups (`behavior1k_bulb`, `_bulb_broken`, `_lamp`, `_ladder`) are **not**
synced: they were the original bulb/lamp/ladder choice and nothing uses them now. They remain
in the dataset; clearing them is a dataset-cleanup decision, not a code one.

### Omniverse asset packs (NVIDIA SimReady / ArchVis)

| HF path                              | Category         | Models                                 | Role           | Size   |
| ------------------------------------- | ---------------- | -------------------------------------- | -------------- | ------ |
| `hf://datasets/haw-ai-i/fiatlux-assets/omniverse_ladder/` | `ladder`       | 98 ladders/platforms (16 designs)      | task asset     | 3.4 GB |
| `hf://datasets/haw-ai-i/fiatlux-assets/omniverse_bulb/`   | `light_bulb`   | 1 (separable LightBulb)                | task asset     | small  |
| `hf://datasets/haw-ai-i/fiatlux-assets/omniverse_climb/`  | `climb`        | Mezzanine_A + OfficeSet_A              | scene dressing | 1.9 GB |
| `hf://datasets/haw-ai-i/fiatlux-assets/omniverse_lamp/`   | `lamp_fixture` | residential lamps/chandeliers/fixtures | scene dressing | 1.2 GB |

**Omniverse total: ~6.5 GB across 4 paths.** Each ladder is authored in **three physics tiers**
beside the original geometry — the scene builder just references the one it wants:

| File | Physics | Behaviour |
|---|---|---|
| `<name>.usd` (original) | none | geometry only (robot clips through) — visual prop |
| `<name>_collision.usd` | collision, static | **solid + fixed** — climbable, never moves |
| `<name>_collision_rigid.usd` | collision + rigid body | **solid + movable** — climbable *and* can tip / be carried |

The static `_collision.usd` files (convex decomposition, high friction) are in the dataset and all 98 are
verified climbable in Isaac Sim.
The movable `_collision_rigid.usd` variants are authored beside each by
`scripts/omniverse/omniverse_ladder_rigid.py` and upload with the
ladder group. `_collision_rigid.usd` sublayers `_collision.usd` sublayers the original, so **the three files
must stay together**. Note: as rigid bodies, **extension and folded ladders can't free-stand** — they
need a wall/support to lean on (they're still climbable leaned). Per-asset detail in
[`omniverse_uploaded_manifest.csv`](omniverse_uploaded_manifest.csv).

> Note: Omniverse ladder assets reference shared Omniverse/Kit MDL material libraries; their
> extracted base materials and textures are uploaded alongside the geometry, so each ladder
> group ships geometry + collision + materials together.

### Verification contract

Every task-critical asset is physics-verified **once at intake** (collider audit /
drop test), with the result recorded in the manifests (`collision_verified`,
`has_physics`, `physics_notes` columns); the binary USD is immutable afterward.
Ongoing verification targets the *composition*, not the assets: run
`scripts/verify_scene.py` (scene solidity) and `scripts/verify_interactions.py`
(graded interactions) after env changes and as a pre-flight before scoring runs —
never in per-commit CI.

## Isaac Sim Nucleus mirror (room dressing)

The table, room backdrop, and HDRI sky are mirrored once from Isaac Sim 5.1's own
Nucleus content library (a public HTTPS/S3 endpoint, no Omniverse client needed)
into our own dataset repo, so nothing is fetched live from NVIDIA's CDN at sim launch.

| HF path                                                        | Nucleus source                          | Role               | Files |
| ---------------------------------------------------------------- | ---------------------------------------- | ------------------ | ----- |
| `hf://datasets/haw-ai-i/fiatlux-assets/isaac_packing_table/`                      | `Isaac/Props/PackingTable/`             | table for lamp/bulb | 112   |
| `hf://datasets/haw-ai-i/fiatlux-assets/isaac_room/Environments/Simple_Room/`      | `Isaac/Environments/Simple_Room/`       | room backdrop (walls/floor/windows) | 89 |
| `hf://datasets/haw-ai-i/fiatlux-assets/isaac_skies/`                              | `Isaac/Materials/Textures/Skies/PolyHaven/` | HDRI dome light | 1     |

`isaac_room/` uses `Simple_Room` rather than the much heavier `Simple_Warehouse`
demo scene: the warehouse ships ~100+ unique MDL materials, and NVIDIA's MDL
compiler compiles each one on first use single-threaded -- that made a single
`--record video` render take hours. `Simple_Room` has ~11 materials and still
gives real walls/floor/windows instead of a bare plane. Nested to match its
original Nucleus depth (`Environments/Simple_Room/`) as a precaution against
relative references, even though it doesn't escape its own tree like the
warehouse's `Props/KLT_Bin/` dependency did.

Mirror-specific per-asset detail (Nucleus source path, role, notes) in
[`isaac_mirror_manifest.csv`](isaac_mirror_manifest.csv). Development-only thumbnail
caches (`.thumbs/`) under the Nucleus source trees were excluded from the mirror.
Isaac Sim's asset-root URL is version-pinned (`.../Assets/Isaac/5.1/...`) -- re-verify
these paths after any Isaac Sim upgrade.

## Source

**BEHAVIOR-1K** (scene dressing, above) is from [BEHAVIOR-1K](https://behavior.stanford.edu/) (Stanford
OmniGibson dataset), decrypted from `.encrypted.usd` using the `omnigibson.key`. The key and
decrypted binaries were never committed to git. The intake scripts are `scripts/behavior1k/`.

**Omniverse** assets are from the NVIDIA [Omniverse downloadable USD packs](https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html#d-openusd-asset-packs)
(Warehouse, SimReady Warehouse 01, Residential, Sample Scenes). Used under non-commercial
academic terms. See `docs/omniverse_pack_scan_log.md` for the full 14-pack scan and
`docs/collision_authoring_explained.md` for collision authoring.

**Room-dressing** assets (table, room, sky) are NVIDIA's own Isaac Sim sample content
(Props/Environments) plus a CC0 PolyHaven HDRI, mirrored per the table above.
