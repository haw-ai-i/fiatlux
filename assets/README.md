# Fiatlux Assets

USD assets for the Fiatlux benchmark (Unitree G1 humanoid replacing a light bulb).
Assets are stored in the [`haw-ai-i/fiatlux-assets`](https://huggingface.co/datasets/haw-ai-i/fiatlux-assets)
HF dataset and synced locally via `download_assets.sh`. Binary USD files are git-ignored.

## Download

```bash
# Robot + task assets + room dressing (table, warehouse backdrop, HDRI sky)
./assets/download_assets.sh

# ... + opt-in scene dressing (Omniverse climb structures and residential lamps)
./assets/download_assets.sh --scene-dressing
```

Room dressing (`isaac_*` paths below) is always synced -- it's what every recorded
run looks like by default -- unlike the opt-in scene dressing.

Requires `uv` (assets are fetched via `uvx --from huggingface_hub hf`). The dataset is meant to be
public and to need no Hugging Face login; until the post-merge publish step lands (see
[#231](https://github.com/haw-ai-i/fiatlux/issues/231)), or if you hit an auth error, run
`uvx --from huggingface_hub hf auth login` or set `HF_TOKEN` to a token scoped to the
`haw-ai-i` org. Override the source dataset repo with the `FIATLUX_ASSET_REPO` env var.

## HF Paths

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
`scripts/verify_scene.py` (scene solidity) after env changes and as a pre-flight
before scoring runs — never in per-commit CI.

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

## Provenance and licences

Every group in the dataset is either our own work or mirrored from a third party. This table is
the record of where each group came from and the terms it carries. The licence audit for the
public release is tracked in [#231](https://github.com/haw-ai-i/fiatlux/issues/231); the *Status*
column records the decision taken on 2026-09-22.

| Group(s) | Source | Terms | Status |
|---|---|---|---|
| `unitree_g1/` | Unitree's pre-assembled G1 USDs, mirrored from [`unitreerobotics/unitree_sim_isaaclab_usds`](https://huggingface.co/datasets/unitreerobotics/unitree_sim_isaaclab_usds) | Apache-2.0 | keep; add the licence file to the group |
| `omniverse_ladder/`, `omniverse_bulb/`, `omniverse_climb/`, `omniverse_lamp/` | NVIDIA [OpenUSD asset packs](https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html): Warehouse (88 models), Residential (28), SimReady Warehouse 01 (11), Sample Scenes (1). The collision overlays (`*_collision.usd`, `*_collision_rigid.usd`) and the `_platform` variants are authored here -- see [`docs/collision_authoring_explained.md`](../docs/collision_authoring_explained.md) -- and so is the bulb/socket split. Per-asset detail in [`omniverse_uploaded_manifest.csv`](omniverse_uploaded_manifest.csv); the full 14-pack scan behind the selection is in [`docs/omniverse_pack_scan_log.md`](../docs/omniverse_pack_scan_log.md); some ladder USDs were also fixed in place at intake, see [`docs/omniverse_material_fixes.md`](../docs/omniverse_material_fixes.md) | NVIDIA [Product-Specific Terms for NVIDIA AI Products](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/) (Omniverse is covered there since May 2026). They permit distributing the software and derivative samples "as part of a Customer Product"; they do not address re-hosting pack content on its own | **kept on HF**: mirrored as part of the benchmark under those terms; NVIDIA's notice and a pointer to the source pack are being added to each group; the authors accept that reading |
| `isaac_packing_table/`, `isaac_room/` | Isaac Sim 5.1 Nucleus content (`Isaac/Props/PackingTable/`, `Isaac/Environments/Simple_Room/`), see [`isaac_mirror_manifest.csv`](isaac_mirror_manifest.csv) | [NVIDIA Isaac Sim Additional Software and Materials License](https://docs.isaacsim.omniverse.nvidia.com/latest/common/license-isaac-sim-additional.html); §2.2 restricts distribution of "any portion of the Software" | **kept on HF**: same reading; NVIDIA's notice and a pointer to the source pack are being added to this group too; 0.27 GB of the sample content Isaac Sim itself fetches at runtime |
| `isaac_skies/` | `kloofendal_43d_clear_puresky_4k.hdr` from [Poly Haven](https://polyhaven.com/a/kloofendal_43d_clear_puresky), via the Isaac Sim mirror | CC0 | keep |
| `behavior1k_*/` (every group, without exception: the four former task-asset groups, the eleven lighting-fixture categories, and `behavior1k_materials`) | [BEHAVIOR-1K](https://behavior.stanford.edu/) (Stanford OmniGibson dataset), decrypted from `.encrypted.usd` | The BEHAVIOR-1K models come from ShapeNet and TurboSquid and are distributed encrypted so that they can only be used inside OmniGibson | **being removed**: redistribution is not permitted, and no benchmark preset loaded them. `download_assets.sh` no longer syncs any of them (#232); deleting the groups from the dataset is the remaining step |

The benchmark code itself is Apache-2.0 (`../LICENSE`); files derived from Isaac Lab are BSD-3
(`../LICENSE.isaaclab`).
