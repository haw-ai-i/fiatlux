---
pretty_name: Fiatlux benchmark assets
license: other
license_name: mixed-third-party
license_link: LICENSE
tags:
  - robotics
  - simulation
  - usd
  - isaac-sim
  - isaac-lab
  - humanoid
size_categories:
  - 1K<n<10K
---

# Fiatlux benchmark assets

USD assets for the [Fiatlux](https://github.com/haw-ai-i/fiatlux) benchmark: a Unitree G1 humanoid
carries a step ladder, climbs it, and replaces a ceiling light bulb, in Isaac Sim 5.1 / Isaac Lab.
The benchmark code is Apache-2.0; **this dataset is a mirror of third-party content and each group
carries its own terms** (table below).

## How to use

Do not download this dataset by hand. The benchmark's `assets/download_assets.sh` pulls the groups a
task needs into the git-ignored `assets/` folder and applies the collision authoring the ladders
and the socket's guide-sleeve require:

```bash
git clone https://github.com/haw-ai-i/fiatlux && cd fiatlux
./assets/download_assets.sh                   # robot + task assets + room dressing (~4.4 GB)
# -- or, to also get the opt-in dressing groups (run only one of these two): --
./assets/download_assets.sh --scene-dressing
```

## Groups

| Group | Role in the benchmark | Files | Size |
|---|---|---|---|
| `unitree_g1/` | the robot: legged G1 with Inspire hands (default), Dex3 and gripper variants, plus fixed-base variants | 48 | 0.46 GB |
| `omniverse_bulb/` | the graspable bulb and the stock socket (`BULB_USD` / the base of `SOCKET_USD`); `download_assets.sh` generates the additive guide-sleeve layer locally, it isn't shipped here | 12 | <0.01 GB |
| `omniverse_ladder/` | 98 step ladders / platforms in three physics tiers; `AlumStep_D` is the one the benchmark climbs | 961 | 3.60 GB |
| `isaac_packing_table/`, `isaac_room/`, `isaac_skies/` | room dressing: the table the fresh bulb rests on, the room backdrop, the HDRI sky | 202 | 0.29 GB |
| `omniverse_climb/`, `omniverse_lamp/` | opt-in scene dressing: elevated platforms, residential lamps and fixtures | 1,212 | 3.25 GB |

Each ladder ships as `<name>.usd` (original geometry), `<name>_collision.usd` (static, climbable) and
`<name>_collision_rigid.usd` (movable); the three files sublayer one another and must stay together.
Per-asset provenance and physics-verification notes: `omniverse_uploaded_manifest.csv` and
`isaac_mirror_manifest.csv` in the benchmark repo.

## Provenance and terms

| Group(s) | Source | Terms |
|---|---|---|
| `unitree_g1/` | Unitree's pre-assembled G1 USDs, mirrored from [`unitreerobotics/unitree_sim_isaaclab_usds`](https://huggingface.co/datasets/unitreerobotics/unitree_sim_isaaclab_usds) | Apache-2.0 |
| `omniverse_*/` | NVIDIA [OpenUSD asset packs](https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html): Warehouse, Residential, SimReady Warehouse 01, Sample Scenes. The collision overlays, `_platform` variants and the bulb/socket split were authored by the Fiatlux team | NVIDIA [Product-Specific Terms for NVIDIA AI Products](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/) |
| `isaac_packing_table/`, `isaac_room/` | Isaac Sim 5.1 sample content (`Isaac/Props/PackingTable/`, `Isaac/Environments/Simple_Room/`) | [NVIDIA Isaac Sim Additional Software and Materials License](https://docs.isaacsim.omniverse.nvidia.com/latest/common/license-isaac-sim-additional.html) |
| `isaac_skies/` | `kloofendal_43d_clear_puresky_4k.hdr`, [Poly Haven](https://polyhaven.com/a/kloofendal_43d_clear_puresky) | CC0 |

The NVIDIA-sourced groups are mirrored here as part of the benchmark under NVIDIA's terms above;
NVIDIA's notice and the name of the source pack are being added to each group. The BEHAVIOR-1K
groups were not redistributable and have been removed from this dataset (Stanford distributes
those models encrypted for use inside OmniGibson only, and no benchmark preset has loaded them
since #232). Progress: [haw-ai-i/fiatlux#231](https://github.com/haw-ai-i/fiatlux/issues/231).

## Citation

If you use the benchmark, cite the paper (see the [repository README](https://github.com/haw-ai-i/fiatlux#readme)).
