# Fiatlux Assets

USD assets for the Fiatlux benchmark (Unitree G1 humanoid replacing a light bulb).
Assets are stored in GCS and synced locally via `download_assets.sh`. Binary USD files are git-ignored.

## Download

```bash
# Robot + task assets + room dressing (table, warehouse backdrop, HDRI sky)
./assets/download_assets.sh

# ... + BEHAVIOR-1K lighting-fixture scene dressing
./assets/download_assets.sh --scene-dressing
```

Room dressing (`isaac_*` paths below) is always synced -- it's what every recorded
run looks like by default -- unlike the opt-in BEHAVIOR-1K lighting fixtures.

Requires `gsutil` (`gcloud` SDK). Override bucket with `FIATLUX_ASSET_BUCKET` env var.

## GCS Paths

| GCS path                                              | Category              | Models                          | Role           | Files |
| ----------------------------------------------------- | --------------------- | ------------------------------- | -------------- | ----- |
| `gs://fiatlux/assets/behavior1k_bulb/`                | `light_bulb`          | 3 (kfmkwd, sxkjea, ymomhw)     | task asset     | 57    |
| `gs://fiatlux/assets/behavior1k_bulb_broken/`         | `broken_light_bulb`   | 1 (cugtye)                      | task asset     | 23    |
| `gs://fiatlux/assets/behavior1k_lamp/`                | `table_lamp`          | 12 with lights metadata         | task asset     | 217   |
| `gs://fiatlux/assets/behavior1k_ladder/`              | `ladder`              | 3 (shfvtl, vpmrlk, axywzt)     | task asset     | 52    |
| `gs://fiatlux/assets/behavior1k_downlight/`           | `downlight`           | 28                              | scene dressing | 421   |
| `gs://fiatlux/assets/behavior1k_room_light/`          | `room_light`          | 30                              | scene dressing | 532   |
| `gs://fiatlux/assets/behavior1k_spotlight/`           | `spotlight`           | 5                               | scene dressing | 86    |
| `gs://fiatlux/assets/behavior1k_square_light/`        | `square_light`        | 10                              | scene dressing | 178   |
| `gs://fiatlux/assets/behavior1k_rectangular_light/`   | `rectangular_light`   | 3                               | scene dressing | 52    |
| `gs://fiatlux/assets/behavior1k_track_light/`         | `track_light`         | 2                               | scene dressing | 35    |
| `gs://fiatlux/assets/behavior1k_wall_mounted_light/`  | `wall_mounted_light`  | 11                              | scene dressing | 188   |
| `gs://fiatlux/assets/behavior1k_chandelier/`          | `chandelier`          | 4                               | scene dressing | 69    |
| `gs://fiatlux/assets/behavior1k_paper_lantern/`       | `paper_lantern`       | 3                               | scene dressing | 52    |
| `gs://fiatlux/assets/behavior1k_lampshade/`           | `lampshade`           | 4                               | scene dressing | 69    |
| `gs://fiatlux/assets/behavior1k_floor_lamp/`          | `floor_lamp`          | 11 with lights metadata         | scene dressing | 188   |

**BEHAVIOR-1K total: 2,219 files across 15 paths.**

### Omniverse asset packs (NVIDIA SimReady / ArchVis)

| GCS path                              | Category         | Models                                 | Role           | Size   |
| ------------------------------------- | ---------------- | -------------------------------------- | -------------- | ------ |
| `gs://fiatlux/assets/omniverse_ladder/` | `ladder`       | 98 ladders/platforms (16 designs)      | task asset     | 3.4 GB |
| `gs://fiatlux/assets/omniverse_bulb/`   | `light_bulb`   | 1 (separable LightBulb)                | task asset     | small  |
| `gs://fiatlux/assets/omniverse_climb/`  | `climb`        | Mezzanine_A + OfficeSet_A              | scene dressing | 1.9 GB |
| `gs://fiatlux/assets/omniverse_lamp/`   | `lamp_fixture` | residential lamps/chandeliers/fixtures | scene dressing | 1.2 GB |

**Omniverse total: ~6.5 GB across 4 paths.** Each ladder is authored in **three physics tiers**
beside the original geometry — the scene builder just references the one it wants:

| File | Physics | Behaviour |
|---|---|---|
| `<name>.usd` (original) | none | geometry only (robot clips through) — visual prop |
| `<name>_collision.usd` | collision, static | **solid + fixed** — climbable, never moves |
| `<name>_collision_rigid.usd` | collision + rigid body | **solid + movable** — climbable *and* can tip / be carried |

The static `_collision.usd` files (convex decomposition, high friction) are in GCS and all 98 are
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

## Isaac Sim Nucleus mirror (room dressing)

The table, room backdrop, and HDRI sky are mirrored once from Isaac Sim 5.1's own
Nucleus content library (a public HTTPS/S3 endpoint, no Omniverse client needed)
into our own bucket, so nothing is fetched live from NVIDIA's CDN at sim launch.

| GCS path                                                        | Nucleus source                          | Role               | Files |
| ---------------------------------------------------------------- | ---------------------------------------- | ------------------ | ----- |
| `gs://fiatlux/assets/isaac_packing_table/`                      | `Isaac/Props/PackingTable/`             | table for lamp/bulb | 112   |
| `gs://fiatlux/assets/isaac_room/Environments/Simple_Room/`      | `Isaac/Environments/Simple_Room/`       | room backdrop (walls/floor/windows) | 89 |
| `gs://fiatlux/assets/isaac_skies/`                              | `Isaac/Materials/Textures/Skies/PolyHaven/` | HDRI dome light | 1     |

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

**BEHAVIOR-1K** assets are from [BEHAVIOR-1K](https://behavior.stanford.edu/) (Stanford OmniGibson dataset),
decrypted from `.encrypted.usd` using the `omnigibson.key`. The key and decrypted binaries are not committed to git.
See `journal/specs/issue-2-behavior1k-lamp-assets.md` for intake details and USD inspection findings.

**Omniverse** assets are from the NVIDIA [Omniverse downloadable USD packs](https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html#d-openusd-asset-packs)
(Warehouse, SimReady Warehouse 01, Residential, Sample Scenes). Used under non-commercial
academic terms. See `docs/omniverse_pack_scan_log.md` for the full 14-pack scan and
`docs/collision_authoring_explained.md` for collision authoring.

**Room-dressing** assets (table, room, sky) are NVIDIA's own Isaac Sim sample content
(Props/Environments) plus a CC0 PolyHaven HDRI, mirrored per the table above.
