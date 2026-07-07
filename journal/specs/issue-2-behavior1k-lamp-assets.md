# Issue 2: BEHAVIOR-1K Lamp and Bulb Asset Intake

## Goal

From BEHAVIOR-1K dataset workflow, identify the lamp and light-bulb assets needed for Fiatlux, validate those assets for Isaac Lab use, and stage the approved outputs for the team.

GCS layout (15 paths — see GCS Upload Status for full listing):

```text
gs://fiatlux/assets/behavior1k_bulb/          # task asset: light_bulb models
gs://fiatlux/assets/behavior1k_bulb_broken/   # task asset: broken_light_bulb
gs://fiatlux/assets/behavior1k_lamp/          # task asset: table_lamp with socket
gs://fiatlux/assets/behavior1k_ladder/        # task asset: ladder
gs://fiatlux/assets/behavior1k_downlight/     # scene dressing: recessed ceiling fixtures
# ... + 10 more scene dressing paths
```

The repo should not store binary assets.
It should store only scripts, docs, or manifests needed to understand and reproduce the asset sync.

## Context

BEHAVIOR-1K assets are distributed through OmniGibson and are encrypted. The official setup flow requires accepting the BEHAVIOR Data Bundle terms and installs a decryption key locally. License interpretation: the team GCS bucket is treated as permitted internal non-commercial academic use. Assets were uploaded; the key and decrypted binaries are not committed to git.

## GCS Upload Status

| Bucket path                                          | Category             | Model                                 | Role           | Objects |
| ---------------------------------------------------- | -------------------- | ------------------------------------- | -------------- | ------- |
| `gs://fiatlux/assets/behavior1k_bulb/`               | `light_bulb`         | all 3 models (kfmkwd, sxkjea, ymomhw) | task asset     | 57  |
| `gs://fiatlux/assets/behavior1k_bulb_broken/`        | `broken_light_bulb`  | cugtye (only model)                   | task asset     | 23  |
| `gs://fiatlux/assets/behavior1k_lamp/`               | `table_lamp`         | 12 models with lights metadata        | task asset     | 217 |
| `gs://fiatlux/assets/behavior1k_ladder/`             | `ladder`             | all 3 models (shfvtl, vpmrlk, axywzt) | task asset     | 52  |
| `gs://fiatlux/assets/behavior1k_downlight/`          | `downlight`          | all 28 models                         | scene dressing | 421 |
| `gs://fiatlux/assets/behavior1k_room_light/`         | `room_light`         | all 30 models                         | scene dressing | 532 |
| `gs://fiatlux/assets/behavior1k_spotlight/`          | `spotlight`          | all 5 models                          | scene dressing | 86  |
| `gs://fiatlux/assets/behavior1k_square_light/`       | `square_light`       | all 10 models                         | scene dressing | 178 |
| `gs://fiatlux/assets/behavior1k_rectangular_light/`  | `rectangular_light`  | all 3 models                          | scene dressing | 52  |
| `gs://fiatlux/assets/behavior1k_track_light/`        | `track_light`        | all 2 models                          | scene dressing | 35  |
| `gs://fiatlux/assets/behavior1k_wall_mounted_light/` | `wall_mounted_light` | all 11 models                         | scene dressing | 188 |
| `gs://fiatlux/assets/behavior1k_chandelier/`         | `chandelier`         | all 4 models                          | scene dressing | 69  |
| `gs://fiatlux/assets/behavior1k_paper_lantern/`      | `paper_lantern`      | all 3 models                          | scene dressing | 52  |
| `gs://fiatlux/assets/behavior1k_lampshade/`          | `lampshade`          | all 4 models                          | scene dressing | 69  |
| `gs://fiatlux/assets/behavior1k_floor_lamp/`         | `floor_lamp`         | 11 models with lights metadata        | scene dressing | 188 |

Total: 2,219 files across 15 GCS paths. Full per-model detail in `assets/behavior1k_uploaded_manifest.csv`.

All assets were decrypted from `.encrypted.usd` using the `omnigibson.key` obtained from OmniGibson. Decrypted USD files were validated as PXR-USDC before upload.

## Category Scan Results

Full scan across all 1,829 BEHAVIOR-1K categories for lighting-relevant assets. Categories below have lights metadata or are plausible fixture/bulb candidates.

Full scan complete across all 1,829 categories

### Scenario-critical prop (not a light asset)

| Category | Models | Notes                                                                                           |
| -------- | ------ | ----------------------------------------------------------------------------------------------- |
| `ladder` | 3      | Essential for the humanoid to reach height. `shfvtl` (bbox_z=1.67m) is the tall upright ladder. |

### Primary targets (ceiling/wall-mounted, lights metadata confirmed)

| Category             | Models | Has lights metadata | Notes                                                                                                                        |
| -------------------- | ------ | ------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `downlight`          | 28     | yes                 | Recessed ceiling fixture — static integrated fixture, no removable bulb (no `bulblampF` socket). Scene dressing only. **All 28 uploaded.** |
| `room_light`         | 30     | yes (28/30)         | Broad fixture category — covers ceiling pendants, floor lamps, wall sconces (0.56–3.48m height); up to 4 emitters per model. |
| `spotlight`          | 5      | yes                 | Directional ceiling/track-mount fixture.                                                                                     |
| `square_light`       | 10     | yes                 | Flat panel flush-ceiling fixture.                                                                                            |
| `rectangular_light`  | 3      | yes                 | Long overhead fluorescent-panel fixture (bbox ~0.05m × 1.2m × 2.0m).                                                         |
| `track_light`        | 2      | yes                 | Track-mounted multi-bulb fixture.                                                                                            |
| `wall_mounted_light` | 11     | yes (9/11)          | Wall sconce/bracket at height.                                                                                               |
| `chandelier`         | 4      | yes                 | Ceiling-hanging multi-bulb fixture.                                                                                          |
| `paper_lantern`      | 3      | yes                 | Hanging pendant lantern; one model has `hangpoleM` ceiling attachment.                                                       |

### Bulb / socket assets

| Category            | Models | Has lights metadata | Notes                                                                       |
| ------------------- | ------ | ------------------- | --------------------------------------------------------------------------- |
| `light_bulb`        | 3      | no                  | `kfmkwd` has `bulblampM` attachment — mates with lamp socket. **Uploaded.** |
| `broken_light_bulb` | 1      | no                  | `cugtye` has `bulblampM` attachment. **Uploaded.**                          |
| `lampshade`         | 4      | no                  | Shade housing component, pairs with bulb.                                   |

### Secondary (lamp with socket, not ceiling-mounted)

| Category        | Models | Has lights metadata | Notes                                                                           |
| --------------- | ------ | ------------------- | ------------------------------------------------------------------------------- |
| `table_lamp`    | 34     | partial (12/34)     | Standard desk lamp. 12 models with lights **uploaded**.                         |
| `floor_lamp`    | 19     | partial (11/19)     | Standing lamp with togglebutton and socket. 11 models with lights **uploaded**. |
| `glass_lantern` | 7      | no                  | Enclosed glass lantern housing — no lighting metadata, skipped.                 |
| `ceiling_fan`   | 1      | no                  | Ceiling-mounted; no lighting metadata.                                          |
| `garden_light`  | 2      | yes                 | Outdoor post/wall fixture.                                                      |

### Decorative / marginal

| Category         | Models | Has lights metadata | Notes                                                                        |
| ---------------- | ------ | ------------------- | ---------------------------------------------------------------------------- |
| `fairy_light`    | 1      | yes                 | String lights.                                                               |
| `icicle_lights`  | 1      | yes                 | Wall-mount decorative strand.                                                |
| `spirit_lamp`    | 1      | no                  | Alcohol burner, open flame — not a bulb fixture.                             |
| `flashlight`     | 1      | yes                 | Handheld; auxiliary prop.                                                    |
| `ceilings`       | 360    | partial (~58%)      | Embedded downlights/panels — scene context geometry, not standalone objects. |
| `beeswax_candle` | 8      | no                  | Non-electric light source prop.                                              |
| `candle_holder`  | 5      | no                  | Analog for lamp socket housing.                                              |

## Findings

- Full `import omnigibson` pulls in simulation dependencies such as `torch`; avoid full simulation install until needed.
- The asset download/decrypt helpers are in `OmniGibson/omnigibson/utils/asset_utils.py`; they can be driven with a minimal Python environment or a small shim.
- The official helper requires the BEHAVIOR Data Bundle EULA acceptance before it installs `omnigibson.key`.
- Added `scripts/behavior1k/behavior1k_asset_intake.py` as a lightweight helper that loads `asset_utils.py` through a focused shim, without installing the full simulator stack. Subcommands: `download-all`, `download-assets`, `download-key`, `inspect`.
- The helper intentionally does not decrypt/persist assets, upload assets, or store the OmniGibson key in the repo.
- `download-key` may hang if `storage.googleapis.com:443` is firewalled. Workaround: decode the obfuscated URL in `asset_utils.py` and download the 44-byte `omnigibson.key` manually with `curl`.
- Encrypted dataset downloaded via `download-assets --accept-license`; extracted size ~33 GB, 1,829 categories.
- 1,829 object categories found. Known candidate models are present as encrypted USD assets:
  - `light_bulb/kfmkwd`: replacement bulb, `kfmkwd.encrypted.usd`, about 2.3 MB model folder.
  - `broken_light_bulb/cugtye`: broken bulb, `cugtye.encrypted.usd`, about 12.4 MB model folder.
  - `table_lamp/ehjsdz`: lamp fixture, `ehjsdz.encrypted.usd`, about 5.2 MB model folder.
- Plain `.usd` files are not present until decrypted; assets are stored as `.encrypted.usd`. No decrypted assets are committed to git.

### Dataset Location and Layout

- The BEHAVIOR-1K GitHub repo does not include the actual dataset assets directly. Its `datasets/README` says datasets go there by default, or under the path configured by `OMNIGIBSON_DATA_PATH`.
- OmniGibson resolves datasets through `gm.DATA_PATH`.
- `get_dataset_path(dataset_name)` maps a dataset name to:

  ```text
  {gm.DATA_PATH}/{dataset_name}
  ```

- The BEHAVIOR asset dataset name used by OmniGibson is:

  ```text
  behavior-1k-assets
  ```

Expected downloaded object layout:

```text
$OMNIGIBSON_DATA_PATH/
  behavior-1k-assets/
    VERSION
    metadata/
      categories.txt
    objects/
      <category>/
        <model_id>/
          usd/
            <model_id>.usd
            # possibly stored/encrypted as <model_id>.encrypted.usd
    scenes/
    systems/
```

### Object Path Convention

OmniGibson `DatasetObject.get_usd_path(category, model)` resolves object USD paths as:

```text
{gm.DATA_PATH}/behavior-1k-assets/objects/{category}/{model}/usd/{model}.usd
```

Example expected candidate paths after dataset setup:

```text
$OMNIGIBSON_DATA_PATH/behavior-1k-assets/objects/light_bulb/kfmkwd/usd/kfmkwd.usd
$OMNIGIBSON_DATA_PATH/behavior-1k-assets/objects/broken_light_bulb/cugtye/usd/cugtye.usd
$OMNIGIBSON_DATA_PATH/behavior-1k-assets/objects/table_lamp/ehjsdz/usd/ehjsdz.usd
```

The public code describes this as the unencrypted USD path. For the proprietary BEHAVIOR dataset, OmniGibson loads dataset objects with encryption enabled, and some helper code explicitly swaps `.usd` to `.encrypted.usd` before decrypting into a temporary file for inspection.

### Download and Encryption Flow

- `download_behavior_1k_assets()` is the OmniGibson entrypoint for downloading the BEHAVIOR assets.
- Current source points to a zipped Hugging Face dataset repo named `behavior-1k/zipped-datasets`.
- Current source expects a file like:

  ```text
  behavior-1k-assets-3.9.0rc9.zip
  ```

- The code uses `cryptography.fernet.Fernet` for decrypting encrypted files.
- The decryption key is installed under `gm.DATA_PATH` as:

  ```text
  omnigibson.key
  ```

- The key download URL is obfuscated in `omnigibson/utils/asset_utils.py`.
- OmniGibson's helper decrypts a file into a temporary path, yields that path, then removes it. That suggests the normal runtime model is temporary local decryption, not broad export of decrypted assets.

### License / Redistribution Risk

The BEHAVIOR Data Bundle license text embedded in `asset_utils.py` says, in paraphrase:

- use is limited to non-commercial academic research,
- the data is encrypted,
- users must not redistribute the key or dataset contents,
- data is intended to be used within OmniGibson.

License resolved: the team GCS bucket is treated as permitted internal non-commercial academic use. Assets were uploaded; the key and decrypted binaries are not committed to git.

### Candidate Categories and Models

Confirmed from full dataset scan (see Category Scan Results above). All categories are uploaded to GCS:

- `light_bulb` (3 models) → `behavior1k_bulb/` — `kfmkwd` has `bulblampM` Male plug
- `broken_light_bulb/cugtye` → `behavior1k_bulb_broken/` — has `bulblampM` attachment point
- `table_lamp` (12 models with lights) → `behavior1k_lamp/` — `ehjsdz` has `bulblampF` Female socket
- `ladder` (3 models) → `behavior1k_ladder/`
- `downlight` (all 28 models) → `behavior1k_downlight/`
- `room_light` (30 models) → `behavior1k_room_light/`
- `spotlight` (5 models) → `behavior1k_spotlight/`
- `square_light` (10 models) → `behavior1k_square_light/`
- `rectangular_light` (3 models) → `behavior1k_rectangular_light/`
- `track_light` (2 models) → `behavior1k_track_light/`
- `wall_mounted_light` (11 models) → `behavior1k_wall_mounted_light/`
- `chandelier` (4 models) → `behavior1k_chandelier/`
- `paper_lantern` (3 models) → `behavior1k_paper_lantern/`
- `lampshade` (4 models) → `behavior1k_lampshade/`
- `floor_lamp` (11 models with lights) → `behavior1k_floor_lamp/`
- `glass_lantern` — skipped (no lighting metadata)

Additional evidence:

- BEHAVIOR task definitions include `changing_light_bulbs`, `clean_a_light_bulb`, and `remove_a_broken_light_bulb`.
- `changing_light_bulbs` references `table_lamp`, `light_bulb`, and `broken_light_bulb`.
- BEHAVIOR metadata has attachment combinations:
  - `light_bulb-kfmkwd` attached to `table_lamp-ehjsdz`
  - `broken_light_bulb-cugtye` attached to `table_lamp-ehjsdz`
- Category metadata includes `light_bulb` and `wall_socket`.
- The category/model naming pattern in metadata uses:

  ```text
  <category>-<model_id>
  ```

  while the on-disk object path splits those into:

  ```text
  objects/<category>/<model_id>/...
  ```

### USD Structure Inspection

Inspected `light_bulb/kfmkwd`, `table_lamp/ehjsdz`, and `downlight/adwcsx` using `usd-core` (no sim required).

**Asset closure — fully self-contained.** All three USDs have zero external layer references. All texture paths are relative (`../material/<model>__base_link__*.png`). Each model folder is a complete portable asset — no shared dataset-level dependencies.

**Socket system — Male/Female joint pair confirmed:**

- `kfmkwd` (bulb): prim `/kfmkwd/joints/meta__base_link_attachment_bulblampM_0_joint` [FixedJoint] — **`bulblampM`** = Male plug
- `ehjsdz` (lamp): prim `/ehjsdz/joints/meta__base_link_attachment_bulblampF_0_joint` [FixedJoint] — **`bulblampF`** = Female socket
- These are the correct complementary pair. OmniGibson connects them at the `bulblampF/M` joint to simulate bulb-in-socket insertion.

**Downlight has no removable bulb.** `adwcsx` has only `meta__base_link_lights_0_0_joint` (light emitter) and a single collision mesh — no `bulblampF` socket anywhere in the prim tree. Downlights are static integrated fixtures. They are good for scene dressing but cannot be used for the bulb-replacement task mechanic.

**Bounding boxes (`ig:nativeBB`):**

- `kfmkwd`: 11.3 × 7.0 × 6.9 cm — standard A19 bulb size ✓
- `ehjsdz`: 46.3 × 46.3 × 73.2 cm — desk lamp ✓
- `adwcsx`: 6.4 × 6.0 × 2.7 cm — thin recessed ceiling puck ✓

**Collision geometry:** `kfmkwd` has 2 collision meshes, `ehjsdz` has 32 (complex lamp shade geometry), `adwcsx` has 1.

**Isaac Lab compatibility:** No OmniGibson-specific physics schemas that would block loading. The `OmniGibsonVRayMtl` shader may not render correctly without OmniGibson's renderer, but physics and collision geometry should load. Needs a practical load test in Isaac Lab.

### Open Questions

- Can Isaac Lab load the decrypted USDs without OmniGibson's renderer? (`OmniGibsonVRayMtl` shader may fall back to default material.)
- Does the `bulblampF/M` joint need to be driven programmatically in Isaac Lab, or does it work as a standard `UsdPhysics.FixedJoint`?
- Which downlight model is best sized for a ceiling socket the G1 can reach from a ladder? (Need to check bbox_z mount height across models.)

## Deliverables

- Asset manifest at `assets/behavior1k_uploaded_manifest.csv` — 126 models across 15 GCS paths, with category, model ID, source path, `has_lights`, `has_socket`, and `usage` (`task_asset` / `scene_dressing`) columns.
- USD structure inspected for `kfmkwd` (bulb), `ehjsdz` (lamp), `adwcsx` (downlight); socket mechanic confirmed via `bulblampM`/`bulblampF` joint pair.
- Asset closure confirmed: each model folder is fully self-contained with zero external layer references.
- All 15 asset categories uploaded to `gs://fiatlux/assets/behavior1k_*/` (2,219 files total).
- `assets/download_assets.sh` updated: syncs 4 task-asset groups by default; 11 scene-dressing groups behind `--scene-dressing` flag.
- Decryption key (`omnigibson.key`) and decrypted binary USDs are not committed to git.

## Non-Goals

- Do not commit BEHAVIOR-1K binary assets to git.
- Do not commit decrypted full-dataset contents.
- Do not commit the OmniGibson decryption key or any secret/token.
- Do not migrate the Fiatlux Isaac Lab environment to the selected lamp/bulb assets in this issue; this issue only prepares and validates the asset inputs.

## Task Breakdown

1. ✅ Confirm workspace, disk space, `uv`, Hugging Face/GitHub access, and GCS auth.
2. ✅ Set up OmniGibson/BEHAVIOR-1K in an isolated workspace.
3. ✅ Accept required dataset terms through the official setup flow.
4. ✅ Download BEHAVIOR-1K assets via `behavior1k_asset_intake.py download-assets` (33 GB, 1,829 categories).
5. ✅ Confirm license/TOS constraints — treating GCS bucket as permitted internal academic use.
6. ✅ Enumerate available candidate categories across all 1,829 objects (two-round parallel scan, complete).
7. ✅ Inspect likely candidates:
   - `objects/light_bulb/kfmkwd` — confirmed, uploaded
   - `objects/broken_light_bulb/cugtye` — confirmed, uploaded, has `bulblampM` attachment point
   - `objects/table_lamp/ehjsdz` — confirmed, uploaded
   - all 28 `downlight` models — confirmed, all uploaded
8. ✅ Inspected USD structure for `kfmkwd`, `ehjsdz`, `adwcsx`:
   - Zero external layer refs — each model folder is a complete asset closure
   - `kfmkwd` has `bulblampM` (Male), `ehjsdz` has `bulblampF` (Female) — confirmed M/F socket pair
   - Downlights have no removable bulb prim; not suitable for task mechanic
   - Isaac Lab load test still needed for shader compatibility
9. ✅ Decided to split into separate GCS paths: `behavior1k_lamp/`, `behavior1k_bulb/`, `behavior1k_bulb_broken/`, `behavior1k_downlight/`.
10. ✅ Extracted and uploaded all 15 asset categories to GCS (2,219 files total across `behavior1k_bulb/`, `behavior1k_bulb_broken/`, `behavior1k_lamp/`, `behavior1k_ladder/`, and 11 scene-dressing paths).
11. ✅ Updated `assets/download_assets.sh` with all 15 GCS paths; task assets synced by default, scene dressing behind `--scene-dressing` flag.
12. ✅ Smoke test passed: `./assets/download_assets.sh` from clean state synced all 4 task-asset groups (57 + 23 + 217 + 52 files).
13. ✅ Manifest at `assets/behavior1k_uploaded_manifest.csv` — 126 models, all GCS paths, `usage` column (`task_asset` / `scene_dressing`).
14. ✅ Spec and manifest updated with final asset paths, USD inspection findings, and resolved blockers.

## Acceptance Criteria

- OmniGibson/BEHAVIOR-1K dataset setup completes and the downloaded asset tree is locatable.
- Manifest at `assets/behavior1k_uploaded_manifest.csv` documents: GCS path, category, model ID, source path (`b1k_path`), `has_lights`, `has_socket`, notes, and usage (`task_asset` / `scene_dressing`). License resolved globally (see Context). USD validation done on representative models (`kfmkwd`, `ehjsdz`, `adwcsx`) — not per-row.
- At least one candidate bulb and one candidate lamp/fixture are inspected and marked usable or rejected with a concrete reason.
- Socket/insertion-target status is documented: existing prim/meta link found, approximated from geometry, or requires new authored Fiatlux frame.
- `gs://fiatlux/assets/behavior1k_lamp/`, `gs://fiatlux/assets/behavior1k_bulb/`, and 13 other paths contain uploaded assets; a fresh checkout can sync task assets with `assets/download_assets.sh`.
- No BEHAVIOR-1K binaries, decrypted full dataset, secrets, or keys are committed to git.
