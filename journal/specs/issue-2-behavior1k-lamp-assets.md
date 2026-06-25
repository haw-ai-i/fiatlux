# Issue 2: BEHAVIOR-1K Lamp and Bulb Asset Intake

## Goal

Use DGX to set up the official OmniGibson/BEHAVIOR-1K dataset workflow, identify the smallest lamp and light-bulb asset subset needed for Fiatlux, validate those assets for Isaac Lab use, and stage the approved outputs for the team.

Preferred GCS layout, pending license/redistribution approval:

```text
gs://fiatlux/assets/behavior1k_lamp/
gs://fiatlux/assets/behavior1k_bulb/
```

The repo should not store binary assets. It should store only scripts, docs, or manifests needed to understand and reproduce the asset sync.

## Context

`gs://fiatlux/assets/` is currently empty. Igor does not have a decrypted BEHAVIOR-1K checkout, so we need to perform the OmniGibson/BEHAVIOR-1K dataset setup on DGX.

BEHAVIOR-1K assets are distributed through OmniGibson and are encrypted. The official setup flow requires accepting the BEHAVIOR Data Bundle terms and installs a decryption key locally. Before uploading any decrypted asset files to GCS, we must confirm that sharing those files through the Fiatlux bucket is allowed under the dataset terms. If it is not allowed, this task should still produce a manifest and extraction/inspection workflow, but the assets should remain in the approved local dataset location.

## Findings So Far

These are repo/code findings from BEHAVIOR-1K reconnaissance before running the DGX setup.

## Execution Notes

Current branch:

```text
issue-2-behavior1k-lamp-assets
```

DGX access:

```text
host: srl-hawaii-1
user: pbushuyeu
home: /raid/home/pbushuyeu
workspace: /raid/home/pbushuyeu/fiatlux_issue2_behavior1k
```

DGX environment checks:

- SSH access works.
- `/raid` has about 1.9 TB free.
- GPUs visible: 8x Tesla V100-SXM2-32GB.
- `uv` is available in the login shell at `/raid/home/pbushuyeu/.local/bin/uv`.
- Hugging Face auth works via `uvx --from huggingface_hub hf auth whoami`; authenticated user is `bushuyeu`.
- `gcloud` and `gsutil` are not currently installed or visible in DGX PATH.
- System Python is Python 3.10.12.
- No pre-existing BEHAVIOR-1K / OmniGibson dataset checkout was found under `/raid/home/pbushuyeu` before this task.

BEHAVIOR-1K checkout:

```text
/raid/home/pbushuyeu/fiatlux_issue2_behavior1k/BEHAVIOR-1K
```

- Shallow clone completed.
- Current commit: `38af119`.
- Checkout size: about 976 MB.

Minimal import test:

- Full `import omnigibson` pulls in simulation dependencies such as `torch`.
- We should avoid full simulation install until needed.
- The asset download/decrypt helpers are in `OmniGibson/omnigibson/utils/asset_utils.py`; they can likely be driven with a minimal Python environment or a small shim.
- The official helper requires the BEHAVIOR Data Bundle EULA acceptance before it installs `omnigibson.key`.
- Added `scripts/behavior1k_asset_intake.py` as a lightweight helper that loads `asset_utils.py` through a focused shim, without installing the full simulator stack.
- The helper exposes:
  - `download-all`: accept EULA, install `omnigibson.key`, then download the encrypted asset zip.
  - `download-assets`: accept EULA and download only the encrypted asset zip, using a persistent Hugging Face `local_dir`.
  - `download-key`: accept EULA and install only `omnigibson.key`.
  - `inspect`: scan the local encrypted dataset layout and write a JSON manifest without requiring a BEHAVIOR-1K repo checkout.
- The helper intentionally does not decrypt/persist assets, upload assets, or store the OmniGibson key in the repo.
- First `download-key` attempts hung inside the official `asset_utils.download_key()` call. A follow-up bounded attempt after the asset download also timed out after 90 seconds; no `omnigibson.key` was created yet.
- `download-assets --accept-license --download-dir /tmp/tmpshjpkxua` completed on DGX and extracted the encrypted dataset into `/raid/home/pbushuyeu/fiatlux_issue2_behavior1k/omnigibson_data`.
- Extracted dataset size is about 33 GB at `/raid/home/pbushuyeu/fiatlux_issue2_behavior1k/omnigibson_data/behavior-1k-assets`.
- The temporary Hugging Face download directory `/tmp/tmpshjpkxua` was removed after extraction and manifest copy; DGX root filesystem returned to about 98 GB free.
- Offline manifest generation completed at `/raid/home/pbushuyeu/fiatlux_issue2_behavior1k/outputs/behavior1k_asset_manifest.json` and was copied into this repo as `journal/specs/issue-2-behavior1k-asset-manifest.json`.
- Manifest result: 1,829 object categories found. Known candidate models are present as encrypted USD assets:
  - `light_bulb/kfmkwd`: replacement bulb, `kfmkwd.encrypted.usd`, about 2.3 MB model folder.
  - `broken_light_bulb/cugtye`: broken bulb, `cugtye.encrypted.usd`, about 12.4 MB model folder.
  - `table_lamp/ehjsdz`: lamp fixture, `ehjsdz.encrypted.usd`, about 5.2 MB model folder.
- Plain `.usd` files are not present for those candidates; the expected files are `.encrypted.usd`. No decrypted assets were persisted or uploaded.

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

This is a hard gate for this issue. Before uploading decrypted BEHAVIOR files to `gs://fiatlux/assets`, confirm that the Fiatlux team bucket counts as permitted internal project use. If this is not clearly permitted, do not upload decrypted BEHAVIOR assets to GCS.

### Candidate Categories and Models

Known useful category/model hints from BEHAVIOR metadata and code search:

- `light_bulb/kfmkwd`
- `broken_light_bulb/cugtye`
- `table_lamp/ehjsdz`
- likely lamp categories: `table_lamp`, `floor_lamp`, `wall_mounted_light`, `room_light`, `downlight`
- possible fixture/socket-related category: `wall_socket`

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

The exact USD prim structure is unknown until the decrypted USD files are opened. In particular, we need to inspect whether a usable "socket" exists as a named prim, attachment meta link, collision mesh, or only as visual geometry embedded inside a lamp model.

### Open Questions for DGX Inspection

- Are the candidate USD files present exactly at the expected category/model paths?
- Are lamp and bulb assets independent enough to upload/version separately?
- Does `table_lamp/ehjsdz` include a socket-like prim, attachment point, or collision mesh?
- Are attachment/meta links usable as a bulb insertion target?
- Are textures and nested USD references local to each object model folder or shared elsewhere in the dataset?
- Can Isaac Lab load the decrypted USDs outside OmniGibson without missing OmniGibson-specific runtime setup?
- What is the smallest asset closure that loads cleanly?
- What license interpretation is approved for sharing asset subsets via the Fiatlux GCS bucket?

## Deliverables

- Asset manifest added to the repo with source category/model IDs, source paths, object roles, file format, license/TOS status, validation status, and intended GCS target.
- Inspection notes for candidate lamp, bulb, broken bulb, and socket/fixture assets.
- Minimal asset closure identified for each approved candidate: USDs, referenced textures, materials, collision meshes, and nested USD dependencies.
- If license permits redistribution: upload approved asset subset to `gs://fiatlux/assets/behavior1k_lamp/` and `gs://fiatlux/assets/behavior1k_bulb/`.
- If license does not permit redistribution: document the approved local DGX/OmniGibson usage path and do not upload decrypted assets to GCS.
- Notes or script for extraction/inspection workflow, excluding secrets, decryption keys, and decrypted full-dataset contents.
- Verification that `fiatlux_assets/scripts/download_assets.sh` pulls the uploaded files into `fiatlux_assets/models/`, if upload is approved.

## Non-Goals

- Do not commit BEHAVIOR-1K binary assets to git.
- Do not commit decrypted full-dataset contents.
- Do not commit the OmniGibson decryption key or any secret/token.
- Do not migrate the Fiatlux Isaac Lab environment to the selected lamp/bulb assets in this issue; this issue only prepares and validates the asset inputs.

## Task Breakdown

1. Confirm DGX workspace, disk, `uv`, Hugging Face/GitHub access, and GCS auth.
2. Install or run OmniGibson/BEHAVIOR-1K setup in an isolated DGX workspace.
3. Accept required dataset terms through the official setup flow.
4. Download BEHAVIOR-1K assets through the official OmniGibson workflow.
5. Confirm license/TOS constraints for uploading extracted/decrypted asset files to the Fiatlux GCS bucket.
6. Enumerate available candidate categories and models under `behavior-1k-assets/objects`.
7. Inspect likely candidates, starting with:
   - `objects/light_bulb/kfmkwd`
   - `objects/broken_light_bulb/cugtye`
   - `objects/table_lamp/ehjsdz`
   - additional `table_lamp`, `floor_lamp`, `wall_mounted_light`, `room_light`, and `downlight` candidates
8. For each candidate, inspect:
   - main USD path
   - dependency references
   - visual meshes
   - collision meshes
   - rigid/articulated body setup
   - scale and orientation
   - attachment/meta links
   - whether a socket/insertion target can be identified
9. Decide whether assets should be split into `behavior1k_lamp/` and `behavior1k_bulb/` or kept as one coupled package if USD dependencies are tangled.
10. If license permits upload, extract the smallest usable asset closure and upload it to:
    - `gs://fiatlux/assets/behavior1k_lamp/`
    - `gs://fiatlux/assets/behavior1k_bulb/`
11. If license does not permit upload, document the local DGX/OmniGibson path and stop before GCS upload.
12. Run `gsutil ls -r gs://fiatlux/assets/behavior1k_lamp/` and `gsutil ls -r gs://fiatlux/assets/behavior1k_bulb/`, if upload is approved, and save the listing in notes.
13. Run the Fiatlux asset download script locally or on DGX and confirm uploaded assets sync into `fiatlux_assets/models/`, if upload is approved.
14. Update repo docs/manifest with final asset paths, candidate decisions, validation notes, and any unresolved blockers.

## Acceptance Criteria

- DGX can run the official OmniGibson/BEHAVIOR-1K dataset setup and locate the downloaded asset tree.
- Manifest documents source category/model IDs, source paths, object roles, file paths, GCS targets if allowed, license/TOS status, and validation status.
- At least one candidate bulb and one candidate lamp/fixture are inspected and marked usable or rejected with a concrete reason.
- Socket/insertion-target status is documented: existing prim/meta link found, approximated from geometry, or requires new authored Fiatlux frame.
- If license permits upload, `gs://fiatlux/assets/behavior1k_lamp/` and `gs://fiatlux/assets/behavior1k_bulb/` contain usable assets and a fresh checkout can sync them with `fiatlux_assets/scripts/download_assets.sh`.
- If license does not permit upload, decrypted assets are not uploaded and the issue documents the approved alternative workflow.
- No BEHAVIOR-1K binaries, decrypted full dataset, secrets, or keys are committed to git.
