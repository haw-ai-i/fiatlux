# Asset Collection TODO

This document tracks the required assets for the Fiatlux benchmark that need to be aggregated, converted (if necessary), and uploaded to the GCP bucket (`gs://fiatlux/assets`).

## Required Assets

### 1. Unitree G1 Robot (with Inspire Hands) — DONE
- **Source:** Unitree's **official** HuggingFace dataset
  [`unitreerobotics/unitree_sim_isaaclab_usds`](https://huggingface.co/datasets/unitreerobotics/unitree_sim_isaaclab_usds)
  (`assets.zip`, ~1.3 GB, Apache-2.0). The dexterous hands are attached **by Unitree at
  the URDF level** and converted to USD on their build machine (the embedded
  `config.yaml` shows `/home/unitree/.../g1withinspire_hand/...` + `UrdfConverter`), i.e.
  these are *not* an in-house assembly. All variants are full-body 29-DoF G1 (with legs).
  There are no standalone hand-only USDs in this dataset.
- **Done:**
  - [x] Pulled the official, pre-assembled G1 USDs (Inspire + Dex3 + gripper) — no local assets used.
  - [x] Verified joint/link names, actuator-group coverage (53 joints, disjoint, complete),
        and reference closure (self-contained, no stray external/local paths).
  - [x] Uploaded all baseline variants to `gs://fiatlux/assets/unitree_g1/` (see layout below).
  - [x] Wired the legged **wholebody Inspire** variant into `g1_bulb_env_cfg.py` as the env
        default (`G1_USD`) — keeps the legs so the same robot can later locomote/climb —
        with `G1_DEX3_USD` staged for an easy swap.

**Bucket layout** (`gs://fiatlux/assets/unitree_g1/`, pulled by `download_assets.sh`).
`base`: `free` = floating/legged base (can stand, walk, climb); `fixed` = pelvis welded
to the world (stationary manipulation only).

| group folder        | USD                                   | base   | hand    | wired |
|---------------------|---------------------------------------|--------|---------|-------|
| `wholebody_inspire/`| `g1_29dof_with_inspire_rev_1_0.usd`   | free   | Inspire | **default** |
| `wholebody_dex3/`   | `g1_29dof_with_dex3_rev_1_0.usd`      | free   | Dex3    | staged (`G1_DEX3_USD`) |
| `wholebody_dex1/`   | `g1_29dof_with_dex1_rev_1_0.usd`      | free   | gripper | baseline |
| `inspire/`          | `g1_29dof_with_inspire_rev_1_0.usd`   | fixed  | Inspire | baseline |
| `dex3/`             | `g1_29dof_with_dex3_base_fix.usd`     | fixed  | Dex3    | baseline |
| `gripper/`          | `g1_29dof_with_dex1_base_fix1.usd`    | fixed  | gripper | baseline |

- **Note / future (deferred assembly):** standalone hand-only assets are not in this
  dataset. The Inspire hand exists as URDF in `unitreerobotics/xr_teleoperate`
  (`assets/inspire_hand`) and Dex3 in Unitree's repos; converting/attaching those is a
  separate step and is **not** needed for the current benchmark.

### 2. Light Bulbs & Lamps — DONE (different source than planned)
- **Planned source:** the `BEHAVIOR-1K` dataset.
- **Shipped source:** the graspable bulb and its socket are the **curated Omniverse LightBulb**,
  split into `LightBulb_bulb_z_rigid.usda` + `LightBulb_socket_z_static_sleeve.usda` under
  `assets/omniverse_bulb/` — the B1K lamp renders with flat texture paths, and the Omniverse pair
  is authored assembled at identity, which is what makes "seated" a pose comparison. Paths and the
  measured seat/plug offsets are in `source/fiatlux_task/fiatlux_task/assets.py`.
- **No BEHAVIOR-1K asset is loaded by any task.** Its ceiling-mount categories were once an
  opt-in random `fixture` dressing entity; that entity and its asset pool were removed, so no
  scene entity, gate, reward or observation references them. The lighting categories are still
  synced by `download_assets.sh --scene-dressing` (with the required `behavior1k_materials`
  bundle) so they are available for scenes of your own — see `assets/README.md`.
- **Done:**
  - [x] Bulb and socket collected, wired, and graspable with authored collision.
  - [x] Seat/retention is the `mdp.bulb_attachment` state machine plus the authored guide sleeve
        (issue #171), not a raw collision fit — see `docs/task_spec.md`.
  - [x] Uploaded and synced by `assets/download_assets.sh`.

### 3. Ladders — DONE
- **Source:** the Omniverse USD ecosystem (`assets/omniverse_ladder/`, a pack of step ladders
  and work platforms). The BEHAVIOR-1K ladder was evaluated and dropped with the rest of that
  dataset.
- **Shipped:** `AlumStep_D` — `AluminumStepLadder_D01_PR_NVD_01_collision.usd`
  (0.608 x 0.979 x 1.861 m, cm-authored, spawn scale 0.01), with a `_rigid` overlay for the
  free-standing case. Mass is stamped in code (`LADDER_MASS_KG`), not read from the asset.
- **Done:**
  - [x] Selected the step ladder and verified scale and physics: all four treads hold a fitting
        probe with no tunnelling.
  - [x] Collision authored, not assumed — `download_assets.sh` runs
        `scripts/omniverse/omniverse_ladder_collision.py` / `_rigid.py` / `_platform.py`, because
        the stock convex decomposition omits the top tread and a robot placed there falls through.
  - [x] Uploaded and synced by `assets/download_assets.sh`.
- **Open:** the tread is 8.5 cm deep against a ~22 cm foot, which is the real obstacle to
  climbing — an asset fact, not a missing asset.

## GCP Bucket Information
- **Bucket:** `gs://fiatlux/assets`
- **Access:** Private to the project team. A 403 on download or upload means your
  `gcloud` account has not been granted access to the bucket; it is not a public
  endpoint.

## Status

Every asset this document tracked is collected, uploaded and wired in; it is kept as the record
of where each came from. Run `./assets/download_assets.sh` to sync them into the git-ignored
`assets/` directory — `setup_sim_teleop.sh` does it for you on first run. You do not need to
check these assets into Git.
