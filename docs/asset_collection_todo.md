# Asset Collection TODO

This document tracks the required assets for the Fiatlux benchmark that need to be aggregated, converted (if necessary), and uploaded to the `haw-ai-i/fiatlux-assets` HF dataset.

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
  - [x] Uploaded all baseline variants to the dataset's `unitree_g1/` (see layout below).
  - [x] Wired the legged **wholebody Inspire** variant into `g1_bulb_env_cfg.py` as the env
        default (`G1_USD`) — keeps the legs so the same robot can later locomote/climb —
        with `G1_DEX3_USD` staged for an easy swap.

**Dataset layout** (`unitree_g1/` in `haw-ai-i/fiatlux-assets`, pulled by `download_assets.sh`).
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

### 2. Light Bulbs & Lamps — DONE
- **Source:** The Omniverse USD ecosystem — a graspable bulb and its guide-sleeve socket
  (`BULB_USD`/`SOCKET_USD`).
- **Done:**
  - [x] The bulb is a graspable, rigid body object with correct collision meshes.
  - [x] The lamp socket is defined for the light bulb insertion task.
  - [x] Uploaded to the dataset's `omniverse_bulb/`.

### 3. Ladders
- **Source:** The Omniverse USD ecosystem contains ladder assets.
- **TODO:**
  - [ ] Locate a suitable ladder asset from the Omniverse ecosystem (or generate one).
  - [ ] Verify the physics, collision meshes, and scale of the ladder in Isaac Lab (must support the robot climbing it).
  - [ ] Upload the final assets to the dataset's `ladders/`.

## Asset Dataset
- **Dataset:** `haw-ai-i/fiatlux-assets` on Hugging Face (private).
- **Access:** members of the `haw-ai-i` HF org. Log in with `hf auth login`, or set `HF_TOKEN` to a
  token scoped to the org.
- **Upload:** `hf upload haw-ai-i/fiatlux-assets <local_dir> <group>/ --repo-type dataset`.
- **Troubleshooting:** a 401/404 on download or upload means your account is not in the org yet --
  ask a `haw-ai-i` org admin to add you.

## Next Steps
Once the assets are uploaded to the dataset, run `./assets/download_assets.sh` to
sync them into the git-ignored `assets/` directory (`assets/unitree_g1/`,
`assets/bulb_socket/`, `assets/ladder/`). You do not need to check these assets into Git.
