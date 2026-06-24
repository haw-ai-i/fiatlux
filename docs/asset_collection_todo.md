# Asset Collection TODO

This document tracks the required assets for the Fiatlux benchmark that need to be aggregated, converted (if necessary), and uploaded to the GCP bucket (`gs://fiatlux/assets`).

## Required Assets

### 1. Unitree G1 Robot (with Inspire Hands)
- **Source:** We have previously used the Unitree G1 with the necessary Inspire hands. These assets exist in our internal repositories/filesystems but need to be located.
- **TODO:**
  - [ ] Locate the existing Unitree G1 USD/URDF files with the Inspire hands attached.
  - [ ] Verify the assets load correctly in Isaac Lab.
  - [ ] Upload the final assets to `gs://fiatlux/assets/unitree_g1`.

### 2. Light Bulbs & Lamps
- **Source:** The `BEHAVIOR-1K` repository (`~/github/tmp/BEHAVIOR-1K`) contains high-quality light bulb and lamp assets.
- **TODO:**
  - [ ] Extract the relevant lamp and light bulb USDs or URDFs from the BEHAVIOR-1K dataset.
  - [ ] Ensure the light bulbs are separated as graspable, rigid body objects with correct collision meshes.
  - [ ] Verify the lamp sockets are properly defined for the light bulb insertion task.
  - [ ] Upload the final assets to `gs://fiatlux/assets/behavior1k_lamps`.

### 3. Ladders
- **Source:** The Omniverse USD ecosystem contains ladder assets.
- **TODO:**
  - [ ] Locate a suitable ladder asset from the Omniverse ecosystem (or generate one).
  - [ ] Verify the physics, collision meshes, and scale of the ladder in Isaac Lab (must support the robot climbing it).
  - [ ] Upload the final assets to `gs://fiatlux/assets/ladders`.

## GCP Bucket Information
- **Bucket:** `gs://fiatlux/assets`
- **Access:** The bucket is accessible to everyone involved in the project. 
- **Troubleshooting:** If you encounter an access error (e.g., 403 Forbidden) when trying to upload or download assets, please reach out to **molybog@hawaii.edu**.

## Next Steps
Once the assets are uploaded to the GCP bucket, run `./assets/download_assets.sh` to
sync them into the git-ignored `assets/` directory (`assets/unitree_g1/`,
`assets/bulb_socket/`, `assets/ladder/`). You do not need to check these assets into Git.
