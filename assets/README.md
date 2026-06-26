# Fiatlux Assets

USD assets for the Fiatlux benchmark (Unitree G1 humanoid replacing a light bulb).
Assets are stored in GCS and synced locally via `download_assets.sh`. Binary USD files are git-ignored.

## Download

```bash
# Task assets only (bulb, lamp, ladder)
./assets/download_assets.sh

# Task assets + scene dressing
./assets/download_assets.sh --scene-dressing
```

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

**Total: 2,219 files across 15 paths.**

Full per-model detail (category, model ID, source path, has_lights, has_socket, usage) in [`behavior1k_uploaded_manifest.csv`](behavior1k_uploaded_manifest.csv).

## Source

All assets are from [BEHAVIOR-1K](https://behavior.stanford.edu/) (Stanford OmniGibson dataset),
decrypted from `.encrypted.usd` using the `omnigibson.key`. The key and decrypted binaries are not committed to git.
See `journal/specs/issue-2-behavior1k-lamp-assets.md` for intake details and USD inspection findings.
