# Omniverse Ladder Material Fixes

Detailed record for **issue 18**: the material/texture errors seen when
loading `assets/omniverse_ladder` in Isaac Sim
(via the playground) and how each was fixed. The ladders always *rendered*
correctly; these were mostly log errors from redundant/broken references. After the
fixes the material log is clean (0 texture / SimPBR / SdrShaderNode errors).

## Errors seen → root cause → fix

| Error (Isaac log) | Root cause | Fix | Reproducible? |
|---|---|---|---|
| `[UsdToMdl] … References an asset that can not be found: '…/Materials/Base/…png'` (hundreds) | Ladders reference base textures **twice**: via the MDL (paths fine → this is what rendered) **and** via redundant texture overrides baked on the shader/material using the pack's original **5-level** `../../../../../Materials/` path, which overshoots the flattened `Materials/`. `fix_materials` had only collapsed the mdl **module** path (`sourceAsset`), not these texture **inputs** — and some inputs are authored on the **Material** prim, not the Shader. | `fix_materials` extended to collapse `(../)+Materials/` on **any asset-valued attribute** (Shader *and* Material prim inputs), not just `sourceAsset`. | ✅ scripted |
| same, for `Aluminum_Brushed/…` and `MetalPainted_White…` (6 refs) | Source-asset **typos**: overrides name a texture with a wrong folder (`Aluminum_Brushed/` vs `Aluminum/`) or filename (`MetalPainted` vs `Metal_Painted`). The real texture exists under the correct name. | `pack_extract` reconciles them: copies the same-content file to the typo path, matched by normalized name (case + underscores stripped). | ✅ scripted |
| textures simply missing | `pack_extract` previously pulled only base `.mdl` files, not the `.png` textures or decals. | `pack_extract` extended to pull the ladder-scoped base textures (~63) + decal mdls/textures. (Also fixed a `Usd.Stage` GC bug that made the texture scoping silently return 0.) | ✅ scripted |
| `MDLC comp error C120: could not find module '…SimPBR'` / `'../../../materials/SimPBR.mdl' is Invalid` (SimReady tilt-and-roll ladders) | SimReady ladders reference the shared **SimPBR** material via a broken relative path `../../../materials/SimPBR.mdl`; that file isn't shipped. SimPBR ships with Isaac (`isaacsim/kit/mdl/core/Base/SimPBR.mdl`). | `fix_materials` rewrites any `…/SimPBR.mdl` (or `OmniPBR.mdl`) ref to the **bare module name**, which Isaac resolves from its own MDL library. | ✅ scripted |
| `Could not open asset … thumbnail_rigs/thumb_rig_prop.usd … /Tagging/ThumbRig` (SimReady ladders, 12) | SimReady originals carry a `/Tagging/ThumbRig` **payload** — NVIDIA's thumbnail-generation rig tooling, not shipped. **Warning only**, no render/physics impact. | `fix_materials` deactivates the `/Tagging` prim (`SetActive(False)`), dropping the dead payload. | ✅ scripted |

After the fixes the playground log has **0 `[Error]` lines and 0 ThumbRig warnings**; the only
remaining lines are benign engine notices (a mesh smooth-normal fallback; a GPU perf hint).

## Where the fixes were applied

- All fixes were applied to **`assets/omniverse_ladder/`** — the repo copy the playground
  loads (synced from the HF dataset, git-ignored binaries).
- Files changed: ~50 ladder USDs edited **in place** (path strings only); **6 texture `.png`
  files added** (the typo copies). Nothing deleted. `.mdl` count unchanged (343); base `.png`
  63 → 69.

## Reproducibility / propagation status

- **Fully scripted** — a fresh regen (`pack_extract` → `fix_materials`) reproduces every fix:
  texture extraction + typo reconcile (`scripts/omniverse/omniverse_pack_extract.py`); path
  collapse (incl. Material-prim inputs), SimPBR bare-name rewrite, and ThumbRig `/Tagging`
  strip (`scripts/omniverse/omniverse_ladder_fix_materials.py`).
- **Re-synced to the asset store:** the fixed set (edited USDs + 6 texture copies) was rsync'd to
  the original GCS bucket (67 objects / 521 MiB) and carried over into `haw-ai-i/fiatlux-assets`'s
  `omniverse_ladder/` in the move to Hugging Face, so `download_assets.sh` pulls the clean assets. This doc covers only the ladder material fix (the one with
  reproducible scripts).
