# Issue 18: Red / Wrong-Rendering Assets on Video Recordings

## Problem

Some assets render wrong in Isaac Sim and in the recorded videos — a flat / red-fallback,
untextured look — and/or flood the load log with material/texture "asset can not be found"
errors. The root cause is diagnosed **per asset family**: for the Omniverse ladders it was
material/texture references that broke when the assets were curated out of their source packs;
for the BEHAVIOR-1K assets it was a **missing shared material** (OmniGibson's `OmniGibsonVRayMtl`
mdl) plus, for some, broken texture paths. Both are fixed below. Notably the "red" is **not** a
camera/color-format problem — it's the renderer's fallback when a material fails to compile.

## Goal

Every task/scene asset loads with its real materials + textures and a clean log (0
material/texture errors), so recordings look correct.

## Status checklist

- ✅ **Omniverse ladders** — material/texture paths fixed + re-synced to GCS (detail below).
- ✅ **BEHAVIOR-1K lights / lamps / bulbs / ladders** — the shared OmniGibson material bundled and
  referenced by relative path; flat-layout texture paths fixed. Asset-side, no code; re-synced to
  GCS (detail below).
- ✅ **Video pipeline** (`viz.py`) — confirmed the camera color format was fine; the red was the
  missing material, not the recorder. No change needed.

---

## Omniverse ladders (done)

### Context

The ladders reference shared pack **MDL** material libraries and the packs' **texture**
files. In the pack's original deep folder layout those references resolved; the curated
flattened layout (`<root>/<design>/…`, materials at `<root>/Materials/`) broke them, so
Isaac logged a flood of "asset can not be found" errors. The ladders still *rendered*
(the mdl supplied the texture) — the errors were from redundant/broken duplicate
references — but the log noise buried genuine problems and the assets carried dead refs.

### Problem → fix (summary)

Five distinct issues, all resolved. Full error→cause→fix table in
[`docs/omniverse_material_fixes.md`](../../docs/omniverse_material_fixes.md).

| Symptom | Fix | Where |
|---|---|---|
| base `.png` textures + decals not extracted | pull ladder-scoped textures (~63) + decals | `pack_extract` |
| texture refs `../../../../../Materials/…` overshoot (incl. inputs on **Material** prims) | collapse `(../)+Materials/` on every asset attribute | `fix_materials` |
| 6 dead-ref textures (folder/name typos) | copy same-content file by normalized name | `pack_extract` |
| `SimPBR` core material via broken relative path | rewrite to bare module name (Isaac resolves it) | `fix_materials` |
| `/Tagging/ThumbRig` payload warnings (SimReady) | deactivate `/Tagging` | `fix_materials` |

Also fixed a `Usd.Stage` garbage-collection bug that made the texture scoping silently
return 0.

### Result

- The two-row playground loads `assets/omniverse_ladder` with **0 `[Error]` lines and 0
  ThumbRig warnings**; ladders render with real materials + textures.
- Every fix is **reproducible**: `scripts/omniverse/omniverse_pack_extract.py` →
  `scripts/omniverse/omniverse_ladder_fix_materials.py` regenerate a clean set from scratch —
  no manual steps.
- Asset changes: ~50 ladder USDs edited in place (paths + SimPBR + `/Tagging`); 6 texture
  `.png` files added (typo copies); nothing deleted. Base `.mdl` 343 (unchanged); base
  `.png` 63 → 69.
- **Re-synced to GCS.** The fixed `assets/omniverse_ladder` set was rsync'd back to
  `gs://fiatlux/assets/omniverse_ladder` (67 objects / 521 MiB), so a fresh
  `download_assets.sh` pulls the clean, working assets.


## BEHAVIOR-1K lights / lamps / bulbs / ladders (done)

### Context

Every BEHAVIOR-1K object binds OmniGibson's shared material **`OmniGibsonVRayMtl`** via
`info:mdl:sourceAsset = @omnigibson_vray_mtl.mdl@` — a bare module name that only resolves when
OmniGibson's `materials/` folder is on the MDL search path. The issue-2 intake copied only the
per-object USDs + textures, never that shared framework mdl, so the shader fails to compile and
Isaac renders every object a flat **red** fallback. Separately, 31 "flat-layout" objects
(`<id>/<id>.usd` instead of `<id>/usd/<id>.usd`) had `../material/` texture paths pointing one
directory too high, so those also lost their textures.

Confirmed in the actual task scene: `FIATLUX-Insert-v0` (G1 + socket-lamp on the packing table)
rendered the lamp — the central prop — as a solid red blob.

### Problem → fix (summary)

| Symptom | Fix | Where |
|---|---|---|
| all objects red — missing `omnigibson_vray_mtl` mdl | bundle the 3 OmniGibson mdls (`omnigibson_vray_mtl` + `vray_maps` + `vray_materials`) | `assets/behavior1k_materials/` |
| shader binds the mdl by bare name (needs a search path) | rewrite `sourceAsset` → **relative path** to the bundle, so it resolves on its own | `behavior1k_fix_mdl_path.py` |
| 31 flat-layout objects miss their textures | rewrite `../material/` → `material/` | `behavior1k_fix_flat_texpaths.py` |

The mdl is kept **original** (OmniGibson's own material, not rebound to OmniPBR); the 3 mdls are
attributed in `assets/behavior1k_materials/NOTICE.md`.

### Result

- **Fully asset-side / permanent** — the relative mdl path + bundled mdls make each object
  self-contained. **No `MDL_USER_PATH`, no launcher/entry-script wiring, no code changes** — works
  in the playground, `verify_scene`, `record_run`, or any tool that opens the asset.
- **Validated in-scene:** re-recorded `FIATLUX-Insert-v0` with no wiring and no env var → **0 vray
  errors**, the lamp renders its real material (before/after posters kept outside the repo).
- Reproducible: `behavior1k_fix_mdl_path.py` + `behavior1k_fix_flat_texpaths.py` (both idempotent,
  dry-run/`--apply`); `behavior1k_playground.py` lays out all 127 lighting assets to eyeball them.
- Asset changes: **137 mdl `sourceAsset`s** rewritten across 130 objects + **224 texture paths**
  across 31 objects (path strings only — verified nothing else changed); 3 mdls added under
  `behavior1k_materials/`. Nothing deleted.
- **Re-synced to GCS:** the 130 edited USDs + the `behavior1k_materials/` bundle were rsync'd to
  `gs://fiatlux/assets/`, and `behavior1k_materials` was added to `download_assets.sh` (task set),
  so a fresh `download_assets.sh` pulls the working assets + the shared mdls.

### Addendum: 3 flat-layout objects missed by the original re-sync (2026-07-07)

`behavior1k_fix_flat_texpaths.py` already handles this bug class, but the mdl-path fix's re-sync
(above) didn't include every flat-layout object -- `behavior1k_lamp/ehjsdz`, `behavior1k_bulb/
kfmkwd`, and `behavior1k_bulb_broken/cugtye` (the task's actual bulb + lamp + broken-bulb assets)
still had `../material/` texture paths after a fresh `download_assets.sh`, confirmed by re-running
`FIATLUX-Insert-v0`: the mdl resolved (no more red fallback) but every texture logged "asset can
not be found" and the lamp rendered flat grey/untextured.

Ran `behavior1k_fix_flat_texpaths.py --apply` locally (28 texture paths across these 3 files),
confirmed 0 errors and real materials (white lampshade, dark metal neck, white bulb) in a
re-recorded `FIATLUX-Insert-v0`, and `gsutil cp`'d the 3 fixed USDs back to their `gs://fiatlux/
assets/` paths (targeted single-file copies, not a directory rsync). Round-trip verified: a fresh
download of `ehjsdz.usd` from GCS now carries the corrected `material/...` path.


## Task Breakdown

1. ✅ Diagnose the ladder material/texture errors from the Isaac (kit) log.
2. ✅ Extend `pack_extract` to pull scoped base textures + decals; reconcile typo dead-refs.
3. ✅ Extend `fix_materials` to collapse deep `Materials/` paths on all asset attributes
   (Shader + Material prims), rewrite core mdls (SimPBR/OmniPBR) to bare name, and drop the
   `/Tagging` ThumbRig payload.
4. ✅ Apply to `assets/omniverse_ladder`; verify 0 errors + 0 ThumbRig warnings in Isaac.
5. ✅ Record in `docs/omniverse_material_fixes.md`; re-sync the fixed set to GCS.
6. ✅ Validate the full pipeline from a fresh download (before-fix 53 errors → after-fix 0).
7. ✅ Diagnose BEHAVIOR-1K red: all objects bind OmniGibson `OmniGibsonVRayMtl` whose mdl was never
   shipped → red fallback; confirmed in `FIATLUX-Insert-v0`. Camera color format was fine.
8. ✅ Bundle the 3 OmniGibson mdls into `assets/behavior1k_materials/` (+ attribution NOTICE).
9. ✅ Bake the relative mdl path into all 130 BEHAVIOR-1K objects (`behavior1k_fix_mdl_path.py`);
   fix the 31 flat-layout texture paths (`behavior1k_fix_flat_texpaths.py`).
10. ✅ Confirm asset-side fix in-scene — no wiring/env var, 0 vray errors, lamp renders real.
11. ✅ Re-synced the edited BEHAVIOR-1K USDs + the `behavior1k_materials/` bundle to GCS; added
    `behavior1k_materials` to `download_assets.sh` (task set).
