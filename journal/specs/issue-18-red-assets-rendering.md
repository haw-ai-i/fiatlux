# Issue 18: Red / Wrong-Rendering Assets on Video Recordings

## Problem

Some assets render wrong in Isaac Sim and in the recorded videos — a flat / red-fallback,
untextured look — and/or flood the load log with material/texture "asset can not be found"
errors. The root cause is diagnosed **per asset family**: for the Omniverse ladders it was
material/texture references that broke when the assets were curated out of their source packs
(fixed below). Other families are diagnosed as we get to them — the **BEHAVIOR-1K bulbs have
not been investigated yet**, so their cause is still unknown.

## Goal

Every task/scene asset loads with its real materials + textures and a clean log (0
material/texture errors), so recordings look correct.

## Status checklist

- ✅ **Omniverse ladders** — fixed + re-synced to GCS (full detail below).
- ⬜ **BEHAVIOR-1K light bulbs** (`assets/behavior1k_bulb*`) — **← next.** Problem not yet
  identified: check whether/how they misrender, diagnose the cause, then fix + re-sync.
- ⬜ **Video pipeline** (`source/fiatlux_task/fiatlux_task/viz.py`) — confirm the camera color
  format is correct once the asset materials are clean.

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
  `download_assets.sh` pulls the clean, working assets. (This issue documents the ladder
  fix, which has reproducible scripts; other asset families are tracked separately.)


## Task Breakdown

1. ✅ Diagnose the ladder material/texture errors from the Isaac (kit) log.
2. ✅ Extend `pack_extract` to pull scoped base textures + decals; reconcile typo dead-refs.
3. ✅ Extend `fix_materials` to collapse deep `Materials/` paths on all asset attributes
   (Shader + Material prims), rewrite core mdls (SimPBR/OmniPBR) to bare name, and drop the
   `/Tagging` ThumbRig payload.
4. ✅ Apply to `assets/omniverse_ladder`; verify 0 errors + 0 ThumbRig warnings in Isaac.
5. ✅ Record in `docs/omniverse_material_fixes.md`; re-sync the fixed set to GCS.
6. ✅ Validate the full pipeline from a fresh download (before-fix 53 errors → after-fix 0).
7. ⬜ Investigate the BEHAVIOR-1K light bulbs (`assets/behavior1k_bulb*`) — diagnose whether/how
   they misrender (cause not yet identified); fix and re-sync if a problem is found.
8. ⬜ Confirm the video camera color format (`viz.py`) once asset materials are clean.
