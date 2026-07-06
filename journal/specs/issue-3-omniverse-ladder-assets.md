# Issue 3: Omniverse Ladder (and Lamp/Bulb) Asset Intake

## Goal

Source climbing and lighting assets for the Fiatlux benchmark from the public NVIDIA
Omniverse USD asset packs, make the ladders physically climbable (PhysX collision), and
publish the curated set to GCS.

GCS layout (4 paths):

```
gs://fiatlux/assets/omniverse_ladder/   # task asset: 98 climb-ready ladders/platforms
gs://fiatlux/assets/omniverse_bulb/     # task asset: separable LightBulb
gs://fiatlux/assets/omniverse_climb/    # scene dressing: Mezzanine/OfficeSet climb structures
gs://fiatlux/assets/omniverse_lamp/     # scene dressing: residential lamps/chandeliers
```

The repo stores the docs, manifest, and reproduction scripts (`scripts/omniverse/`) needed to
understand and rebuild the asset set — never the binary USDs (those are pulled from GCS).

## Context

The NVIDIA [Omniverse downloadable USD packs](https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html#d-openusd-asset-packs)
are public (no auth) and used here under non-commercial academic terms. These are **rendering
content** — geometry only, **with no PhysX collision** — so collision had to be authored before
the ladders are usable for a climbing task.

## GCS Upload Status

| GCS path | Category | Models | Role | Size |
|---|---|---|---|---|
| `omniverse_ladder/` | ladder | 98 (16 designs) | task asset | 3.4 GB |
| `omniverse_bulb/` | light_bulb | 1 (LightBulb) | task asset | small |
| `omniverse_climb/` | climb | Mezzanine_A + OfficeSet_A | scene dressing | 1.9 GB |
| `omniverse_lamp/` | lamp_fixture | residential lamps/chandeliers | scene dressing | 1.2 GB |

**Total: ~6.5 GB across 4 paths** (ladder group includes the three physics tiers plus the
extracted base materials/textures). Full per-asset detail in
`assets/omniverse_uploaded_manifest.csv` (128 rows, per model; `state` = open/folded, `collision_verified`).

## Pack Scan Results

All 14 downloadable packs were scanned (`docs/omniverse_pack_scan_log.md`). Only 4 yielded
relevant assets:

- **Warehouse** → 86 ladder models (16 designs: extension, A-frame, scaffold, folding, step,
  tilt-roll, work platform) + Mezzanine/OfficeSet climb structures.
- **SimReady Warehouse 01** → the original 7 ladders + 4 work platforms.
- **Residential** → 30 lamps/fixtures found, 27 uploaded (2 ceiling/wall fixtures dropped) + 1 ladder.
- **Sample Scenes** → the **only separable LightBulb** found in any pack.

The other 10 packs (Furniture, Commercial, Industrial, Containers ×2, Data Center, Characters,
Showcase, Templates, SimReady Warehouse 02) held nothing for the benchmark — all fully
itemized in the scan log for future reference.

## Collision Authoring & Verification

- **Authored** (`scripts/omniverse/omniverse_ladder_collision.py`, no GPU): each ladder gets a
  `<name>_collision.usd` overlay applying `CollisionAPI` + `MeshCollisionAPI(convexDecomposition)`
  on every mesh, a high-friction static physics material; original USDs untouched. Handles both
  the Warehouse (cm, `/World`) and SimReady (m, instanceable, `/RootNode`) formats.
- **Verified** (one-off Isaac Sim drop-test, not committed; the in-repo playground is a visual
  check): **98/98 ladders confirmed
  climbable** — a solid collider is present across each structure. Details in
  `docs/collision_authoring_explained.md`.

## Physics tiers — static collision + rigid

Each ladder is authored in **three tiers** beside the original, so the scene builder just
references the one path whose behaviour it wants (no physics setup on their side):

| File | Physics | Behaviour |
|---|---|---|
| `<name>.usd` (original) | none | geometry only — visual prop |
| `<name>_collision.usd` | collision, static | solid + fixed — climbable, never moves |
| `<name>_collision_rigid.usd` | collision + rigid body | solid + movable — climbable *and* can tip / be carried |

- **Rigid** (`scripts/omniverse/omniverse_ladder_rigid.py`): a ~900-byte overlay per ladder that
  sublayers `<name>_collision.usd` and adds `RigidBodyAPI` + a size-based `MassAPI`
  (`round(clamp(2 + 3*height_m, 2, 40), 1)` kg). No geometry duplicated. 98/98 authored; sampled
  in Isaac (all fall/tip/settle under gravity → dynamic confirmed). CCD is set at the PhysX scene
  level, so no `PhysxSchema` dependency is baked into the asset.
- Because `_collision_rigid.usd` → `_collision.usd` → original is a sublayer chain, **the three files must
  stay together** when copied/uploaded.
- **Free-standing vs. lean-type.** Most rigid ladders self-stand (wide-based A-frames, step ladders,
  platforms). But **extension ladders and folded ladders can't free-stand** as rigid bodies — they
  need a wall (or other support) to lean on, exactly like in real life. A folded ladder leaned on a
  wall is still climbable. Scene builders using `_collision_rigid.usd` for these must supply support.
- **Demonstrated** by `scripts/omniverse/omniverse_ladder_playground.py`, a two-row Isaac Sim
  viewer (visual, GUI only): front row = static `_collision.usd` (a box drops on each step + one
  above the ladder), back row = dynamic `_collision_rigid.usd` (topple on play); extension/folded
  ladders use a wall — the front leans statically and holds a box, the back rigid one starts upright
  and topples onto the wall. Physics plays automatically; judged by eye.

## Findings

- **The `LightBulb` is geometry-only.** The Sample-Scenes `LightBulb` has a separable
  `/World/Geom/BulbGrp` (glass + screw base) distinct from the fixture, but **no socket joint** — to
  use it for a bulb-swap, a socket joint would need to be authored. Kept as a geometry-only spare.
- **Foldable ladders ship in two states.** 3 designs are `mixed` (AlumStep_A, FoldingStep_A,
  AlumMultiPurpose_A): odd variants are folded-flat, even are open. 10 of the 98 variants are
  folded; the manifest's `state` column records this. A folded ladder is **still climbable when
  leaned against a wall** (like an extension ladder) — not just a storage pose. The playground
  detects these tall/thin ladders and leans them on a wall in both rows (verified: they stay up).
- **Materials reference external pack libraries.** Ladders bind shared pack MDL libraries and the
  packs' textures; geometry + collision are complete regardless of materials.

## Deliverables

- Manifest `assets/omniverse_uploaded_manifest.csv` — 128 models across 4 GCS paths, with
  category, source pack, variants, units, `state`, `collision_verified`, and `usage`.
- All 98 ladders authored with convex-decomposition PhysX collision and verified in Isaac Sim.
- 4 GCS groups uploaded; `assets/download_assets.sh` updated to sync `omniverse_ladder`/
  `omniverse_bulb` (task) by default and `omniverse_climb`/`omniverse_lamp` behind `--scene-dressing`.
- Assets are reproducible from the raw packs by the scripts in `scripts/omniverse/`; re-verified
  by regenerating all three tiers + base materials from a fresh Warehouse download.
- Docs: full 14-pack scan log, collision-authoring explainer.
- Binary USDs are not committed to git (synced from GCS).

## Task Breakdown

1. ✅ Download + scan all 14 Omniverse packs; record the inventory.
2. ✅ Extract ladders, climb structures, lamps, and the LightBulb.
3. ✅ Author convex-decomposition PhysX collision on all 98 ladders (static).
4. ✅ Verify collision in Isaac Sim (98/98).
5. ✅ Author rigid (carry-able) variants on all 98 ladders; sampled dynamic in Isaac.
6. ✅ Upload the 4 `omniverse_*` groups to GCS (non-commercial academic terms).
7. ✅ Manifest + `download_assets.sh` + README + scan log.
8. ✅ Verified reproducibility: regenerated all three tiers + full materials (base mdls, the
   ~63 textures the ladders reference, and per-design decals) from a fresh Warehouse download.
9. ✅ Upload the `_collision_rigid.usd` variants to GCS alongside the ladder group; add manifest rows.
10. ✅ Smoke-test `download_assets.sh` from a clean state.

## Acceptance Criteria

- Manifest documents every uploaded asset (path, category, source, units, state, collision,
  usage).
- All 98 ladders load with working collision in Isaac Sim.
- `download_assets.sh` syncs the `omniverse_*` task groups from a clean state.
