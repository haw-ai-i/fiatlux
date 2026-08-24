# Issue 70 — Ladder Collision Fix (summary)

Concise "what and why." For the USD overlay/sublayer pipeline — how the collision is authored into the
ladder USDs — see [`collision_authoring_explained.md`](../../docs/collision_authoring_explained.md).

## The problem

In VR teleop the graspable ladder didn't match what you saw: it **moved before the hand visually
touched it**, the **gaps between the rungs felt filled in**, and you **couldn't get a finger into the
C-channel rail groove**. Physics collides against a simplified *collider*, not the render mesh, and the
authored collider was too coarse and sat proud of / bridged across the real surface.

## The solution — everything in the asset, no code

The whole fix is authored into the ladder USDs by
[`scripts/omniverse/omniverse_ladder_collision.py`](../scripts/omniverse/omniverse_ladder_collision.py),
so every task (teleop *and* RL) gets correct collision straight from the file with **zero code**:

| What | Value | Why |
|---|---|---|
| collision shape → **SDF** (signed distance field) | `sdfResolution = 256` | the only collider that is **exact** (opens the concave rail groove, keeps rung gaps) **and** legal on a **dynamic** body (an exact triangle mesh is exact too but PhysX forbids it on a moving body, and these ladders spawn dynamic) |
| **contact offset** | `6 mm` (was PhysX ~2 cm), `restOffset = 0` | contact begins at the visible surface, not a couple centimetres out |
| inverted-mesh designs → **convexDecomposition** fallback | tuned (64 hulls + shrink-wrap) | 6 of 28 designs have an inside-out mesh; SDF's inside/outside is defined by winding, so it would be inside-out — convex ignores winding. Detected generically by a **per-mesh** signed-volume check. |

`convexDecomposition` is the fallback (not an exact triangle mesh) because these files are also spawned
**dynamic**, where triangle mesh is illegal — and convex can't do concave grooves, which is exactly why
SDF is the default.

Both SDF and the 6 mm are authored as raw USD attributes + applied-schema tokens (the script runs in a
plain-`pxr` env), and the 98 `_collision_rigid.usd` variants inherit them by **sublayering** their
`_collision.usd` — one change, both variants, every design. A prior runtime code override (an SDF
spawner + `apply_tight_ladder_collision`) was **removed** once everything moved into the asset.

## How it was verified

On disk (approx + params + `apiSchemas` tokens), at runtime (schema resolves, cooks clean, rests
stably, `contactOffset=0.006` reads from the asset), the unit-scale via a 20 mm resting-height test, and
the inverted mesh via a signed-volume scan. Test harness: the
[ladder gallery](../source/fiatlux_teleop/fiatlux_teleop/ladder_gallery_teleop_env_cfg.py)
(`FIATLUX-LadderGallery-Teleop-v0`) — every design, dynamic and grabbable, to feel across the set.

## Caveats

- **RL cost / reverting to selective.** SDF is heavier than convex across many parallel training envs.
  Nothing trains on these yet, so all-SDF is fine — and it's a one-line revert in `_approximation_for`
  (SDF only for grooved fiberglass designs, convex for the rest).
- **Bucket re-upload.** Assets sync from a GCS bucket (`assets/download_assets.sh`, `gsutil rsync`), so
  the regenerated files are local until re-uploaded, else a future download overwrites them.
- **Collision-geometry fix, not grasp-retention.** This corrects *where* the hand contacts the ladder.
  It does **not** make friction *hold* a heavy ladder against gravity while carrying — that's a separate
  problem (a pose-follow attach, the bulb/socket `attach.py` pattern), not collision tuning.
