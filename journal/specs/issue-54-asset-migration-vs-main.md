# PR #60 vs `main`: two independent Omniverse bulb/socket migrations

Written 2026-07-28, after pushing `68ec17e` to PR #60 and discovering `origin/main`
had advanced 28 commits (PRs #59 and #63) with overlapping work.

**Recommendation: adopt `main`'s asset design and drop most of PR #60's. `main` solved
the problem PR #60 worked around.** Details and the salvage list below.

---

## 1. What happened

Both branches independently concluded the BEHAVIOR-1K bulb/lamp pair had to go, and
both moved to Omniverse. They picked different source assets and arrived at different
architectures. Neither knew about the other.

| | PR #60 (this branch) | `main` (`d9f36b4`, Anton Nikolaev) |
|---|---|---|
| bulb source | Omniverse `chandelier_A` (Residential) | Omniverse `LightBulb` (Sample Scenes) |
| bench source | Omniverse `BlackLamp`, shade dropped | same `LightBulb` |
| how split | `omniverse_fixture_split.py` (new) | `omniverse_bulb_rigid.py`, wrapping Yujin Chen's `_bulb`/`_socket` layers |
| pairs needed | **two** (`CEILING_PENDANT`, `BENCH_LAMP`) | **one**, used everywhere |
| socket collider | convex, plus `--no-collide` to avoid interpenetration | **exact triangle mesh** (`physics:approximation = "none"`) |
| "seated" test | `socket_pose * seat_offset == bulb_pose * plug_offset` | `bulb root pose == socket root pose` |
| screw rotation | not modelled in the geometry | free about an explicit mating axis |
| masses | bulb 50 g (guessed) | bulb 35 g, fixture 0.30 kg, crate 1.5 kg (authored) |

## 2. Why `main`'s design wins

### 2.1 The socket hole is actually open

`main` keeps the socket's colliders as an exact triangle mesh, so the screw hole stays
open and *a bulb lowered in nests and rests on its own*. The comment in `assets.py`
states the constraint precisely: a convex hull closes the hole, and an exact mesh is
illegal on a dynamic body, so the socket can only ever be static or kinematic.

PR #60 hit exactly this wall and settled for less. Its convex-hull socket is a sealed
cylinder that the seated bulb's screw base occupies — 24.8 mm of overlap, radially
inside. Codex caught it; the fix was `--no-collide`, which keeps the socket visible but
gives it no collider at all. The bulb never physically rests; it is held by the FSM and
nothing else. `main` gets real seating; PR #60 gets a pose-slaved approximation.

This is the single most important difference, and it is not close.

### 2.2 One pair instead of two

PR #60's whole `FixturePair` abstraction exists to solve one problem: a pendant cannot
stand on a bench and an upright lamp does not hang from a ceiling. That forced two
pairs, two offset sets, a lookup keyed on the spawned socket USD, and per-preset
swapping.

`main` dissolves the problem by choosing a compact fixture whose origin *is* its
floor-contact plane (`SOCKET_BASE_Z_OFFSET = 0.0`), so the same asset sits on a bench,
on the floor, on a wall, or overhead — `ELEVATED_SOCKET_USD = SOCKET_USD`. No abstraction
needed, because there is nothing to abstract over.

An abstraction that exists only to paper over a bad asset choice should lose to the
better asset choice.

### 2.3 Seating is pose equality

Because `main` authors both halves assembled at identity, `SOCKET_SEAT_OFFSET` and
`BULB_PLUG_OFFSET` are the *same* value (`0.036259`), and "seated" reduces to *bulb root
pose == socket root pose*. PR #60 carries two different offsets per pair (four numbers
total) and a sign convention that differs between the pendant and the bench lamp — a
trap I already fell into once, deriving the seat from the wrong end of the bulb.

### 2.4 Screwing is modelled

`main` adds `BULB_PLUG_AXIS` / `SOCKET_SEAT_AXIS` and scores alignment *about* that axis,
so rotation around it — the screwing motion — is free. This is directly what issue #54's
attach FSM gates on. PR #60 has no equivalent; its FSM tracks wrist roll independently of
any geometric mating axis.

### 2.5 Offsets are re-measured, not registered

`main` has `verify_scene` re-measure the offsets from the geometry at runtime, so they
cannot drift. PR #60's `fixture_pair_for` keying on the spawned socket USD is good — it
cannot disagree with the asset in the scene — but it still trusts hand-transcribed
constants. Re-measuring is strictly stronger.

### 2.6 Physics work PR #60 does not have

`main` also lands: authored masses and insertion-scale contact offsets instead of PhysX
deriving tens of kg from collider volume; the room moved per-env with live wall
colliders (probe-verified with a 6 m/s bulb rebound); the room floor aligned to z=0 by
measuring the walking slab at spawn rather than hardcoding; the crate spawned hollow so
a bulb rests inside it; `G1_OVERHEAD_REACH` and a Replace layout feasible by
construction; wall fixtures actually on the walls.

## 3. Where the two agree

Convergent conclusions, reached separately, which is decent evidence both are right:

- The BEHAVIOR-1K lamp had to go. `main`'s reason — "the B1K lamp is not hollow, so a
  bulb could not physically enter it" — is the same defect PR #60 found by vertex
  profiling (`ehjsdz` is a single fused mesh).
- `spawn_b1k_single_body` and `apply_carry_preset` are dead. Both branches deleted them.
  `main` (`651136c`) adds a reason PR #60 missed: the spawner's `Usd.PrimRange` walk
  lacks instance-proxy traversal, so it would silently strip nothing on an instanced
  asset.
- The installed old bulb must be dynamic, not kinematic, or removal is not a real event.
- Unit conversion belongs baked into the asset, not applied at spawn. `main` authors
  Z-up/metre wrapper layers; PR #60 arrived at baking the hard way, after a spawn-time
  `scale` on a rigid body left bulbs falling through the ground plane while still
  colliding with mesh geometry.

## 4. What PR #60 still contributes

Ordered by how much would be lost if the branch were simply abandoned.

### 4.1 The manifest correction — keep, unaffected by the conflict

`main` still carries all **27** `omniverse_lamp` rows reading `scene_dressing` /
*"fused model, no separable bulb"*. That annotation is wrong and was applied wholesale
without opening the files. PR #60 rewrote every row from measured geometry:

- 5 genuine mating pairs (`BlackLamp`, `chandelier_A`, `Isabelle`, `Colbert`, `Theory`)
- 15 with a separable bulb but no distinct socket
- 7 with neither, including `Dylan` and `VintageFloorLamp` flagged as having socket
  parts but *no bulb in the model*, and `AshleyLamp` flagged Y-up

This is independent of which asset the env uses and is worth keeping on its own. It also
records that `Mechant` and Sample-Scenes `OldAttic/TableLamp` are the same geometry.

### 4.2 The survey method — keep as documentation

All 14 public Omniverse packs (~183 GB) were surveyed without downloading them, by
HTTP-range-reading each zip's central directory, then range-fetching only the ~30
lighting USDs. Findings worth recording: `Commercial`, `Industrial`, `Data_Center`,
`Showcase` and all five `SimReady_*` packs contain **zero** lighting props — so no
Omniverse light fixture anywhere ships with colliders, which is why both branches had to
author them.

### 4.3 `omniverse_fixture_split.py` — probably redundant, possibly useful

Generalised splitter with `--bulb` / `--drop` / `--plug-end` / `--bulb-approx` /
`--no-collide`. `main`'s `omniverse_bulb_rigid.py` already covers its own asset. Keep
only if more fixtures get split later; otherwise drop.

### 4.4 Obsolete under `main`

- `FixturePair`, `fixture_pair_for`, and the two-pair plumbing in `rewards.py`,
  `attach.py`, `scene_cfg.py`
- `chandelier_A` and `BlackLamp` assets and their GCS prefixes
- `--no-collide` (superseded by the open-hole exact-mesh socket)
- `SOCKET_HANG_HEIGHT` and the pendant-specific mounting

## 5. Errors in PR #60 worth carrying forward as warnings

1. **`omniverse_bulb` was rejected on a bad read.** PR #60 dismissed Sample-Scenes
   `LightBulb` as "Y-up, no socket, wrapped in an OmniGraph demo" and removed it from
   `download_assets.sh`. `main` chose that exact asset and calls it "the only separable
   modelled bulb-in-a-socket in any Omniverse pack". The rejection looked at the raw
   sample scene; `main` looked at Yujin Chen's `_bulb`/`_socket` layers. **The
   `download_assets.sh` change must not land — `main` depends on `omniverse_bulb`.**
2. **A `chandelier_A` socket that is a cap, not a cup.** Called a "decorative cap" on
   render evidence; vertex profiling then showed a real ~1 mm-clearance conforming bore.
   Occlusion was read as absence. Measure before concluding.
3. **`apply_carry_preset` was rewritten before checking its callers.** It had none.

## 6. Suggested resolution

1. Reset PR #60 to `origin/main`.
2. Re-apply only the manifest correction (§4.1), as a standalone commit.
3. Optionally keep §4.2 as a note in this file and `omniverse_fixture_split.py` if more
   fixtures are coming.
4. Drop everything in §4.4. Delete the `omniverse_chandelier` and `omniverse_table_lamp`
   GCS prefixes once the branch is settled (37 MB + 138 MB).
5. Re-point issue #54 at `main`'s pair, and re-run `verify_attach` there — the attach FSM
   itself (`mdp/attach.py`'s state machine) is unaffected by the asset choice and is the
   part of PR #60 worth preserving.

Conflicting files if merged instead of reset: `scripts/verify_scene.py`,
`assets.py`, `scenes.py`, `mdp/rewards.py`, `replace_env_cfg.py`, `scene_cfg.py`.
They conflict because both sides restructured the same things, not by textual accident.
