# Domain randomization: scale, visual tint, light direction

Design record for the three DR axes requested in the "Missing Domain Randomization
(Sim-to-Real Gap)" issue, matching sections (A), (C), (E) of the formerly-disabled
scaffold in `base_env_cfg.py`. Sections (B) start-pose and (D) physics-material remain
scaffold comments — untouched by this work.

## Scope decision: scaffold defaults, RL opt-in

Prestartup USD scale randomization and per-env material writes require
`InteractiveSceneCfg.replicate_physics=False` (the event manager raises on
`prestartup` terms under replicated physics; per-env USD writes are only trustworthy on
unreplicated stages). The RL training cfgs deliberately keep `replicate_physics=True`
for throughput, and `FIATLUX-Replace-v0` is the scored benchmark — silently changing
its difficulty was ruled out.

Therefore:

- **Scaffold envs** (`FamilyBaseEnvCfg` → Base/Descend/Remove/Install): all three axes
  ON by default, gated by `enable_dressing_randomization` (the dressing-off branch
  strips the scale terms and reduces tinting to the shared room — same branch that
  re-enables replicated physics, so a missed site fails loudly at construction).
- **RL cfgs** (Insert/Climb/Carry/Replace): replicate-safe axes only by default —
  light direction + global room tint. Scale DR is a documented opt-in: set
  `scene.replicate_physics = False` and add the prestartup `randomize_*_scale` terms
  from `FamilyBaseEnvCfg.EventCfg`.

## Axis 1: prop scale

- Meter-authored B1K props (socket, bulb) use Isaac Lab's built-in
  `mdp.randomize_rigid_body_scale` (range (0.9, 1.1), isotropic, per-env).
- The built-in **overwrites `xformOp:scale` absolutely**, which would inflate the
  cm-authored SimReady ladder/crate (baked spawn scale 0.01) 100×. The project term
  `mdp.randomize_prop_scale` is the multiplicative sibling: sampled factor × authored
  `spawn.scale`. Ladder ranges x/y (0.95, 1.05), z (0.95, 1.1) — the scaffold's own
  authored values.
- The bin/crate is excluded (bare scenery in Remove/Install); `randomize_prop_scale`
  is the opt-in if that changes. The robot cannot be scaled (Isaac Lab rejects
  articulations; use multi-asset spawning instead).
- Known, accepted geometry side-effects at the range extremes: the metalink calibration
  constants (`SOCKET_SEAT_OFFSET`, `BULB_PLUG_OFFSET` in `assets.py`) do not scale with
  the prop — worst-case seat-point error ≈ 6.4 mm at ±10 %, under the 1.5 cm success
  threshold. Climb's `SUCCESS_HEIGHT` tolerates the ladder z range (± ~17 cm at the
  top vs a 0.15 m success margin — scale DR is not an RL default there anyway).

## Axis 2: visual tint (why a custom term, not the built-ins)

The scaffold's TODO claimed Isaac Lab 2.3.2 ships no visual randomizer — stale: it
ships `randomize_visual_color` / `randomize_visual_texture_material`. They were
rejected because they are Replicator-based, **rebind an OmniPBR material over whatever
is authored** (the team deliberately keeps the B1K `OmniGibsonVRayMtl` and SimReady
MDLs original — see issue-18), and hard-require `replicate_physics=False`, which
excludes the shared `/World/Room` under the RL cfgs.

`mdp.randomize_material_tint` instead multiplies a sampled HSV tint into the color
inputs the materials already author (`diffuse_tint`, `diffuse_color_constant`,
`diffuseColor` — written at both Material and Shader level when both are authored,
since Material-level inputs override shader params). `diffuse_tint` is created when
missing only on shaders positively identified as OmniPBR (the parameter exists there
with default (1,1,1) — the Simple Room's texture-only walls). Probe results that pinned
this: room = OmniPBR, mostly texture-only; packing table = SimReady `Plastic_B`
authoring both levels; ladder SimReady MDLs author `diffuse_tint` (the brushed-aluminium
part authors none → skipped); crate `Plastic_Yellow_A` and B1K VRayMtl author none →
naturally skipped.

Anti-compounding: the original color is cached in attribute customData
(`fiatlux:base_color`) on first touch; every reset recomputes `original × tint`.
One global sample per reset, drawn before any prim lookups (fixed 3-draw cost, so the
RNG stream does not depend on which optional entities a preset keeps).

**Texture swapping stays future work**: the repo has no texture assets (single HDRI,
no texture library); the Omniverse skies pack was deliberately not ingested.

## Axis 3: light direction

`randomize_light_properties` gained `rotation_range_deg` (and `color_range`): a
perturbation composed onto the light's **authored** `init_state.rot`, so the cone never
drifts across resets. Dome = yaw only (HDRI sun azimuth, 0–360°; tilting the sky's
horizon reads as a render bug — same reasoning as the no-tint rule). Key light =
pitch ±15° / yaw ±30° around its authored 40° downward tilt. All three angles are
always drawn (absent keys as (0,0)) to keep the per-reset draw count constant.
Per-env lights remain TODO (scene lights are single global prims).

## Determinism

All draws ride the seeded torch default generator (`env_cfg.seed`, set after
`random.seed` per the eval.py contract); prestartup events run after seeding, so
same-seed runs reproduce every sampled scale/tint/direction —
`scripts/verify_randomization.py` asserts byte-identical cross-process signatures.
**One-time break**: extending the existing light terms changes per-reset draw counts,
so same-seed numbers from before this change are not comparable to numbers after it.
Expected and accepted.

## Verification

`scripts/verify_randomization.py --headless --seed 0` (subprocess fan-out; USD-attr
assertions only, runs without RTX): scale ranges + per-env variation, tint presence +
B1K exclusion + 5-reset anti-compounding, light-orientation cones, same-seed
determinism, and the RL-default contract (no scale DR, replicated physics, room tint +
light direction active) on Carry. Plus `verify_scene.py` on Base/Remove/Carry and the
`verify_interactions.py` suite (its deterministic cfg builders null the new terms).

Deferred to a machine with a working RTX renderer (this box's crashes on launch):
`verify_scene.py --record` MP4 spot-check of tint/shadow variation, and a two-run
same-seed `eval.py` numbers check on `FIATLUX-Replace-v0` (needs `--enable_cameras`).
