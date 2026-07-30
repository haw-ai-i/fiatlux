# Critique of the subtask plans

Adversarial pass over `00-foundation.md`, `ABSTRACTIONS.md`, `CONTINUITY.md` and the 15 subtask
plans. Written against my own work; findings are ordered by how much damage they do if they ship.

## A. Confirmed defects, with numbers

**Status:** A1 and A2 are **fixed in code** on this branch and tracked as issue #69 (they were
pre-existing, not introduced by these plans). A3 is fixed (fixture lowered to 2.2 m). A4, A5 and A6 are
corrected in the affected subtask plans.

### A1. `ladder_contact` sees no hands at all on Dex3 — the scored variant · **FIXED**

`add_ladder_contact_sensor` uses:

```python
prim_path="{ENV_REGEX_NS}/Robot/.*(ankle_roll|hand_base)_link"
```

Inspire's palms are `left_hand_base_link` / `right_hand_base_link` — matched. **Dex3's palms are
`left_hand_palm_link` / `right_hand_palm_link` — not matched.** On Dex3 the sensor resolves only the
two ankle links, so `ladder_contact_fraction` averages over feet alone and the "palms on the ladder"
half of the bootstrap reward silently does not exist.

This is a **pre-existing bug**, live in `FIATLUX-Climb-v0` and `FIATLUX-Descend-v0` right now, and
the plans inherit it into S05, S07, S13 and S15. Worse, my S13 plan reasons at length about how "a
permanently-free palm caps the term at 3/4 and the policy is charged for holding the bulb" — that
analysis is wrong from the premise up, because on Dex3 there is no palm in the denominator. I wrote a
careful-sounding paragraph about a mechanism that does not exist.

Fixed by naming the bodies explicitly from `G1_LADDER_CONTACT_BODIES` (both feet plus *every*
variant's palm) instead of a regex that encodes one variant's naming. The sensor is built before
`swap_robot_variant` may change the hand, so it has to name them all; names belonging to the other
variant simply never resolve.

Same class of bug as `palm_body_index` hardcoding `right_hand_base_link`, which PR #64 just fixed —
the regex was that bug's twin, one file over. Worth a standing rule: **never spell a G1 body name
outside `robots/g1.py`.**

**Contract change:** on Dex3 the sensor now resolves 4 bodies instead of 2, so Climb's and Descend's
`ladder_limb_contact` observation goes from 6 to 12 values and `ladder_contact_fraction`'s denominator
from 2 to 4. Dex3 checkpoints and recorded scores for those two tasks do not transfer.

### A2. `LADDER_READY_XY_RADIUS = 0.9` admits placements from which the fixture is unreachable · **FIXED**

Geometry, all from existing constants:

| Quantity | Value |
|---|---|
| ladder top (`STEP_LADDER_TOP_OFFSET[2]`) | 1.70 m |
| ceiling fixture (`CEILING_FIXTURE_Z`) | 3.00 m |
| vertical gap to cover | **1.30 m** |
| standing overhead reach (`G1_OVERHEAD_REACH`) | 1.3738 m |
| required reach at 0.9 m horizontal offset | **1.581 m** |
| max horizontal offset still reachable | **0.444 m** |

So a ladder parked 0.9 m off-axis satisfies `ladder_ready` while putting the socket ~21 cm beyond
full stretch. S04 can therefore *succeed* at a placement from which S14 is geometrically impossible —
a cross-subtask contradiction between two gates I wrote in the same sitting, and neither plan caught
it because each checked only its own numbers.

Fixed by **deriving** it instead of picking it:
`sqrt(G1_OVERHEAD_REACH² − (CEILING_FIXTURE_Z − STEP_LADDER_TOP_OFFSET[2])²) − 0.05` = **0.394 m**,
sized for the ceiling mount (the worse of the two — a wall fixture sits 0.8 m lower and allows far
more slack), with an import-time guard that raises if the fixture height ever eats the whole reach
budget. Derived means it cannot drift out of agreement with the fixture height again.

The audit also turned up that the constant was **defined twice** — `carry_env_cfg.py` and
`replace_env_cfg.py` each held their own `0.9`, under a comment claiming the subtask and the full task
"judge the ladder identically". Both now import the derived value.

**Contract change:** Carry's success predicate *is* `ladder_ready`, so its scores do not transfer.

### A3. The overhead reach margin is 7.4 cm, and measured from the wrong stance · **FIXED**

1.3738 − 1.30 = **0.074 m** of margin straight up. But `G1_OVERHEAD_REACH` is measured with the
robot *standing on a flat surface, arm straight up, all other arm joints zero*. The robot at S14 is
in a balanced crouch on a ladder step, leaning, with one hand holding a bulb it must not crush. Every
one of those costs vertical reach.

Quantified without a probe: Climb's gate accepts a pelvis at 1.700 m, so feet at 0.910 m and
fingertips at 2.284 m against a 3.00 m fixture — **0.716 m short**, while the guard passed because it
assumed feet on the 1.70 m top platform (pelvis 2.49 m), a stance no task requires.

Fixed by deriving the bound from the working stance and lowering the fixture to **2.200 m**, with
guards on both sides: reachable from the stance, and above standing floor reach so the ladder stays
necessary. `LADDER_WORK_PELVIS_Z` is now the single definition of that height and Climb imports it as
its `SUCCESS_HEIGHT`, so clearing the climb gate and being able to work the fixture are one condition.

**Contract change:** the scene geometry moved, so Replace, Carry, Climb and Descend scores do not
transfer.

### A4. S02's grasp gate is satisfied by nudging the ladder 5 degrees

Gate: hand↔ladder contact on ≥2 bodies **and** "ladder root lifted > 0.03 m". The ladder root is at
its base centre, so **tilting the A-frame raises the root**. With a half-footprint of 0.34 m, a 3 cm
root rise needs `asin(0.03/0.34)` = **5.1°** — and `ladder_tipped` does not fire until 0.6 rad (34°),
where the root has risen 0.19–0.31 m. So "lift" is achievable by leaning on the ladder hard enough to
tilt it slightly, which is the exact degenerate solution the gate was written to exclude. My own
justification — "lift is the gate, not contact; taking its weight is what a grasp is" — is undermined
by the metric I chose to express it.

Better gate, and it says what I actually meant: **the hand must carry roughly the ladder's weight.**
`mg = 7.25 × 9.81 = 71 N` on the filtered hand↔ladder channel, plus all four feet clear of the floor
(foot body positions, not the root). That is a direct measurement of load transfer and cannot be
faked by tilting.

### A5. S09's success window excludes the orientation the bulb will actually be in

Gate: bulb root z within 0.03 m of `BIN_BULB_INTERIOR_Z − BULB_STAND_Z_OFFSET` = 0.0187 m, i.e. the
window [−0.011, 0.049].

- standing on its cap → root z **0.0187** ✓
- lying on its glass → root z **0.0946** ✗

`poses.py` argues the Omniverse bulb self-rights onto its cap, so the window is not necessarily
unsatisfiable — but it is over-specified: a bulb resting against a crate wall, wedged, or still
settling fails a gate that should only be asking "is it in the crate". Replace the z-equality with a
containment test against the crate's interior AABB, which is orientation-agnostic.

### A6. `LADDER_APPROACH_RADIUS` is used for two different requirements

S01/S02 use it for "close enough to grasp a rail". S12 reuses it for "arrived at the ladder to climb
it". Those are different standoffs — one is an arm's-reach question, the other a foot-placement
question — and S12 borrows the constant without noticing. Two constants, or one with a stated
justification for covering both.

## B. Flaws in my own abstraction proposal

### B1. The `sustained` combinator does not compose the way I claimed

I proposed `all_of([...])` plus `sustained(pred, seconds)`, and separately claimed that wiring the
same predicate into both `terminations.success` and `rewards.success_bonus` from one site guarantees
they agree. Those two claims are incompatible.

`sustained` is **stateful** (it holds a counter). `completion_bonus` calls
`predicate_fn(env, **params)` as a plain callable, so a stateful predicate cannot be dropped in as
one. And if the gate is implemented as a `ManagerTermBase`, the **termination manager and the reward
manager each instantiate their own copy** — two independent counters, reset independently, evaluated
at different points in the step. They will usually agree and will occasionally not, which is the
worst failure mode available.

The fix has to be explicit: compute the gate **once per step** in a single owner and have both
managers read the cached result, or keep every gate predicate stateless and put `sustained` only on
the termination. Either is fine; asserting "one wiring site therefore no drift" without addressing
statefulness is not.

### B2. The three-level cfg hierarchy is unproven in this codebase

The repo contains **zero** examples of even two-level `@configclass` env-cfg inheritance — all six RL
cfgs derive straight from `ManagerBasedRLEnvCfg` and re-declare everything. I am proposing three
levels, with inherited nested inner classes (`ObservationsCfg.PolicyCfg`) partially extended by
subclasses, plus `ClassVar` hooks. Dataclass machinery is sharp exactly there.

I flagged verifying `ClassVar`. I did not flag that the entire hierarchy is unvalidated. It should be
a **spike before anything else**: base + one intermediate + one leaf, confirm all five managers
construct and the term ordering is what field declaration order implies. If that fails, the fallback
is a flat base + explicit composition helpers, and it is much cheaper to learn that now than after
15 files exist.

### B3. The frozen `policy` observation group has a cost I asserted away

I presented freezing it as pure upside. It is not: every subtask then carries `ego_rgb`
(a ResNet-18 forward pass per step) and `lidar_ranges`, including the tabletop grasp subtasks where
the lidar sees nothing relevant. For RL training throughput across 15 tasks that is a real tax, paid
so that the observation space stays uniform.

The uniformity argument still holds — it is the sim-to-real argument, and it is correct — but the
plans should state the price rather than pretend there isn't one.

### B4. `MateSubtaskCfg` forces a symmetry that isn't there

S06 (unmate) and S14 (mate) share an intermediate on the grounds that both are "mating". But S14
needs fine alignment, free rotation about the mating axis, and retention-after-release; S06 needs to
break a constraint and *catch a falling object*. The shared structure is thin — mostly the seat-point
distance functions, which are `mdp` functions anyway and need no cfg class to be shared. Weakest of
the six intermediates; likely should be two leaves off the base directly.

## C. Framing that overclaims

### C1. The discretization rule does not actually partition the 15

"Every switch between navigation, balance, and grasping is a subtask boundary" reads as a partition
rule, but the modes are not exclusive and the plans admit it in their own headers: S03 is
"navigation (loaded)" — navigating *while* grasping; S06 is "grasping on balance". If a subtask can
be two modes at once, "a switch between modes" does not define a boundary, and the 15 are a
hand-authored list that the rule rationalizes after the fact.

That is fine — the list is the spec, and it is a good list — but the docs should say so instead of
implying the boundaries were derived. As written, the rule invites a future reader to "correct" the
list by applying it literally.

### C2. I oversold the handoff check, then the decisions gutted it

The original text called `handoff:` "the check that catches the failure mode this re-discretization is
most exposed to". With deterministic on-ladder start states and independent envs, it is a static lint
over frozen constants that can only fail if someone edits one. Now corrected in the foundation, but
it was load-bearing rhetoric for a while and it should not have been.

### C3. The visual gate is unfalsifiable as specified

"Frames must show nothing floats, nothing intersects; the payload enclosed by the digits." At
1280×720 on an orbit render, an agent cannot distinguish 3 mm of interpenetration from contact, and
"enclosed by the digits" is a judgment call. I leaned on renders because coded checks fooled me
twice — but the lesson was *use both*, not *renders are the gate*.

Every visual item needs the numeric check that actually decides it paired next to it:
`overlap_mesh` for penetration (already specified in the parent plan's Phase F.2), the settle-drift
soak for resting, filtered contact force for holding. The render's job is catching what nobody
thought to assert; it cannot adjudicate millimetres.

## D. Scope and process

### D1. Six calibrated poses where two would do

The plans imply distinct calibrated start poses for S06, S07, S13, S14, S15 and the standing-cradle
subtasks. But S06/S07/S14/S15 are all *the robot balanced on the upper steps*, differing only in
whether a payload is in the hand; S03/S08/S12 are all *standing on the floor with a payload*.

Two calibrated artefacts cover all of it: **one on-ladder stance** and **one standing cradle**, each
with an optional payload offset. Given that this project's recent history is a catalogue of
hand-calibrated states that turned out physically wrong, reducing six chances to two is the single
cheapest risk reduction available.

### D2. No scoring definition, and the naive one is wrong

Fifteen gates are not a benchmark. Five subtasks (S01, S03, S08, S10, S12) are "walk to X", so an
unweighted average gives a policy that can only walk 5/15 = 33%, while the hardest single subtask
(S14) is worth 1/15. Partial credit is undefined. Now recorded as open in the foundation, but it
should have been in the first draft — scoring is the point of a benchmark.

### D3. Episode lengths are presented as if calibrated

S14 at 40 s, S06 at 30 s, S03 at 30 s. The only measured datum in the whole set is the walk probe
(2.43 m in ~8 s at ~0.5 m/s), which justifies the navigation horizons and nothing else. The
manipulation-under-balance numbers are guesses and should be labelled provisional.

### D4. No vertical slice, so the abstraction stack is untested until it is 15 files deep

The plans order work by what is unblocked (S01, S05, S10) but never say "finish one subtask through
all three consumers before authoring the next". With VLA + RL + scripted all in scope, S01 should go
end to end — env, start state, gate, visual+numeric validation, `PPORunnerCfg`, scripted baseline,
score entry — and only then should S02 begin. Otherwise B2's hierarchy risk, D2's scoring gap and the
runner-cfg requirement are all discovered fifteen times.

### D5. The ego-camera decision is deferred into 15 separate discoveries

Every plan says "confirm the target is in the ego frustum at t=0; if not, report it and stop". The
46°-down mount is a known open question from an earlier session. As written, up to 15 agents stop on
the same unresolved decision. Decide it once, before S01.

### D6. ~~Retiring Carry/Climb/Descend is asserted without a migration list~~ — WITHDRAWN

Moot: **nothing is retired.** The coarse tasks span several mode switches with no episode boundary
to help, which is a strictly harder and genuinely different capability — coordination across
transitions rather than competence within one mode. Keeping both tiers turns the family into a
difficulty ladder that localizes *which* capability a policy lacks. See the foundation.

What survives of this finding: shared constants now retune both tiers at once (tightening
`LADDER_READY_XY_RADIUS` is a contract change for Carry and Replace, not just S04), and scoring must
never average across tiers.

## What survives

The parts I would defend unchanged: the 15-way split itself and its ordering (reviewed and endorsed);
one module/class/id per subtask for independent versioning; coverage-over-equality as the handoff
framing; randomize-what-is-fat / calibrate-what-is-thin; deriving every on-ladder gate from the
ladder's live pose rather than `TOP_ROBOT_POSITION`; requiring a *held* conjunct wherever a payload
is involved (which is what stops S06 scoring a falling bulb); and moving the grasp calibration out of
`scripts/` into the package.
