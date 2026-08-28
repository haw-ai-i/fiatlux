# Continuity audit of the 15-subtask chain

A pass over all 15 plans asking one question at each of the 14 boundaries: **is subtask N's actual
end state the same physical situation as subtask N+1's authored start state?** Seven gaps found.
Each subtask plan must be read together with this file.

> Written against the original fifteen subtasks; its numbering is theirs. The four ladder legs
> are now one `S01-MoveLadder` and the chain renumbered — see the map in `00-foundation.md`
> ("As built — the four ladder legs are one subtask"). Left un-renumbered on purpose: this is a
> dated review record, and rewriting it would falsify what was reviewed.

Verdict: the chain is physically coherent in its *ordering* — nothing is picked up twice, nothing
is needed before it exists, the disposal leg correctly precedes the fetch leg.

## Re-scored under domain randomization

This audit was originally written against an **equality** handoff contract, and it over-indexed on
exact continuity as a result. Under the coverage contract (`00-foundation.md`) — subtask *N*'s
success region must lie inside subtask *N+1*'s randomized start support — most of these stop being
defects. DR absorbs pose mismatch; it does not absorb infeasibility or a missing object.

| # | Finding | Status under DR |
|---|---|---|
| C1 | payload pose: threshold vs frozen pose | **dissolved** — the successor's payload range covers it |
| C2 | S04 leaves the robot unconstrained | **survives, reframed** — feasibility, not continuity |
| C3 | #54: old bulb on the floor by S05 | **survives** — presence, not pose; DR cannot cover it |
| C4 | wall vs ceiling mount | **corrected** — my reach claim was wrong; see below |
| C5 | all 15 must share a seed | **dissolved for training**, applies only to a chain rollout |
| C6 | 1 cm handoff tolerance too tight for the ladder | **dissolved** — no tolerance to argue about |
| C7 | S15's terminal render needs #54 | **survives** |

Three real findings, one correction. The two that survive are both cases where the *world* differs,
not where a pose differs — which is exactly the line DR draws.

---

## C1 — ~~Every loaded subtask must pin its payload to the successor's grasp pose~~ (dissolved)

**The gap.** Grasp and transport subtasks currently end on a *threshold*, while their successors
start from a *frozen pose*. S02 succeeds at "ladder lifted > 3 cm", but S03 starts from
`LADDER_IN_ROOT_CARRIED`; a 3 cm lift is not the carry pose. S11 succeeds at "bulb lifted > 5 cm"
while S12 starts from `BULB_IN_ROOT_STANDING`. The two states are related but not equal, so the
handoff is a gap the chain silently jumps.

Worse at S07→S08: S07 *starts* from `BULB_IN_ROOT_ON_LADDER` and ends standing on the floor, but
S08 starts from `BULB_IN_ROOT_STANDING`. Nothing in either subtask performs the re-pose between the
two, and it cannot become a 16th subtask — re-posing an arm while standing still is not a
navigation/balance/grasping switch, so the rule does not put a boundary there.

**Resolution under DR.** No pinning needed. The successor randomizes its in-hand payload pose over
a range, and as long as that range covers where a successful grasp or carry actually leaves the
object, the mismatch is absorbed — and the successor wanted that range anyway, since a policy that
only works from one exact in-hand pose is useless.

What remains is a **validity** requirement on the range, not a pinning requirement on the
predecessor: every sampled in-hand pose must be one the hand can actually hold, not an arbitrary box
around the centre. Sample the extremes, render them, confirm the payload is enclosed by the digits.
The table below is therefore a guide to what each successor's range must *cover*, not a set of
equalities to enforce.

| Subtask | Successor's range must cover its outcomes around | Consumed by |
|---|---|---|
| S02 | `LADDER_IN_ROOT_CARRIED` | S03 |
| S03 | `LADDER_IN_ROOT_CARRIED` | S04 |
| S06 | `BULB_IN_ROOT_ON_LADDER` | S07 |
| S07 | **`BULB_IN_ROOT_STANDING`** | S08 |
| S08 | `BULB_IN_ROOT_STANDING` | S09 |
| S11 | `BULB_IN_ROOT_STANDING` | S12 |
| S12 | `BULB_IN_ROOT_STANDING` | S13 |
| S13 | **`BULB_IN_ROOT_ON_LADDER`** | S14 |

The two bolded rows are pose *changes* across a subtask: S07 descends holding the bulb and ends
standing, S13 climbs holding it and ends on the ladder. Under coverage these need no gate condition
at all — S08's and S14's ranges simply have to span both cradle poses, which is a wider range than
either alone and a strictly better training distribution.

---

## C2 — S04 does not constrain where the robot ends up (survives: feasibility, not continuity)

**The gap.** S04's gate is about the *ladder*: placed, upright, at rest, released. It says nothing
about the robot beyond "standing". But the robot arrives at S04 having **carried** the ladder — it
is beside or behind it, holding a rail. S05's start state is "robot standing at the ladder's base,
facing the steps", and the A-frame's steps face one way only. A robot that sets the ladder down and
stops is very likely on the wrong side of it.

**Why DR does not rescue this one.** Widening S05's spawn range to cover the back of the ladder does
not help: you cannot climb the back of an A-frame, so those samples are unsolvable episodes, not
useful variation. This is the feasibility limit — DR covers variation inside the feasible set and
cannot extend it.

**The fix.** S05's spawn range covers the step-facing arc and stops there, and S04's success gate
must land the robot inside it: root within `LADDER_APPROACH_RADIUS` of the ladder root, on the
step-facing side. That side is a function of the layout draw's `ladder_yaw`, so it must be computed,
not hardcoded.

This is the boundary most likely to be missed in implementation, because both plans read fine on
their own.

---

## C3 — Without #54 the old bulb is already on the floor by S05, not just at S06

**The gap.** S06's plan flags #54 for its own success gate. The chain-level consequence is larger:
the old bulb is dynamic in an inverted fixture from **S01 onward**. With nothing retaining it, it
falls out during S01's very first episode and lies on the floor for S01–S05. So S05's *actual* end
state has no bulb in the fixture, while S06's *authored* start state has one seated. That is a
broken boundary four subtasks upstream of where the blocker is currently recorded.

Each subtask is its own env and re-authors its start state at reset, so every subtask still runs in
isolation — which is exactly why this would go unnoticed without the `handoff:` check. It is the
first thing that check will catch, and it earns its keep on the first pass.

**Not fixable here.** #54 owns it. What the plans must do: record that S01–S05's rendered scenes
will show the old bulb on the floor, and not treat that as an S01–S05 defect.

---

## C4 — S06 and S14 describe only the ceiling half of the sampled fixture (corrected)

**The gap.** Both plans say "fixture inverted, socket mouth facing down". That is the **ceiling**
draw. `_sample_fixture_mount` picks ceiling or wall with probability 0.5 each, and a wall mount
applies `_quat_z_deg(yaw) * _quat_y_deg(90)` — the socket opening points **horizontally into the
room** at `WALL_MOUNT_Z = 2.2`, not downward.

Three consequences the plans currently miss:

1. **The insertion axis is horizontal**, so the approach, the wrist orientation, and the reachable
   pose from the ladder are all different. A start-state pose validated against a ceiling mount
   will be wrong half the time.
2. **Retention behaves differently, in the chain's favour.** With a horizontal mating axis gravity
   pulls *perpendicular* to the axis rather than straight out of the socket, so a resting bulb is
   not simply dropped by gravity. The wall case is therefore substantially less dependent on #54
   than the ceiling case.
3. `WALL_MOUNT_Z = 2.2` versus `CEILING_FIXTURE_Z = 3.0` — a wall fixture is 0.8 m lower, so the
   working stance is a lower step rather than the top.

**Correction.** An earlier version of this audit claimed a 2.2 m wall mount might be workable from
the floor, making the ladder subtasks pointless for that draw, and recommended restricting the chain
to ceiling mounts on that basis. That was wrong: `G1_OVERHEAD_REACH = 1.3738` is standing fingertip
height, which is 0.83 m short of 2.2 m. **A wall-mounted fixture still requires the ladder**, the
chain's premise holds for both draws, and the recommendation to force ceiling mounts is withdrawn.

**The fix, then, is narrower than it looked.** Mount kind stays a randomization axis — restricting it
would only shrink the benchmark. S06 and S14 must handle both geometries: the wrist orientation and
approach direction differ, and their start-state renders must be validated for *both* draws, not
just the ceiling one. The wall case is also the easier of the two for #54, since gravity acts
perpendicular to a horizontal mating axis rather than pulling the bulb straight out of the socket.

---

## C5 — ~~All 15 must share one layout seed~~ (dissolved for training)

**The gap.** The Replace layout is drawn **once per cfg build** (`apply_replace_preset`'s own `rng`
draw), and each of the 15 subtasks builds its own cfg in its own process. Two subtasks run at
different seeds are two different rooms — different ladder zone, different fixture mount, different
crate position — and the chain is not one physical process at all.

**Resolution under DR.** Not required for training. If each subtask's start distribution covers its
predecessor's outcomes, the chain is a claim about distributions and every subtask can randomize its
own layout — which is what you want, since a subtask trained in exactly one room has learned that
room. A shared seed matters only when rolling the 15 out as one continuous demo or evaluation.
Scores still may not be compared across seeds, because the geometry differs.

---

## C6 — ~~The ladder's handoff tolerance cannot be 1 cm~~ (dissolved)

**The gap.** The foundation's `handoff:` check asserts shared entity poses match "to within 1 cm".
The ladder is dynamic and gets climbed by a 35 kg humanoid in S05, S07, S13 and S15. It will settle,
shift and rock. A 1 cm tolerance on a repeatedly-climbed dynamic body will produce failures that
are noise, not defects.

**Resolution under DR.** The tolerance question disappears with the equality contract that created
it. Under coverage there is nothing to tune: the successor's ladder-pose range either contains the
predecessor's post-climb settling or it does not, and if it does not, the range was too narrow. Keep
the separate `ladder_tipped` check for the thing that actually matters — that it is still standing.

---

## C7 — S15's terminal state is unverifiable without #54, and it is the benchmark's success image

Minor, but worth stating: S15's start state requires the fresh bulb *retained* in an inverted
fixture, and its success gate requires it still seated on arrival at the floor. Both are #54. Since
S15's start state doubles as the "what does task success look like" reference render for the whole
benchmark, that render cannot be produced honestly until #54 lands. Do not substitute a kinematic
bulb to get a nice picture — it would be a picture of something the physics does not do.

---

## What holds up

Worth recording explicitly, so the audit is not read as wholly negative:

- **Ordering is right.** Dispose-then-fetch (S07–S09 before S10–S12) means the robot never holds
  two bulbs, and the hand is free exactly when it needs to be. Alternative orderings would need a
  two-object grasp.
- **The hand is free at every boundary that requires it** — S05 (climb), S09→S10 (after disposal),
  S15 (descend). Checked each.
- **Every object exists before it is needed** and nothing is consumed twice.
- **The fixture is empty exactly between S06 and S14**, which is the interval it should be.
- **The ladder stays where S04 put it** for the rest of the chain, and every subtask downstream
  derives its geometry from the ladder's live pose rather than `LADDER_POSITION` / `TOP_ROBOT_POSITION`
  — the four plans that touch it (S05, S07, S13, S15) each say so.
- **Reach is plausible at height**: `STEP_LADDER_TOP_OFFSET[2] = 1.70` plus
  `G1_OVERHEAD_REACH = 1.3738` gives 3.07 m against `CEILING_FIXTURE_Z = 3.0`. Thin, but positive —
  and S14 is told to re-measure it from the *on-ladder* stance rather than a standing one, which is
  where that 7 cm of margin will actually be decided.
