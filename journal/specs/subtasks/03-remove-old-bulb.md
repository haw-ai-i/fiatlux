# S03 — Remove the old bulb while balancing on the ladder

`FIATLUX-S03-RemoveOldBulb-v0` · mode **grasping on balance** · object **old_bulb**
Read `00-foundation.md`, `ABSTRACTIONS.md`, `CONTINUITY.md` and `CRITIQUE.md` first. **Blocked on #54 — read the Blockers section first.**

## Objective

Standing on the upper steps, take hold of the seated old bulb, free it from the inverted fixture,
and end holding it — without falling, without tipping the ladder, and without crushing it.

## Start state

`START_STATE["S03"]` = S02's success state:

- Robot on the ladder's upper steps, pelvis at the top-step height, balanced, hands free.
- Ladder standing at the fixture target, at rest.
- Old bulb **seated** in the fixture: pose == the fixture's own pose, both halves being authored
  assembled at identity (`SOCKET_SEAT_OFFSET == BULB_PLUG_OFFSET == (0, 0, 0.036259)`).
- Fixture **inverted** (`_quat_y_deg(180)` for a ceiling mount), so the bulb hangs downward.

## Success gate

`old_bulb_taken` — all of:

| Condition | Value | Source |
|---|---|---|
| bulb cleared the seat | `REMOVAL_CLEARANCE = 0.10` m | `old_bulb_removed` |
| **bulb held in the hand** | hand↔old_bulb contact > 2 N | filtered channel |
| bulb not dropped | `old_bulb_dropped` false | existing |
| robot not fallen | `FALL_MIN_HEIGHT`, `FALL_TILT_LIMIT` | base |
| ladder not tipped | `LADDER_TILT_LIMIT = 0.6` | existing |
| sustained | 0.5 s | `sustained` |

**The held condition is the entire subtask.** Clearance alone is satisfied by the bulb *falling
out of an inverted socket under gravity* — which, with #54 unimplemented, is exactly what happens
at t=0 with no action at all. A clearance-only gate would score a 100% success rate for a policy
that does nothing. Do not ship the gate without the held condition, and verify by running a
zero-action rollout: it must score 0.

## Fragility

Grip force on the filtered hand↔old_bulb channel, two bounds, no geometric attribution through
the robot body:

- `GLASS_CONTACT_LIMIT_N = 50.0`
- `CAP_CONTACT_LIMIT_N = 300.0`

Real reference: IEC 60968 specifies a 3 N·m cap-to-glass torsion test for E26/E27, which is what
makes the cap tolerant and the glass not. The sensor cannot attribute contact per feature (the
bulb is one rigid body, so PhysX reports per body, not per collider), so the two bounds are
applied as thresholds on the same channel — the decision already taken for the verification
scenarios. Note the limitation at the cfg site.

## Rewards

- `reach_progress` = `distance_progress(distance_fn=palm_old_bulb_distance)`.
- `removal_progress` = `distance_progress(distance_fn=old_bulb_fixture_clearance,
  away_threshold=REMOVAL_CLEARANCE)` — the away-from formulation, because the bulb starts *at* the
  fixture so `d0 ≈ 0` and the normalized form would divide by ~zero. This is why
  `away_threshold` exists.
- `removal_bonus` = `completion_bonus(predicate_fn=old_bulb_taken)`.
- `contact_penalty` = `hand_contact_force_l2` on the filtered channel.
- `old_bulb_dropped` penalty, `ladder_tipped` penalty, `termination_penalty`, base shaping.

## Terminations

`time_out`, `success=old_bulb_taken`, `fell_below`, `fell_over`, `ladder_tipped`,
`old_bulb_dropped`. `episode_length_s = 30.0`.

## Reuse

`old_bulb_removed`, `old_bulb_fixture_clearance`, `_old_bulb_plug_point_w`, `old_bulb_dropped`,
`REMOVAL_CLEARANCE`, `OLD_BULB_DROP_HEIGHT = 0.15`, `object_contact_forces`,
`hand_contact_force_l2`, `HAND_CRADLE_DEX3`, `ARM_CRADLE`, `distance_progress`,
`completion_bonus`. New: `palm_old_bulb_distance`, `old_bulb_taken`.

## Visual start-state validation

Frames must show: the robot balanced on the upper steps, both feet on a step (not one in the air,
not a foot through a step); the old bulb hanging in the inverted fixture, cap up, glass down,
**not intersecting the socket collar**; the bulb within the robot's reach from that stance; and
the bulb visible in the ego camera from the on-ladder pose. Use `--record_view fixture`.

Then the critical one: **run a zero-action rollout and watch what the bulb does.** If it falls out
immediately, report that as the #54 blocker manifesting, and record the fall time.

## Deliverable for the chain

**`BULB_IN_ROOT_ON_LADDER`** (foundation probe 2) — the bulb's root pose in the robot root frame
for the on-ladder cradle grasp. S04 starts from it, so S03 owns measuring and freezing it. The
on-ladder stance carries a torso lean, so this is *not* the standing value from probe 1.

## Blockers

- **#54 (bulb attach/detach)** — the fixture is inverted and nothing retains the bulb. Author the
  subtask, validate the start state, measure the fall, and mark the success gate **provisional**
  until #54 lands. Do not work around it by making the bulb kinematic: a kinematic bulb cannot be
  moved by any action, which is what made `FIATLUX-Remove-v0` unsolvable.
- **`verify_interactions --scenario socket` is 3/6**, all three failures being bulb↔socket
  contact — the same root cause. Do not treat those as S03 regressions.
- **PR #64** — the held condition and the fragility bounds both read the filtered channel.

## As built

`subtasks/s03_remove_old_bulb_env_cfg.py`, on `subtask_tiers.mate.MateSubtaskCfg` — the balance
tier plus manipulation channels, since this is manipulation performed while balancing. Constructs.

The held conjunct is in the gate, as this plan insists. The zero-action rollout that must score 0
has not been run.

The fragility bound moved out of the gate and into a termination (`mate_terms.grip_force_exceeded`
at `GLASS_CONTACT_LIMIT_N`), plus a matching penalty. As a success conjunct it would have been read
only at the scoring step, so a bulb crushed on the way and taken anyway would still score.

No `ladder_contact` bootstrap: it pays for limbs *on* the ladder, and the job here is to get a hand
off it and onto the fixture.

`BULB_IN_ROOT_ON_LADDER` — this subtask's deliverable for the chain — is still the uncalibrated
geometric estimate in `grasp_poses.py`. Not measured.

### Fixed: the working stance spawned inside the fixture on ceiling draws

Reported as "the robot intersects the fixture and just hangs on it", and it did. The stance puts
the pelvis at 1.967 m and the fixture hangs at 2.200 m, so the fixture is at the robot's *chest*,
not overhead — the robot's torso reaches 0.152 m out from the pelvis axis at that height, and its
tallest body 0.463 m above the pelvis. `_sample_fixture_mount` then anchored a **ceiling** mount's
ladder zone at the point directly beneath the fixture, and `stand_robot_on_ladder_top` stands the
robot over the ladder's root, so the socket spawned inside the torso: `imu_in_torso` measured
**0.039 m** from the fixture axis. The socket is a kinematic rigid body with an exact triangle-mesh
collider, so the robot did not fall through it — it hung there. Under zero action the pelvis moved
1.967 → 1.950 m over 3 s, where the same stance on a wall draw collapses to 1.194 m.

Roughly half of all layouts, since the mount kind is a coin flip: **13 of 24 seeds** sampled, all
13 at `dxy = 0.000`. It hit every on-ladder leaf, not just this one — S02, S04, S10, S11 and S12
all take the same stance.

Fixed in the sampler, not in the stance: a ceiling mount now stands its ladder anchor
`LADDER_FIXTURE_STANDOFF` (0.60 m) off the point beneath the fixture along a sampled bearing, the
way a wall mount already stood off from the wall face, and the coupled ladder yaw turns back down
that bearing so the stance faces the fixture on either mount kind. `scene_cfg` carries a new
import-time guard tying the standoff to the measured clearance it has to buy
(`LADDER_FIXTURE_MIN_STANDOFF` = torso 0.152 + fixture 0.081 + reset jitter 0.071 = 0.304 m).

Consequence to accept: a ceiling fixture is now 0.60 m away in the floor plane instead of 0 m, so
it is exactly as hard to reach as a wall one — and `scene_cfg`'s standing caveat that the honest
forward reach (`G1_HORIZONTAL_REACH` = 0.5045 m) is shorter than that standoff now applies to both
mount kinds rather than only to wall draws. The reach was never *usable* on a ceiling draw before
this, since the fixture was inside the robot; what changes is that the open question is now
uniform.
