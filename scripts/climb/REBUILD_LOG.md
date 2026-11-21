# Climb reference clip: rebuild log

One entry per rebuild of the S02 climb reference clip made with `build_edgehold_motion.py`.
The clips themselves are not in the repository; each entry gives the flags that reproduce it.

## Shared flags

Every rebuild below uses these, plus what its entry lists:

```
--hold_tol 0.02 --roll_z 1.50 1.74 --release_at 819
--climb 47 138 219 300 386 468 575 645 752 818
--crouch 0.0 --wrist_yaw_max 89 --head_w 1.0
--pause 44:0.5 --pause 146:5.0 --pause 302:4.4
--upright2_on 325 400 --upright2_off 550 590
--hand_y 0.115 0.115 --hand_yc 0.02 --open_lift 0.0
```

`--base` is the retargeted LadderMan clip (`topout`), `--mesh` the ladder's collision triangles.

## Checks

| Name | Tool | What passes |
|---|---|---|
| Gate | the builder's own `[gate]` lines | every line PASS |
| Hands | `check_hand_collision.py --stride 5` | no hand part inside the ladder, the other hand or the forearm |
| Torso | `check_arm_torso.py` | elbow and wrist-roll hulls outside the torso hull |
| Legs | `check_leg_collision.py` | shin and thigh hulls outside the ladder (parts above the foot's own tread) |
| Scan | `physics_segment_test.py --body_scan 10` | no body pushed with more than 400 N one physics step after staging a pose |
| Frozen | `physics_segment_test.py --hold_steps 150` | robot keeps the staged pose for 3 s on plain joint position control |
| Static | `static_support_check.py --foot sole --mass 34.40` | forces exist that hold the pose within friction and joint torque limits |

The frozen test has no balance controller. A fall there does not prove a pose is bad, and a pass
does not prove it is good: in rebuild 52 the shins were resting inside the treads.

## Rebuild 52 (baseline for this log)

Flags: shared only.

- Gate: all pass. Hands: clean.
- Legs: FAIL. Shin inside the tread above for 22.4 s (left) and 21.2 s (right) of 29.6 s, 13 s
  deeper than 10 mm, deepest 36 mm. Thighs about 4 s each.
- Torso: FAIL. Left elbow 23 mm inside the torso hull at the start and at steps 283-466.
- Scan: torso pushed at 13 of 148 poses (up to 773 N); left shin up to 9,603 N and left thigh up to
  14,175 N at each left step-up.
- Frozen: 7 of 7 poses hold 3 s. Not trustworthy, see above.

## Rebuild 56: forearm kept clear of the torso

Change: the planner measures the forearm against the torso's hull (`torso_clear`, `--torso_want`)
when choosing and steering arm poses. Flags: shared only.

- Gate: fails only the new torso line (-18.5 mm at step 0). Hands: clean.
- Torso: left elbow inside during steps 0-4 and 306-362 (was 0-29 and 283-466); right arm 6-7 mm.
- Scan: torso pushed at 10 poses (was 13): step 0 (413 N), 310-340 (about 500 N), 680, 1370-1400.

## Rebuilds 57 and 58: rejected

- 57 (elbow steered every frame, more iterations): start overlap came back (steps 0-33), a wrist
  went behind the body, two hand grazes. Reverted.
- 58 (fresh arm pose for the first two opening waypoints; release forward instead of back): start
  unchanged, thumbs grazed the handrail at release. Release change reverted; the opening code stays
  but did not trigger.

## Rebuild 60: stand further back on every tread

Change: `--tread_back` moves the feet and the pelvis back together on each tread, `--ground_back`
does the same for the standing start.
Flags: shared + `--tread_back 0.055 0.055 0.055 0.07 --ground_back 0.055`.

- Gate: all pass, including the torso line (+7.5 mm). Hands: clean.
- Legs: clean, no overlap anywhere.
- Torso: elbows clear; left wrist-roll link up to 10 mm inside at steps 227, 635-652, 1377-1395.
- Scan: no shin, thigh, elbow or torso contact. A foot is pushed as it lands at steps 160 (542 N),
  1000 (3,239 N), 1020-1030 (3,049 N), 1200 (1,763 N).
- Frozen: 4 of 7 hold. Falls: standing on step 2 with one hand (1.3 s), step 2 to 3 (1.8 s),
  standing on step 4 (1.5 s). Feet shift 41-69 mm even where it holds. The standing start stays up
  but leans 234 mm onto the ladder.
- Static: with hands, most poses hold at 15-42% of joint limits and the body's weight is 4-52 mm
  ahead of the feet (28-75 mm in rebuild 52). No hold exists at steps 565-600, 880-885 and
  1330-1345 (weight behind the single stance foot). Only 19-29 mm of sole is on tread 4.
- Verdict: legs fixed, but too far back, on tread 4 most of all.

Other things tried on the legs alone (no full clip):

| Variant | Left shin deeper than 3 mm | Notes |
|---|---|---|
| 35/35/35/50 mm | 2.4 s | all of it mid-swing on left steps |
| same + `--swing_back 0.03` | 2.0 s | |
| 45/45/45/60 mm | 1.1 s | |
| same + `--swing_hip_back 0.02` | 1.1 s | easing the hips back does nothing |
| 55/55/55/70 mm | 0.3 s | only the first step off the ground |
| same + `--ground_back 0.055` | 0 s | this is rebuild 60 |

## Rebuild 61: middle setting

Flags: shared + `--tread_back 0.045 0.045 0.045 0.05 --ground_back 0.045`.

- Gate: fails only the torso line by 0.9 mm (left forearm at step 1384, the release). Hands: clean.
- Legs: right leg clean; left shin grazes the next tread mid-swing on each left step (0.1-0.5 s
  each, up to 12 mm).
- Torso: elbows clear; left wrist-roll link up to 19 mm inside at steps 218-231, 635-652, 1372-1402.
- Static: 55-75 mm of sole on treads 1-3 and 39-53 mm on tread 4 (rebuild 60: 19-29 mm on tread 4).
  With hands, no hold exists at steps 565-595 and 1330-1345 (weight 11-15 mm behind the single
  stance foot) and in the short windows where neither foot counts as planted.
- Not run in the simulator; superseded by rebuild 62.

## Teacher training started on rebuild 60 (2026-11-20)

Two cold-start runs of the LadderMan teacher on Koa, job scripts in `ladderman_port/koa/`:

| Job | Partition | Seed | Notes |
|---|---|---|---|
| `lm-edge60` | ece605 (RTX A4000) | 0 | 24 h wall cap; resubmit to resume |
| `lm-edge60h` | kill-shared, H100/H200 nodes | 1 | preemptible; `--requeue` resumes it |

Both: 4,096 environments, 7,500 iterations, checkpoint every 300. On a resume the script reads the
newest `checkpoint_N.pt` of its own run family and trains only the remaining 7,500 - N iterations.
Rebuild 60's known weak points (three frozen poses fall, little sole on tread 4) apply.

## Rebuild 62: rejected (toe levelling, release out to the side)

Flags: shared + `--tread_back 0.045 0.045 0.045 0.05 --ground_back 0.045 --flat_feet 1.0`, with a
release path that went out to the side before coming down to rest.

- Gate: REJECTED (arm speed 6.20 rad/s, torso -21 mm, moving hand 19 mm from the ladder).
- Hands: 4 grazes (thumbs at the release, a finger on a side rail at step 219).
- Legs: worse than 61. Left shin 3.2 s deeper than 3 mm (up to 30 mm), left thigh 0.5 s. Levelling
  the toes lowers the body and the shin leans further.
- Scan: a foot pushed with 0.6-12.5 kN at 30 of 148 poses. Levelling during landing and lift-off
  puts the sole inside the tread.
- Frozen: the same three poses fall as in rebuild 60.
- Outcome: `--flat_feet` stays as an option but is not used; the release change is reverted.

## Why the frozen poses fall: sideways, onto the left foot

Read from the recordings of the frozen test on rebuilds 52 and 60:

- The three poses that fall in rebuild 60 all fall to the robot's LEFT (pelvis +255 to +704 mm in y).
  Rebuild 52 drifted the same way by 50-64 mm and was stopped by the shins in the treads.
- The feet are only 95-105 mm apart, and the body's weight sits at y +37..+52 mm, over the left
  foot (left ankle +44..+54, right -40..-60). The base clip has the same.
- Both ankle-roll joints are pinned at their +-0.26 rad limit for 88% of the clip and the soles are
  tilted sideways (left 4-6 deg, right 11-14 deg): the feet stand on their edges.

Two builder options follow from this: `--level_roll` (remove the sideways tilt with the ankle-roll
joint; ankle roll falls to +0.22 / -0.14 rad) and `--centre_weight` (move the pelvis sideways to
the midpoint of the feet while both are planted, feet held; backs off where ankle roll would pass
its limit). Asking for the weight over the single stance foot as well needs 100 mm of pelvis
travel and ankle roll past its limit, so that is not done.

## Rebuild 63: feet level sideways, weight centred (best so far)

Flags: shared + `--tread_back 0.055 0.055 0.055 0.07 --ground_back 0.055 --level_roll 1.0
--centre_weight 1.0` (rebuild 60 plus the two new options).

- Builder: sideways foot tilt 0 (was up to 12.8 deg left, 15.8 deg right); ankle roll +0.22 / -0.14
  rad at most (was pinned at +-0.26); pelvis moved up to 60 mm sideways; weight off the midpoint of
  two planted feet: median 0 mm (was 40), max 33.
- Gate: all pass, ACCEPTED. Hands: clean. Legs: clean.
- Torso: left arm clear. Right wrist-roll link up to 9 mm inside at steps 745-758 and 1379-1400.
- Scan: a foot pushed at steps 160 (5,231 N) and 1020 (1,456 N); right wrist against the torso at
  1380-1390 (about 470 N). Nothing else.
- Frozen: 6 of 7 hold 3 s (rebuild 60: 4 of 7). Standing on step 2 with one hand and standing on
  step 4 now hold. Step 2 to 3 (finger pads on the handrail edge) still falls, at 2.1 s, sideways.
  The standing start stays up but leans 231 mm forward onto the ladder.
- In every on-tread pose the ankles sink 40-59 mm and slide 16-20 mm back in the first 0.25 s, then
  stay. The clip plants the feet toe-down (7-17 deg) on flat treads with the ankle ahead of the
  tread's edge; under load the foot rotates flat.
- Static, with hands: no hold exists at steps 150-165, 565-595 and 1330-1345 (weight 24-33 mm
  behind the single stance foot just after it lands).

## Rebuild 64: toes level too (all seven frozen poses hold)

Flags: rebuild 63's + `--level_pitch 1.0`.

- Builder: toe-down pitch removed on the right foot (was up to 44 deg in swing, 7-17 deg planted);
  the left keeps up to 15 deg where its ankle-pitch joint reaches the limit (-0.85 of -0.87 rad).
  The pelvis rides a median 46 mm lower, because a level sole puts the ankle closer to the tread.
- Gate: REJECTED on one line, the torso (right forearm -10.5 mm at step 1384, the release).
  Hands: clean.
- Legs: right leg clean. Left shin grazes the next tread for about 0.5 s on each of the four left
  steps (2.1 s in total, up to 26 mm); left thigh 0.4 s. Stance itself is clean.
- Torso: right wrist-roll link up to 22 mm inside at steps 741-743 and 1375-1408.
- Scan: a foot pushed as it lands or leaves at steps 150-160, 880 and 1410-1420 (4-14 kN); left
  shin 2,328 N at step 960; right wrist against the torso at 1380-1400 (about 580 N).
- Frozen: 7 of 7 hold 3 s, the first clip to do so with the stance legs clear of the ladder. Feet
  settle 22-33 mm (rebuild 63: 41-49 mm) and the pelvis 8-36 mm. The standing start still leans
  225 mm forward onto the ladder.
- Static, with hands: no hold at steps 145-165, 565-580, 1155-1165, 1340-1345 (weight 20-45 mm
  behind the single stance foot just after it lands).

Where the shin graze comes from: the left foot lands on the next tread while the hip is still low
and about 100 mm behind the ankle, which puts the knee 150 mm ahead of the ankle, at the nose of
the tread above. It clears as the hip rises. Tried on the legs alone:

| Variant | Left shin deeper than 3 mm | Notes |
|---|---|---|
| `--swing_hip_up 0.03` / `0.05` (option since removed) | 2.0 s / 1.9 s | trailing leg cannot reach by 70-137 mm |
| 65/65/65/75 mm back | 1.4 s | none deeper than 20 mm; this is rebuild 65 |
| 75/75/75/80 mm back | 0.7 s | only 51-71 mm of sole left on the tread |

## Rebuild 65: rejected (65 mm back is too far)

Flags: rebuild 64's with `--tread_back 0.065 0.065 0.065 0.075 --ground_back 0.065` and
`--release_up 0.0 0.0 0.13` (let go of the handrail straight up).

- Gate: REJECTED. The left hand leaves its handrail hold for 0.01 s during the third foot step.
- Hands: two finger links dip 3-4 mm below the handrail's top face at step 975.
- Legs: left shin graze down to 1.4 s (up to 18 mm), left thigh 0.3 s.
- Torso: the release is clean now; one frame at step 743 with the right wrist 3 mm inside.
- Frozen: 6 of 7. Step 2 to 3 falls again at 2.2 s.
- Static: weight up to 53 mm behind the single stance foot (rebuild 64: 45 mm).
- Outcome: the extra 10 mm of stand-back costs the handrail reach and the step 2 to 3 hold. The
  release change is kept; the stand-back goes back to 55/55/55/70.

## Rebuild 66: rebuild 64 with the clean release (accepted)

Flags: shared + `--tread_back 0.055 0.055 0.055 0.07 --ground_back 0.055 --level_roll 1.0
--level_pitch 1.0 --centre_weight 1.0 --release_up 0.0 0.0 0.13`.

- Gate: all pass, ACCEPTED. Hands: clean.
- Torso: clean at the release. Right wrist-roll link 3 mm inside for steps 741-743.
- Legs: as rebuild 64. Left shin 2.1 s (up to 26 mm) and left thigh 0.4 s, all as the left foot
  lands; right leg clean.
- Scan: a foot pushed at steps 150-160 (4-7 kN), 880 (10 kN) and 1410 (4.8 kN); left shin 2,328 N
  at step 960. No arm or torso contact.
- Frozen: 7 of 7 hold 3 s. Hands carry 12-13% of body weight on the flat-palm holds and 3% on the
  step 2 to 3 finger-pad hold. Feet settle 22-33 mm. The standing start leans 225 mm forward.
- Static: as rebuild 64.
- Planted soles sit between -6.5 and +4.4 mm of their surface; the negative end is at the edges of
  stances and matches the scan's foot pushes.

## Rebuild 67: soles never below the tread (best so far)

Flags: rebuild 66's + `--no_sink`.

- Builder: soles were up to 10.3 mm (left) and 5.1 mm (right) below their surface on 9-12% of
  frames, at the edges of stances; lifted. Planted soles now sit between -0.9 and +6.5 mm.
- Gate: all pass, ACCEPTED. Hands: clean.
- Torso: right wrist-roll link 3 mm inside for steps 741-743; nothing else.
- Legs: unchanged. Left shin 2.1 s (up to 26 mm) and left thigh 0.4 s as the left foot lands.
- Scan: ONE entry in 148 poses: left shin 2,328 N and left thigh 1,011 N at step 960. All the foot
  pushes of rebuild 66 are gone.
- Frozen: 7 of 7 hold 3 s.
- Static, with hands: no hold at steps 145-165, 565-580, 1155-1165 and 1340-1345 (weight 20-45 mm
  behind the single stance foot just after it lands), and in the brief windows where neither foot
  counts as planted.

Tried after rebuild 67, on the legs alone: `--left_extra` (the leading foot a further 20 or 40 mm
back, pelvis unchanged) left the shin graze at 1.8 s and 1.6 s, and at 40 mm the leg could not
reach at the top-out. Not kept.

## Rebuild 67 staged for training, not submitted (2026-11-20)

- Trains for two iterations in a local 64-environment smoke test.
- The clip and its anchors are on Koa next to rebuild 60's; job scripts
  `ladderman_port/koa/lm_train_edge67.sbatch` (ece605) and `lm_train_edge67h.sbatch`
  (kill-shared, H100/H200) are copies of the rebuild 60 scripts with the clip changed.
- Resume checked locally on rebuild 67: a 5-iteration run saved `checkpoint_2`, `checkpoint_4`; a
  second run given `checkpoint_4.pt` printed `[iter 4]` first and saved `checkpoint_5.pt`. The job
  scripts' checkpoint glob skips `checkpoint_final.pt`.

## Rebuild 60 training, first hours

`lm-edge60` on ece605: 850 of 7,500 iterations after 2 h 38 min of training, about 11 s per
iteration, checkpoints at 300 and 600. That pace needs about 23-24 h, so it may stop just short
of 7,500 at the partition's 24 h cap; resubmitting the same script finishes it. ece605 allows one
submitted job per user, so the follow-up cannot be queued in advance. `lm-edge60h` was still
pending on priority.

## Rebuild 60 teacher: what it learned and why it fails (2026-11-20)

`lm-edge60` was cancelled at iteration ~3,700. Its training log:

| Iteration | Success | Episode length | Old topout cold start: success | length |
|---|---|---|---|---|
| 950 | 5.5% | 41 | 10% | 83 |
| 1,900 | 6.1% | 50 | 14% | 110 |
| 2,850 | 6.2% | 54 | 70% | 362 |

Played back locally at iteration 3,000 with two new probe scripts (`ladderman_port/play_probe_rsi.py`,
`play_probe_hand.py`; both go in LadderMan's `scripts/`):

- From the ground (8 robots): best pelvis gain 43 mm; all terminated by step 173 with the pelvis
  23-28 cm below the reference.
- Random start phase (5,183 episodes): 6.7% reach the clip end, all of them starts after step
  1,300. Median episode 30 steps. `cum_ladder_lost_contact_steps` (30-step limit) ends 90% of
  failed episodes.
- With that rule switched off at playback: median 60 steps; `cum_body_z_error` 77%,
  `cum_foot_any_lost_contact` 18%. The policy never saw episodes past 30 steps, so this is
  partly out of distribution.
- `cum_foot_any_lost_contact` ends an episode after 60 steps with either foot unsupported,
  whatever the reference says. Rebuild 60's right-foot swings last 70, 79 and 65 steps; rebuild
  67's swings last 56-59.

Hand rule, per condition (share of labelled hands passing):

| | Topout teacher | Rebuild 60 teacher |
|---|---|---|
| Force up > 5 N, 2 steps after reset | 4-18% | 28-47% |
| Force up > 5 N, 20 steps in | 76-87% | 21-61% |
| Hand point above target, 20 steps in | 83-98% | 35-79% |
| Reference only, 20 steps: rule counting | 77% | 90% (rebuild 67: 89%) |

The topout reference also sags and trips the rule when played alone; its policy learned to load
the hands within 20 steps (median 13-30 N). Ours does not: on the handrail the hand point is a
median 4.5 cm below its target and 11 cm back after 20 steps. The topout run used LadderMan's
stock G1 hooked over rungs; ours has flat open palms and reference-driven wrists. Which
difference is the blocker is not established.

## Rebuild 68: hands pressed into their holds -- rejected

`--press 0.015` (new): after the gates, a holding hand's wrist is lowered 15 mm, ramped inside
the hold. Clip gate accepted, peak arm speed 6.01 rad/s. In the trainer it is worse than rebuild
67: the wrists keep their angles, so the finger-pad point ends 24 mm inside the tread in the
clip, and a reset that starts there is kicked by the simulator -- two steps later that point is
67 mm below the tread top, and hands loaded on the platform fall from 44-57% to 29-33%. The
policy's arm targets are also `default + action`, not offsets from the reference, so a pressed
reference does not make the controller push. Not trained.

## Rebuild 69: rebuild 67 with lower hold anchors -- training

The rebuild 67 motion unchanged, anchors 40 mm lower (`make_anchors.py --drop 0.04`). Under
reference playback at 20 steps the hand point is above its target for 24-45% of hands, against
14-31% with the original anchors. Submitted as `lm-edge69` on ece605.

## Koa nodes

The H100/H200 nodes (`gn-09-19-00`, `gn-09-21-00` tried) hang this job at simulator startup: one
thread at 100% CPU, GPU idle, no log line after the `omni.graph.visualization` warning; three
attempts, one left for 2 h 12 min. Rebuild 67 now runs as `lm-edge67h` on kill-shared with those
nodes excluded and `no_mig` required.

## Why the reference drops in the trainer: the hand was one convex wedge (2026-11-20)

`ladderman_port/play_probe_drop.py` plays the reference open-loop from the training reset and prints
support, pelvis error and per-hold hand forces step by step. Rebuild 67, 1,024 random phases.

Ruled out, each by switching it off: reset noise, action delay, the zero action buffer at reset
(the PD target is the default standing pose for the first 2-6 substeps), ankle stiffness (5x stiffer:
vertical sag 44 -> 11 mm at step 10, backward drift unchanged at 59 mm), and any kinematic mismatch
(robot bodies equal the clip's at the reset frame, median 0.0 mm).

Found: on the first step each hand on tread 4 or the platform takes about 100 N pointing back and
up (x -62, z +76 N median) and is thrown 47 mm back and 19 mm up within four steps; on the handrail
10% of resets see over 4,000 N. Cause: the URDF importer puts all of a link's collision meshes under
one `collisions` prim with `PhysxMeshMergeCollisionAPI`, so the rigid open Dex3 (wrist, palm, two
fingers, thumb hanging 100 mm below the palm) is ONE convex hull -- PhysX reports a single shape,
235 x 143 x 87 mm. The clip's holds sit 37 mm inside that hull; with one hull per hand part, as in
the benchmark, they clear it by about 3 mm.

Fix: `ladderman_port/g1_fiatlux.py` now spawns through a wrapper that, after URDF conversion, gives
each collision mesh of the two hand links its own convex hull (9 shapes per hand link, 48 on the
robot, was 32).

Reference only, no noise / delay, rebuild 67, medians:

| Step | Hands carry (% weight) before -> after | Pelvis back (mm) before -> after | Pelvis low (mm) before -> after |
|---|---|---|---|
| 4 | 1 -> 11 | 13 -> 2 | 14 -> 15 |
| 10 | 2 -> 11 | 49 -> 6 | 31 -> 29 |
| 25 | 0 -> 9 | 127 -> 5 | 44 -> 27 |

Hand rule under reference playback at 20 steps (as shipped, with noise): counting toward
termination in 63% of robots, was 89%; hands pressing over 5 N on tread 4 and platform 65-71%, was
10-41%; on the handrail 34-39%, was 17-35%.

Every teacher trained so far on the open Dex3 (rebuild 60, and the rebuild 67 / 69 runs started
before this) trained against the wedge. An earlier version of the probe commanded a stale pose on
the first step after reset; `cmd.update()` after `env.reset()` fixes it, and the numbers above are
from the fixed probe.

## Restart on the fixed hand; friction randomisation limit (2026-11-20)

The first restart with one hull per hand part logged 41,787 `PxPhysics::createMaterial: limit of
64K materials reached` errors at startup. LadderMan's `perturb_body_materials` draws friction per
SHAPE of the ankle-roll and wrist-yaw links; those links went from 6 shapes to 22, and
22 x 4,096 environments is past PhysX's 65,536 materials, so about a quarter of the randomised
shapes kept the default material. Cancelled at iteration 33.

The job scripts now pass `+task.randomization.perturb_body_materials.homogeneous=true` (an option
the class already has): one friction draw per robot, shared by its feet and hands, at most 4,096
materials. Ranges unchanged.

Runs on Koa set aside, not deleted: `teacher/wedge_hand/` (everything trained on the one-hull
hand) and `teacher/matlimit/` (the 33-iteration run above).

Startup hang: also seen on a Quadro RTX 5000 node (`gn-16-22-00`, 76 min, flagged by the cluster's
0% GPU email). The job has only ever started on A4000 cards, so the kill-shared copy is pinned
to `gres/gpu:nv-rtx-a4000`.

## Baseline: LadderMan as published (2026-11-20)

Replayed the 17 Nov teacher (their reference clip and stock G1, rung ladder at our 70.3 deg /
249 mm, cancelled at iteration 2,750) from the ground in their env: 15 of 16 robots follow the
clip to its end at step 710, pelvis gain 460-519 mm against the reference's 539 mm.

`lm-stock` (`ladderman_port/koa/lm_train_stock.sbatch`) trains the fully stock case: a
`git archive` of upstream commit 9a8d5ed unpacked as `LadderMan_stock/` on Koa (103 files, none
of ours), no task or algo overrides, their default 7,500 iterations, kill-shared A4000. The job
refuses to run unless `active_adaptation` imports from that tree.

## First climbing on the step ladder; PhysX pair buffers (2026-11-21)

`lm-edge69` (rebuild 67 motion, anchors 40 mm lower, one hull per hand part) at iteration 950:
success 10.4%, episode length 143 (rebuild 60 on the wedge hand: 5.5%, 41). Played from the
ground at iteration 900, 16 robots: 11 reach tread 1, 8 reach tread 2, best pelvis gain 621 mm.
Nearly every termination is `cum_ladder_lost_contact_steps` at a hand change: steps 81-87
(tread 4), 316 and 438 (platform), 779-821 and 1,008-1,011 (handrail).

The run then stopped advancing at iteration 1,196: `PxGpuDynamicsMemoryConfig::
foundLostAggregatePairsCapacity` and `foundLostPairsCapacity` errors every 12 s, GPU idle.
LadderMan hard-codes these buffers for its own robot (2,538,320 pairs, 61,999,079 aggregate
pairs); PhysX asked for 2.63M and 65.7M once most robots were up on the ladder with 48-shape
robots. `ladderman_port/ladderman_upstream.patch` (our changes to LadderMan's own files) adds
`task.sim.physx_capacity_scale`, default 1.0; the edge job scripts pass 2.0. Resumed from
checkpoint 900.

## Rebuilds 70 and 71: hands stay on the platform for the tread 2 -> 3 step -- rejected (2026-11-21)

Idea: keep both hands on the platform through the step from tread 2 to tread 3 and move them to
the handrail afterwards, so the handrail reach (where the teacher is being terminated) is made
from a higher stance. Only the third `--pause` moves; base-clip timing of that step: left foot
lands on tread 3 near base step 432, right foot leaves tread 2 near 445 and lands near 480.

Plan-only reach (limit 0.375 m), both feet on tread 3 (`--pause 535:4.4`):

| Crouch | Platform needs | Handrail needs |
|---|---|---|
| 0 | 0.615 | 0.249 |
| 0.20 | 0.461 | 0.303 |
| 0.27 | 0.418 | 0.350 |

No pelvis height reaches both. More waist lean does nothing: the waist is at 0.50 of 0.52 rad.

- Rebuild 70 (`--pause 470:4.4 --crouch 0.24`, crouch 400-468 on, 472-540 off): REJECTED. The
  pause fell while the right foot was in the air. Right forearm 13 mm and left wrist-roll 14 mm
  inside the torso; left hand into a stringer during its move (steps 823-835); thighs 37 / 42 mm
  inside the ladder for over 6 s (steps 720-1,067); right shin 36 mm into the tread above.
- Rebuild 71 (`--pause 440:4.4`, no crouch; hands change in the split stance, left foot on tread
  3 and right on tread 2): REJECTED. Reach in the stance itself is fine (platform 0.371 / 0.378,
  handrail 0.351 / 0.325), but during the left foot's swing before it the right hand cannot stay
  on the platform: 65 mm off its hold at steps 670-693 with the palm tilted 19 deg, right forearm
  8.6 mm inside the torso at step 694. The body motion in that swing was authored with the hands
  on the handrail.

Not trained. Making it work needs the body trajectory of the left-foot swing reshaped, not a flag.

## Why tread 2 -> 3 is the weak point, and rebuild 72 (2026-11-21)

`ladderman_port/play_probe_window.py`: from the ground, iteration-900 teacher, 64 robots, what
each hand satisfies of the hold rule step by step.

- Steps 750-800: 17 of 28 surviving robots terminated. The left hand has been on the handrail
  since step 723 and is within range and above its target in 100% of robots, but its force passes
  the rule in only 7-50%: it pushes on the rim sideways, median (-18, +2) N. The feet carry 100%
  of the weight and the pelvis is within 1 cm of the reference. The right hand is in transit for
  104 steps; the rule allows 30. Once both hands are on (step 850+) the left passes in 75-100%.
- Mount, steps 82-90: 19 of 64 terminated; the right hand on tread 4 passes the force test in
  25% at step 60, 53% at 70, 67% at 80. That one is learning speed, not structure.
- The other hand changes load within 10-20 steps.

Rebuild 72 = rebuild 69 with labels only changed (`relabel_handrail.py`): a hand's handrail label
starts when the second hand's does, plus 0.4 s (left 723 -> 851, right 831 -> 851). Same
checkpoint replayed on it, never trained on it: terminations at steps 750-800 go from 17 to 0;
23 robots reach step 900 (was 8). They then fail in the step itself: 6 at 900-950 and 13 at
1,000-1,050, pelvis 88 mm low at step 1,000.

Side rails for this step were considered: an open flat hand cannot grip a rail, and a palm on
an inclined rail pushes sideways, which is the same failure of the hold rule as the rim edge.

## Rebuild 72 in training (2026-11-21)

Iteration-1,200 teacher replayed from the ground, 64 robots each: on rebuild 72, 3 reach the
1,200-step episode limit still climbing (one filmed onto tread 4, pelvis +964 mm) against 1 on
rebuild 69; both lose about half at the mount and a third at the tread 4 -> platform hand change.

`lm-edge72` on ece605 continues the rebuild 69 teacher from its iteration-1,500 checkpoint (the
motion is the same). It also sets `task.max_episode_length=1600`: LadderMan's 1,200 is longer
than its own 710-step clip but shorter than ours (1,479), so an episode from the ground was cut
before the top-out. The limit is only used for truncation.

## First full climbs: rebuild 72 at iteration 1,800 (2026-11-21)

`lm-edge72`, 300 iterations after switching to rebuild 72 (1,800 in all). From the ground in the
trainer, episode limit 1,600 steps:

- 64 robots: 22 reach the end of the clip (step 1,479), standing on the platform with hands
  released, pelvis on the reference. Lost: 5 in the mount, 31 at steps 300-450 (tread 4 ->
  platform hand change), 6 between steps 600 and 1,050.
- 16 robots (video batch): 9 full climbs, best pelvis gain 1,187 mm (reference 1,179).

A from-zero rebuild 72 run (`lm_train_e72fresh.sbatch`, seed 1) is queued on kill-shared as the
clean comparison; it had not started.

## Bulb variant for S10: one-handed climb, clip `bulb01` (2026-11-21)

S10 climbs with the fresh 35 g bulb in the RIGHT hand, so that arm takes no hold. Working state
before starting: tags `climb-first-full-climb` (this repo) and `fiatlux-first-full-climb`
(LadderMan); `bulb01` is a new clip and none of the two-handed clips, scripts or runs change.

- The S10 spawn pose (ARM_CRADLE, hand 37 cm ahead of the pelvis) cannot be kept: over the
  rebuild 72 motion the hand-plus-bulb sphere is within 30 mm of the ladder in 517 of 592 sampled
  frames, up to 105 mm inside the tray, and already 41 mm inside tread 3 at step 26.
- `scan_carry_grid.py`: 2,046 of the fixed right-arm poses tried clear the ladder by 60 mm and
  the body at every frame. Three were built with `make_bulb_carry_clip.py` (right arm only; all
  other bodies rebuilt to 0.00 mm) and put through the static check with the left hand as the
  only hold (two-handed rebuild 72: 12 rows with no balance):

  | Carry pose | Rows with no balance | Feet alone during the left hand's moves |
  |---|---|---|
  | out wide to the right, forward | 16 | cannot balance (steps 200-285, 620-720) |
  | beside the right hip | 13 | balances, effort 32-40% |
  | behind the back | 19 | balances |

- `bulb01` = beside the right hip (shoulder pitch 0.80, roll -0.50, elbow 0.60, wrist roll 1.57),
  from the first frame. Bulb hand clears the ladder by at least 130 mm, forearm 101 mm; arm vs
  torso and hands vs ladder CLEAN. The left hand is labelled on a hold for 66.7% of the clip and no
  hand for 33.3% (rebuild 72: two hands 60.6%, one 19.4%, none 19.9%).
- The two-handed teacher (iteration 1,800) played on it untrained: 0 of 64 past step 400;
  33 terminated on `foot_any_lost_contact` during the mount. It needs training.
- Trainer robot unchanged for now (right hand open, no bulb body): 35 g is negligible and the
  hand touches nothing. `ladderman_port/koa/lm_train_bulb01.sbatch` is written, NOT submitted.

## Rebuild 72 teacher, checkpoints 1,800 to 2,700 (2026-11-21)

From the ground, 64 robots each, episode limit 1,600 (`play_probe_window.py`):

| Iteration | Full climbs | Lost in the mount (steps 0-200) | Lost at steps 300-450 (tread 4 -> platform) | Lost at steps 850-950 (handrail label starts at 851) |
|---|---|---|---|---|
| 1,800 | 22 | 5 | 31 | 1 |
| 2,100 | 8 | 17 | 30 | 8 |
| 2,400 | 13 | 17 | 14 | 15 |
| 2,700 | 3 | 33 | 25 | 3 |
| 3,000 | 14 | 29 | 1 | 20 |
| 3,300 | 35 | 16 | 6 | 5 |
| 3,600 | 39 | 20 | 0 | 3 |

Checkpoints swing widely (3 to 35 of 64); 3,300 is the best so far at 55%. Pinned copies of 1,800
and 3,300 are in `teacher/keep/` on Koa. The trainer's own success number has not printed since the switch.
The iteration-1,800 checkpoint is kept in the protected copy.

## Bulb variant, second design: the bulb arm supports on its wrist (2026-11-21)

Asked for: bring the bulb arm to the front and use it as support, on the wrist, not the palm, with
the bulb touching nothing. Tools: `measure_bulb_in_hand.py` (benchmark sim), `make_bulb_wristrest_clip.py`,
`render_reference_bulb.py` (draws the bulb).

- Measured at an S10 start: the bulb LIES ACROSS the right palm, axis along the wrist frame's +z,
  root at (0.119, 0.054, -0.131), glass centre 0.117 ahead of the wrist and 0.058 to the palm
  side. On a level palm-up hand it stands about 130 mm above a surface the wrist rests on, so it
  fits under the platform at tread 4 (165 mm) and in the tread gaps (199 mm). My first proxy
  (bulb upright, 180 mm) was wrong and had ruled tread 4 out.
- Reach with the rebuild 72 body motion (straight arm 0.382 m): the right wrist reaches tread 4
  from the ground and through the mount, and the platform from treads 1 and 2. Lower treads are
  0.42-1.0 m away at every phase, and the handrail 0.55 m from tread 2. Palm-up on the handrail
  from tread 3 is at the edge of the arm's range (45 mm off, 6.5 deg).
- From tread 1 the forearm cuts 17-24 mm into the platform's edge, so the platform rest starts on
  tread 2.
- Carrying beside the hip (outside the right stringer) makes every move thread round the stringer:
  misses up to 94 mm, bulb 14 mm inside it, 16 rad/s. Carried in FRONT of the chest it never
  crosses that line. Too close to the chest (wrist 130 mm ahead of the pelvis) puts the elbow
  113 mm inside the torso; a pull of the elbow outward in the solver's free direction is what
  keeps it off the torso at all.

`bulb07` (carry at pelvis-frame (0.16, -0.11, 0.14) m, palm up throughout): wrist on tread 4 for
steps 52-300 (within 2.0 mm, 3.6 deg), on the platform for 620-890 (1.4 mm, 2.1 deg). Closest
approaches: bulb 4 mm from the right stringer (step 45), back of hand 12 mm, forearm 16 mm, own legs
115 mm, own torso 0 mm (step 954, with oversize proxies). Peak arm speed 6.06 rad/s. Not trained;
the static check has not been adapted to a wrist contact; the 50 N limit is covered by geometry only.
Training it needs `contact_eef_pos_offset` for the right hand set to the wrist's contact point
(0.005, -0.030, 0.0).

### bulb07 through the full check suite (2026-11-21)

Run in the plain climb task (S02): right hand OPEN and empty, so no bulb and fingers 5-8 cm longer
than the cupped S10 hand.

- Frozen poses, 3 s each: 7 of 7 hold. Pelvis off the reference 2-41 mm, except the ground pose
  (200 mm, as in rebuild 67) and the tread-2 pause with the wrist on the platform (109 mm, tilt 12 deg).
- Whole-body scan, pushes over 400 N (rebuild 67 had only the step-960 entry): step 360 torso and
  right wrist_pitch 417 N; steps 520-530 right palm and fingers 657-2,744 N; step 960 left knee
  2,301 N and left hip_yaw 984 N; steps 1,320-1,330 right fingers up to 1,826 N and right
  wrist_roll / wrist_yaw 787 N.
- `check_arm_torso.py`: right wrist_roll 5.8 mm inside the torso at step 952; also inside during
  358-362 and 1,356-1,366. Right elbow +20 mm.
- `check_hand_collision.py` flags the right hand at the platform's front face (steps 352-362,
  533) and the tray rim (1,323-1,333). It still reads the right hand's holds from the two-handed
  schedule, so its "while holding" labels for that hand are wrong; the locations agree with the scan.
- Legs: unchanged from rebuild 67 (left knee 2.1 s deeper than 3 mm).
- Static check: 11 rows with no balance, including 620-720 with the wrist on the platform; the
  checker models palm holds and has not been adapted.

Not clean: carried in front of the chest, the hand meets the platform's nose on tread 1 and the
tray at the top-out, and the wrist brushes the chest. The box-model clearances reported by
`make_bulb_wristrest_clip.py` (bulb +4, hand +12 mm) missed these.

## Bulb clips tested in the bulb task itself (2026-11-21, overnight)

`physics_segment_test_bulb.py` (a sibling of the physics test): runs in S10 with the right hand's
cupped grip and the bulb placed in it at every staged pose; reports the bulb-against-ladder force,
the hardest squeeze on the bulb by a hand link, the bulb's distance from its seat, and the palm's
tilt. `helpers/bulb_eval.sh` is not in the repo; the commands are scan (`--body_scan 10`) and the
seven `--modes hold --hold_steps 150` segments used for the two-handed clip.

Benchmark baseline: S10's own spawn pose held with zero action for 3 s moves the bulb 110 mm along
the hand and turns it about 33 deg, without dropping it. Up to ~110 mm "off seat" is therefore normal.

### bulb07 in S10
Scan, 148 poses: bulb against the ladder 0 N at every pose; squeeze max 9 N. The open-hand
contacts at steps 360 and 1,320-1,330 are gone with the fingers curled. Still there: right palm
596-1,956 N at steps 520-530. Left knee 403-518 N at 140-150, 550, 970, 1,150 and 2,459 N at 960.

### bulb08: hand rises before moving in over a hold -- no gain
Same as bulb07 plus an up-first waypoint. Scan: right palm 772-2,131 N at 520-530 (unchanged);
new right thumb against right shoulder 658 N at step 350; bulb against the ladder 0 N; squeeze max
11 N. Frozen holds 7 of 7; squeeze at most 6 N; bulb at most 3 N on the ladder; bulb off its seat
29-99 mm, but 521 mm in the tread 3->4 pose: it left the hand. Right wrist_roll 7.0 mm inside the
torso (steps 362-368, 950-958, 1,356-1,366). Peak arm speed 9.3 rad/s.

Cause of the 520-530 contact: nothing to do with the approach. The carried hand sits 140 mm above
the pelvis, which on tread 1 is only about 60 mm above the platform's top, with the hand reaching
over it; when the body dips at the start of the step to tread 2 the palm lands on the platform.

Held carried poses (steps 400, 1,060, 1,100, 1,450) show why the bulb can leave the hand: under the
benchmark's wrist gains the palm sags 13-33 deg fingers-DOWN from a level reference. The bulb
stayed in these four runs (45-132 mm off seat) but the margin is run-to-run.

### bulb09 (three variants): carried higher on tread 1, fingers pitched up -- rejected
`--carry_z_early 0.24..0.30 --carry_pitch 15..18` with carry x 0.16-0.18. The arm cannot hold a
pitched-up palm that high in front of the chest: carry misses of 44-210 mm, joint speeds 10-88
rad/s. S10 scans: hand against the robot's own head, shoulder and torso (thousands of N) at steps
350-490 in two of them; bulb against the ladder up to 355 N and squeezed up to 126 N. The best of
the three (x 0.18, y -0.09, early z 0.24, pitch 15) clears steps 520-530 and keeps the bulb off
the ladder (0 N, squeeze max 38 N) but drives the thumb into the shoulder and the wrist 28 mm
into the torso at steps 350-360. The space between the chest and the ladder on tread 1 is too
small for a carried hand; the next variants keep the wrist resting through that phase instead.
