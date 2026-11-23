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
| 3,900 | 33 | 15 | 2 | 11 |
| 4,200 | 30 | 17 | 12 | 5 |
| 4,500 | 30 | 12 | 0 | 15 |
| 4,800 | 48 | 10 | 0 | 6 |
| 5,100 | 22 | 20 | 15 | 7 |
| 5,400 | 24 | 16 | 7 | 14 |
| 5,700 | 28 | 16 | 4 | 16 |
| 6,000 | 18 | 12 | 1 | 33 |
| 6,300 | 41 | 12 | 0 | 11 |
| 6,600 | 20 | 17 | 4 | 23 |
| 6,900 | 34 | 8 | 1 | 21 |
| 7,200 | 29 | 10 | 7 | 14 |
| final (7,500) | 27 | 12 | 8 | 16 |

The run finished all 7,500 iterations (job 15557924, no physics errors after the restart with the
30 m/s speed cap). Checkpoints swing widely (3 to 48 of 64) and the last ones are not the best:
4,800 is the best at 75%, then 6,300 at 64%; the final checkpoint is 42%. From 6,000 on, the largest
single loss is at steps 850-950, where the handrail hold starts counting.
Earlier note: 3,300 was the best at 55% when first written. Pinned copies of 1,800
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

### bulb10 (three variants): wrist resting from the start to tread 2 -- the hand contacts are gone
`--rests 52 300 330 890 0 0 --carry 0.16 -0.11 0.14`: wrist on tread 4 for steps 52-300, then
straight onto the platform for 330-890; carried only in the top half. Variants: carry pitch 0,
carry pitch 12, platform inset 65 mm.

S10 scan, all three: the only push over 400 N in 148 poses is the inherited left knee at step 960
(2,346-2,459 N). Bulb against the ladder 0 N at every pose; squeeze max 7 N.

The tool's box-model proxy still reports the forearm 23-26 mm inside the platform's edge while on
tread 1 (steps 330-570); the simulator shows no contact over 400 N there. Left to fix: right
wrist_roll 3-7 mm inside the torso (steps 312, 950-958, 1,356-1,366), and 11.9 rad/s with a 59 mm
miss in the 30-step move from tread 4 to the platform.

### bulb11: 1.5 s for the tread 4 -> platform move, fingers pitched up 12 deg when carried -- no better
`--rests 52 275 350 890 0 0 --carry 0.16 -0.11 0.14 --carry_pitch 12`. A hand or wrist is on a
hold for 87.6% of the clip (left 66.7%, right wrist 48.9%).

- S10 scan: bulb against the ladder 0 N at every pose, squeeze max 8 N. New 405 N between the
  torso and the right wrist at step 310, in the move from tread 4 to the platform by way of the
  chest. Left knee 403-717 N at 140-150, 550-560, 970 and 2,252 N at 960.
- S10 frozen holds: 7 of 7 stay up; pelvis off the reference 2-41 mm, ground 187 mm, tread-2 pause
  106 mm, tread 3->4 pose 93 mm. Bulb squeezed at most 6 N. The bulb stays in the hand in six
  poses (42-110 mm off its seat) and leaves it at once (+0.06 s) in the tread 3->4 pose, then
  reads 39 N on the ladder as it falls.
- The longer move did not lower the speed (9.7 rad/s, 58 mm miss) and added the torso contact.
  Right wrist_roll 7.8 mm inside the torso (steps 304-312, 950-958, 1,354-1,389).

Pitching the fingers up is the wrong direction: the cupped fingers close the front of the hand,
the wrist side is open, and the benchmark's own spawn pose already lets the bulb slide 110 mm
toward the wrist in 3 s. bulb10 (level carry) remains the candidate.

### bulb10 adopted as the bulb candidate (2026-11-21)
`bulb10` = the first bulb10 variant (`--rests 52 300 330 890 0 0 --carry 0.16 -0.11 0.14`, carry
pitch 0). A hand or wrist is labelled on a hold for most of the lower half; the right wrist's
labels are tread 4 (228 steps) and platform (540 steps).

S10 frozen holds, nine poses, 3 s: 9 of 9 stay up. Pelvis off the reference: ground 179 mm, mount
2, tread 1 31, tread 1->2 30, tread-2 pause 109, tread 2->3 40, tread 3->4 34, tread 3 34, tread 4
41 mm. Bulb against the ladder at most 4 N; squeezed at most 3 N. The bulb stays in the hand in
seven poses (40-94 mm off its seat) and leaves it in both tread-3 carried poses (474 and 1,659 mm).
A 0.6 s diagnostic at step 1,090 shows no initial ejection: the bulb settles from where the test
places it (29 mm off at the first step) and creeps 20-35 mm per control step; nothing but the
hand's own links touches it. Retention is the benchmark grasp's, not the clip's.

Trainer: the two-handed teacher (iteration 4,800) played on bulb10 untrained, right contact point
set to the wrist (`contact_eef_pos_offset` right = [0.005, -0.03, 0.0]): 0 of 64 past step 400, 58
ended by the hold rule; but where robots are still up the right WRIST satisfies the rule (in range
100%, above its target 100%, upward force ok 90%, 22-27 N). `lm_train_bulb10.sbatch` continues that
teacher from a pinned copy of its iteration-4,800 checkpoint; NOT submitted.

Open: right wrist_roll 7.0 mm inside the torso at steps 950-958 and 1,356-1,366; 11.9 rad/s in the
0.6 s move from tread 4 to the platform; the bulb leaving the hand on tread 3; left-knee contact
inherited from rebuild 67.

## Two-handed run corrupted at about iteration 5,560; link speed cap (2026-11-21)

`lm-edge72` at 11:54 UTC: PhysX asked for 834,139,447 aggregate pairs (then negative numbers), a
minute later `CUDA error, code 2` (out of GPU memory, 15.6 of 16 GB in use) and `Scene state is
corrupted. Simulation cannot continue!`, 5,892 times, while the trainer kept iterating at 3.7 s
per iteration on a dead simulation. Cancelled at 11:58. `checkpoint_5400.pt` was saved at 11:39,
before the first error, and nothing was saved after it.

This is the same event as the stall at iteration 1,196 (1,006,951,763 pairs asked for): one
robot's state blows up and its links cross the whole grid of environments. Doubling the buffers
only changed how it failed. Changes:

- `ladderman_port/g1_fiatlux.py`: `max_linear_velocity` 30 m/s for every link (LadderMan: 1000).
- job scripts: `physx_capacity_scale` 2.0 -> 1.5. Steady demand was 1.07x LadderMan's sizes; at
  2.0 the run sat at 13.5 GB of 16 from the start.

Resubmitted; it resumes from iteration 5,400 with 2,100 to go. What makes a robot blow up is not
found; the reset probe saw several thousand newtons on a hand in about 10% of noisy resets.

## Bulb clip fixes: torso overlap, fast move, bulb retention, forearm (2026-11-21)

### Diagnostics on bulb10 (bulb task)
- Right arm at a 20 N threshold (`--scan_min 20 --scan_only right_`): while the robot stands on
  tread 1 with the wrist "on" the platform (steps 340-560) it is `right_wrist_roll_link`, the
  forearm's end, that is pushed (57-279 N); `right_wrist_yaw_link` reads nothing. On tread 2
  neither is over 20 N: the wrist sits 2 mm above the surface. Step 950: wrist_pitch 359 N
  against the torso. Top half: 27-82 N between wrist_roll and wrist_yaw.
- Bulb retention, 3 s holds at steps 1,060, 1,090, 1,250 with an offset on the right wrist joints
  (`--wrist_offset roll pitch yaw`, rad). Bulb distance from its seat at the end, mm:

  | Offset | 1,060 | 1,090 | 1,250 |
  |---|---|---|---|
  | none | 471 | 5,208 | 55 |
  | roll +0.2 | 26 | 73 | 2,006 |
  | roll -0.2 | 515 | 1,767 | 725 |
  | pitch +0.15 | 15 | 224 | 76 |
  | pitch -0.15 | 103 | 125 | 114 |
  | pitch -0.30 | 56 | 131 | 72 |
  | roll +0.2, pitch -0.15 | 1,720 | 127 | 104 |
  | roll -0.2, pitch -0.15 | 90 | 1,775 | 868 |

  Over about 150 mm the bulb has left the hand. In these arm poses the wrist PITCH joint mostly
  rolls the hand about its finger axis (at -0.15 / -0.30 the hand's +z edge drops 7 / 14 deg at
  steps 1,060 and 1,250). Small samples and erratic; kept as a lead.

### bulb12 (four variants): direct tread 4 -> platform route -- rejected
`--direct --back4 0.13..0.15`: "up and over" misses 71-108 mm, 13.8-17.9 rad/s; one variant
(carry 0.17, -0.14, 0.15) lost the platform rest altogether (221 mm). What the round did show:
carry (0.18, -0.13, 0.14) takes the right wrist_roll from 7.0 mm inside the torso to +12.0 mm,
elbow +37.3 mm.

### bulb13 (three variants): no detour waypoint, carry (0.18, -0.13, 0.14)
The up-first waypoint added in bulb08 was the 58-59 mm miss; without it every path segment is
within 4 mm. Torso: wrist_roll +15.2..15.8 mm, elbow +39.3 mm. Speed 9.0-10.4 rad/s, now only the
wrist yaw during the tread 4 -> platform move (steps 302-316), plus single-step 7.0 rad/s blips
of the elbow at steps 492 and 579. `--carry_roll -10` raises the carry miss from 1.9 to 11.1 mm.

### bulb14: all four fixes together
`--rests 52 270 360 890 0 0 --carry 0.18 -0.13 0.14 --carry_roll -10 --platform_lift 0.014
--platform_lift_until 610`, plus a light final smoothing of the joints (sigma 0.03 s). Labels: left
66.7%, right wrist 28.9% (tread 4 198 steps, platform 230), neither 15.5%.

- Torso: `check_arm_torso.py` CLEAN (right wrist_roll +15.0 mm, elbow +39.5 mm).
- Speed: 7.11 rad/s, 14 frames over 6, all in steps 297-303 (was 11.9).
- S10 scan at a 20 N threshold: over 400 N only the left knee (431-507 N at 140-150, 550, 970-980;
  2,433 N at 960). Right arm over 20 N: nothing during either rest or the carry on treads 1-3;
  palm and fingers 35-149 N at steps 1,310-1,330 (the hand brushing the tray at the top-out).
  Bulb against the ladder 0 N at every pose; squeeze max 6 N.
- S10 frozen holds, ten poses: 10 of 10 stay up. Pelvis off the reference: ground 184 mm, mount 2,
  tread 1 30, tread 1->2 30, tread-2 pause 106, tread 2->3 39, tread 3->4 35, tread 3 29, tread 4
  38, top 120 mm. Bulb off its seat: 38-68 mm in eight poses, 156 mm at tread 3->4, 2,024 mm at
  the top pose (it left the hand). In the tread-3 pose it stayed but read 21 N on the ladder and
  an 18 N squeeze.
- With the wrist lifted 14 mm the forearm no longer presses on the platform's edge on tread 1
  (bulb10: 57-279 N); the tool's proxy still says -9..-11 mm there.

### bulb15: longer move, hand higher at the top-out
bulb14 with `--rests 52 262 372 890 0 0 --carry_z_keys 0:0.14 1230:0.14 1290:0.19`.

- Speed 6.28 rad/s, 8 frames over 6 (steps 297-300). Torso CLEAN (wrist_roll +14.5, elbow +39.5 mm).
- S10 scan, 20 N threshold: over 400 N only the left knee at step 960 (2,354 N). Right arm: only
  139-156 N between wrist_roll and wrist_yaw at steps 1,420-1,430; the tray brush at 1,310-1,330 is
  gone. Bulb against the ladder 0 N; squeeze max 3 N.
- S10 frozen holds: 10 of 10 stay up; pelvis numbers as bulb14. Bulb off its seat 41-85 mm in
  nine poses, including the top pose (73 mm; bulb14 lost it there), and 206 mm at tread 3->4.
  On tread 3 the bulb reads 18-29 N against the ladder and a 16-22 N squeeze during the hold: the
  carried bulb is close to the right stringer there (the proxy has said -9 mm at step 1,096 since
  bulb13) and the sagging robot brings it into contact.

### bulb16: hand carried 25 mm nearer the centre line -- worse; bulb15 stays the candidate
bulb15 with `--carry 0.18 -0.105 0.14 --rests 52 256 380 890 0 0`. The bulb's contact with the
stringer on tread 3 drops to 2-5 N, but: bulb off its seat 1,036 mm (tread 1->2), 473 (tread 3->4),
269 (tread 3) and 2,135 mm (top), i.e. lost in 4 of 10 holds; right wrist_roll 3.3 mm inside the
torso at steps 300-304 with a 309 N push between wrist and torso at step 300; speed 6.46 rad/s.
Holds 10 of 10 stay up; scan otherwise as bulb15.

The speed peak sits at clip steps 295-300 in bulb14, 15 and 16 whatever the move's timing (rest
ending at 270, 262, 256), so it follows the body's motion there, not the length of the move.

### State of the bulb clip
`bulb15` is the candidate (`ladderman_port/koa/lm_train_bulb15.sbatch`, not submitted).

| Issue | bulb10 | bulb15 |
|---|---|---|
| Right wrist_roll vs torso | 7.0 mm inside | +14.5 mm, CLEAN |
| Peak right-arm speed | 11.9 rad/s | 6.28 rad/s (8 frames over 6) |
| Forearm on the platform's edge, tread 1 | 57-279 N | nothing over 20 N |
| Bulb leaves the hand in a 3 s hold | 2 of 9 poses | 1 of 10 |
| Bulb against the ladder: scan / holds | 0 N / up to 4 N | 0 N / up to 29 N on tread 3 |
| Squeeze on the bulb: scan / holds | 7 N / 3 N | 3 N / 22 N |

Not closed: the bulb touching the right stringer in the tread-3 holds (18-29 N, under the 50 N
limit), the 0.28 rad/s over the speed gate, and bulb retention, which varies from run to run and
with small changes of the carry (bulb14 lost it at the top, bulb15 at tread 3->4, bulb16 in four
poses). The corridor between the torso and the stringers on tread 3 is about as wide as the
forearm, hand and bulb are long. The left-knee contacts are rebuild 67's leg motion and untouched.

## Rebuild 72 teacher finished; why robots are lost when the handrail hold starts (2026-11-21)

`play_probe_window.py`, iteration 6,900, steps 800-1,000 on rebuild 72: 26 of 56 surviving robots
are terminated at steps 870-880 by the hold rule. Both hands are in range and above their targets
in 100% from step 850; the force test passes in 0-60% at steps 850-870 (right hand median
(-17, +3) N at 850, i.e. sideways on the rim), and in 100% of the survivors from step 880 at about
(-16, +15) N. The right hand only arrives at about step 840; the label starts at 851 and the rule
allows 30 steps, so a robot has until 881 to turn a sideways touch into an upward push.

## Rebuild 73: handrail hold starts 1.0 s after both hands are on (2026-11-21)

`relabel_handrail.py --clip <rebuild 69> --settle 1.0`: the handrail label starts at step 881
(rebuild 72: 851). The motion arrays are identical to rebuild 72's; anchors unchanged.

Existing teachers replayed on it from the ground, 64 robots, never trained on it:

| Teacher | Full climbs, rebuild 72 | Full climbs, rebuild 73 | Lost at steps 850-950, 72 -> 73 |
|---|---|---|---|
| 4,800 | 48 | 53 | 6 -> 1 |
| 6,900 | 34 | 40 | 21 -> 13 |

Checks (motion unchanged, so these repeat rebuild 67's): hands CLEAN; torso right wrist_roll 3.2 mm
inside at steps 741-743; legs left shin 2.1 s (deepest 26 mm), left thigh 0.4 s (16 mm).

## Rebuild 74: rebuild 73 with the torso overlap removed (2026-11-21)

`nudge_arm_joint.py --clip <rebuild 73> --joint right_shoulder_roll_joint --deg -6 --steps 730 758`
(a new tool: a raised-cosine bump on one joint, that arm's bodies recomputed inside the window
only; it refuses a window where the hand is on a hold). 66 of 3,549 frames change, all in the
right hand's free move from the platform to the handrail; -3 deg also clears it. Shoulder roll
peaks at 2.10 rad/s in the window (was 1.81).

| Check | Result |
|---|---|
| Hands | CLEAN |
| Torso | CLEAN; right wrist_roll closest +1.1 mm (step 1,389), right elbow +7.5 mm |
| Legs | unchanged, FAIL: left shin 2.1 s (26 mm), left thigh 0.4 s (16 mm; steps 542-547, 955-960, 1,132-1,135) |
| Scan, 148 poses | one entry over 400 N: left knee 2,328 N and left thigh 1,011 N at step 960 (as rebuild 67) |
| Frozen, 3 s | 7 of 7 stay up; pelvis off the reference 7-37 mm (ground 187 mm) |
| Static | as rebuild 67: no hold only where one foot has just landed or neither foot counts as planted |

Not passing: the left-leg contact as the left foot lands, which is rebuild 67's leg motion.

## Bulb training started (2026-11-21)

`lm-bulb15`, job 15562958 on ece605 (`sbatch -A ece605 -p ece605 -t 24:00:00 --export=ALL,ITERS=12300
lm_train_bulb15.sbatch`): continues the two-handed iteration-4,800 teacher for 7,500 more
iterations on `bulb15`. About 9-10 s per iteration.

## Bulb against the ladder, measured on the reference (2026-11-21)

`check_bulb_clearance.py` (new): the bulb at its measured seat in the right hand, glass as a 40 mm
sphere and cap as a 21 mm cylinder, against a 5 mm sampling of the ladder's collision triangles.
`bulb15`: never closer than 17.3 mm (glass under the platform's front edge at step 45); 24-33 mm
from the right stringer through the rests, 24.5 mm at steps 1,050-1,150 (tread 3), 28.6 mm from the
tray at the top-out. The 18-29 N read on tread 3 in the frozen holds therefore needs the bulb to
have moved about 25 mm from its seat, by sliding in the hand or with the sagging robot.

## Rebuilds 72, 73, 74 compared on the iteration-4,800 teacher; bulb grip sweep (2026-11-21)

From the ground, 64 robots, seeds 0 / 1 / 2, the teacher trained on rebuild 72 only:

| Clip | Full climbs | Total of 192 | Lost at steps 850-999 |
|---|---|---|---|
| Rebuild 72 | 48, 50, 48 | 146 (76%) | 6, 5, 8 |
| Rebuild 73 | 53, 46, 54 | 153 (80%) | 1, 7, 6 |
| Rebuild 74 | 47, 52, 52 | 151 (79%) | 6, 2, 4 |

The first single-seed comparison (48 -> 53, losses 6 -> 1) overstated it: over three seeds the
longer settle is worth about 3 points untrained, and rebuilds 73 and 74 are indistinguishable.

Bulb grip, `physics_segment_test_bulb.py --pin_root --right_curl C` on `bulb15` (root fixed at
the reference, fingers closed from the benchmark's cup after seating, 10 s holds at steps 400,
750, 1,090, 1,150, 1,250, 1,450):

| Curl | Bulb stays | Hardest squeeze | Furthest off its seat | Bulb on the ladder |
|---|---|---|---|---|
| 0.667 (cup) | 4 of 6; lost at 1,090 (+0.56 s) and 1,150 (+0.92 s) | 1 N | - | 166 N falling |
| 0.80 | 6 of 6 | 7 N | 114 mm | 8 N |
| 0.95 | 6 of 6 | 8 N | 85 mm | 9 N |
| 1.10 | 6 of 6 | 6 N | 43 mm | 5 N |
| 1.25 | 6 of 6 | 9 N | 150 mm | 11 N |

At 1.10 and 1.25 the fingers reach their commanded angles: they cage the bulb, they do not press on it.

Same ten 3 s holds as before with the body free (no balance): cup 9 of 10 stay, squeeze up to 22 N,
ladder 29 N; curl 0.80 9 of 10, 42 N, 58 N; curl 1.10 8 of 10, 49 N, 56 N. The 56-58 N is the
step-1,090 pose: the body sags 30-40 mm toward the right stringer, further than the bulb's 24.5 mm
clearance, and a firmly held bulb is wedged. At step 1,060 the bulb leaves within 0.1 s at every
curl (the hand moves about 40 mm from it in the first control step, no contact force); not explained.
In the carried poses the fingers pitch down 14-31 deg over the hold.

Reading: the cup's retention is the benchmark's and a curl of 0.8-1.25 remedies it under 10 N when
the hand holds its pose; the stringer contact needs the body more than about 25 mm off the
reference. Neither can be closed on frozen poses; the test is a trained bulb teacher in S10.

The pending kill-shared jobs (`lm-stock` x4, `lm-e72fresh` x4) were cancelled on request; none had started.

## Bulb teacher, first checkpoint; two-handed teacher tried in S02 (2026-11-21)

`lm-bulb15` iteration 5,100 (300 iterations on `bulb15`), from the ground, 64 robots, right contact
point on the wrist: 3 full climbs; 34 lost at steps 300-350 (wrist move tread 4 -> platform), 12 in
the mount, 8 at steps 850-950.

Adapter (`record_climb.py --step_ladder --asset_meta ladderman_port/asset_meta_edge72.json`, all
opt-in, the rung-ladder path unchanged). Iteration-4,800 teacher, rebuild 72, S02, layout seed 0,
started in the clip's first pose (`--rsi`, a diagnostic: not a valid S02 start):

| Adapter state | Result |
|---|---|
| wrists from the clip, tread-1 observation point | fell at step 79; `joint_pos_history` 54 sd from training |
| + joint history as LadderMan computes it (`buffer - action offset`: 23 joints RAW, wrists as error against the clip); history seeded after staging | fell at step 323, pelvis +0.22 m; `prev_actions` 7-11 sd |
| + `prev_actions` joint-major, as LadderMan's `action_buf[N, joint, history]` flattens | no term over 4 sd in the first 50 steps; fell at step 200-220 |
| + ladder made kinematic (`--pin_ladder`, diagnostic) | on tread 1 from step 150 to 550, root 20-90 mm off the reference; fell at step 553 (the tread 1 -> 2 step) |

With the ladder free it moved 17 mm at step 120, as the robot mounts, and the robot was down by
step 200. `--zscore N` prints each observation term's largest |z| against the checkpoint's own
statistics. The checkpoint's joint-history means confirm the reading for the wrists (mean within
0.06 rad of zero, sd 0.03-0.10, where the clip's wrist angles average up to 1.3 rad). The same two
layout points (raw joints, joint-major previous actions) apply to the rung-ladder teachers by the
same code, but are left as they were for them: untested there.

Not yet matched to the trainer: the benchmark robot is 38.6 kg against the trainer's 34.4, has
articulated fingers, and stands on the real ladder mesh, free-standing.

## S02 with the Dex3 hand: first successes by the task's own gate (2026-11-21)

Correction to the entry above: those first S02 trials ran the benchmark's DEFAULT robot, which is
the Inspire hand (the contact trace showed `R_thumb_intermediate`). The teacher is trained on the
open Dex3. `record_climb.py` now takes `--robot dex3`; with `--step_ladder` the finger targets are 0.
Inspire, ladder pinned, 10 varied starts: 0 of 10, none past tread 2.

Trainer-side twin of `--trace` (`ladderman_port/play_probe_trace.py`): through the mount the
trainer shows the same forward lean (8-9 deg), the same root offset (+50-70 mm forward) and the
same shoulder-roll offsets (-20 / +15 deg) as S02, so those are the teacher's own behaviour.

Iteration-4,800 teacher, rebuild 72, S02, layout seed 0, Dex3 (34.40 kg, fingers open), started in
the clip's first pose with +-0.05 rad joint noise (`--rsi --rsi_noise 0.05 --stop_at_clip_end
--episodes 10`), which is a diagnostic start, not S02's:

| Ladder | Success (task gate) | Lost on the ground | Lost in the mount (step ~215) | Lost at tread 1 -> 2 (step ~605) | Lost higher |
|---|---|---|---|---|---|
| pinned (`--pin_ladder`, diagnostic) | 4 of 10, at step 1,342-1,343 | 2 (`fell_below`, steps 57, 81) | 0 | 4 | 0 |
| free-standing (the benchmark's) | 0 of 10 | 1 | 6 | 0 | 3 (steps 1,043, 1,046, 1,286; pelvis to 1.71 m) |

Trainer, same teacher: 76% from the ground. New diagnostics in `record_climb.py`: `--trace N`
(root error, tilt, joints, contact forces by limb), `--mass_to`, `--rsi_noise`, `--stop_at_clip_end`.

## S02 episode counts are not yet trustworthy across episodes in one process (2026-11-21)

Dex3, ladder pinned, started in the clip's first pose, iteration-4,800 teacher:

| Run | Result |
|---|---|
| fresh process, no start noise, stopped at step 640 | passes the tread 1 -> 2 step, tracking the trainer's trace within 10-20 mm and a few degrees |
| 10 episodes in one process, +-0.05 rad, seed 0 | 4 success; 4 fall at steps 602-612; 2 down at steps 57, 81 |
| 20 episodes in one process, +-0.05 rad, seed 1 | 2 success (episodes 17, 19); 13 fall at steps 554-619; 5 elsewhere |
| 4 episodes in one process, NO noise | episode 1 success (step 1,341); episodes 2-4 fall at steps 620, 603, 621 |

The last row is the finding: identical starts, and only the first episode climbs. Either the
multi-episode reset in `record_climb.py` leaves something behind, or the simulation's own
non-repeatability decides a knife-edge step; the first-episode pattern points at the reset. Until
that is settled the multi-episode counts above understate the teacher, and only fresh-process runs
are reported as results. Ruled out for the pinned-ladder gap: robot mass (34.40 vs 34.39 kg),
finger pose (open), observation layout (no term over 4 sd at the start), friction (the trainer
randomises 0.3-7.2).

Benchmark ladder: a free rigid body of 10.9 kg (`LADDER_MASS_KG`); the trainer's is fixed boxes.

## S02, fresh process per episode: first climbs of the FREE-STANDING ladder (2026-11-21)

Dex3, iteration-4,800 teacher, rebuild 72, layout seed 0, started in the clip's first pose with
+-0.03 rad joint noise, seeds 1-8, one process per episode (`--episodes 1`):

| Ladder | Success (task gate) | Lost in the mount (steps 207-231) | Lost at tread 1 -> 2 (606, 619) | Lost higher |
|---|---|---|---|---|
| pinned (diagnostic) | 4 of 8 (steps 1,341-1,343) | 0 | 2 | 2 (steps 1,059, 1,288) |
| free-standing (the benchmark's) | 2 of 8 (steps 1,343, 1,348) | 4 | 0 | 2 (steps 1,280, 1,282) |

These replace the multi-episode counts above. The tread 1 -> 2 fall does occur in fresh runs, less
often than in one-process batches. Still a diagnostic start: S02's own spawn pose is not used.

From S02's OWN spawn pose (no staging), Dex3, free-standing ladder, three fresh runs: all down within
61-64 steps. The spawn stands the robot 0.199-0.229 m behind the clip's start (and 17-58 mm to the
side, 40 mm higher), varying run to run. Nothing bridges the spawn pose to the clip's first frame yet.

## Bulb teacher (`lm-bulb15`), checkpoints from the ground, 64 robots (2026-11-21)

| Iteration (on the bulb clip) | Full climbs | Lost at steps 0-199 | Lost at steps 300-349 (wrist move tread 4 -> platform) | Lost at steps 850-999 |
|---|---|---|---|---|
| 5,100 (300) | 3 | 9 | 34 | 8 |
| 5,400 (600) | 7 | 9 | 25 | 12 |
| 5,700 (900) | 10 | 14 | 23 | 10 |
| 6,000 (1,200) | 12 | 29 | 5 | 9 |
| 6,300 (1,500) | 12 | 21 | 10 | 9 |
| 6,900 (2,100) | 11 | 30 | 5 | 11 |
| 7,500 (2,700) | 20 | 25 | 3 | 12 |
| 8,100 (3,300) | 21 | 32 | 5 | 4 |
| 8,700 (3,900) | 20 | 21 | 9 | 4 |

Free-standing ladder, four other layouts (layout seeds 1-4: climb yaw -88, -180, +37, -124 deg),
seeds 1-3 each, fresh processes, clip start pose: 1 success of 12 (layout 2); 7 lost in the mount
(steps 209-225); 4 lost higher (steps 669, 1,043, 1,329, 1,351). The adapter follows the ladder's
pose and yaw. All free-standing fresh runs so far: 3 of 20; 11 of 20 lost in the mount.

## Rebuild 75: a walk-in from S02's spawn pose (2026-11-21) -- built, NOT trained

S02's spawn, measured (`record_climb.py --dump_spawn`, layout seeds 0 and 2, seeds 1 and 2, Dex3):
root at clip x -1.036..-1.063, y +-0.016, z 0.790, yaw about -1 deg; legs in the default pose
(hip -3, knee 11-13, ankle -8 deg), arms at zero, each joint +-1 deg. `ladderman_port/spawn_s02_nominal.json`.

`make_approach_prefix.py --clip <rebuild 74> --spawn spawn_s02_nominal.json --over 1.15 --ahead 0.0`
(new): drops the clip's first 20 steps, in which the feet slide on the floor out of LadderMan's
default pose, and puts a walk-in in their place: settle 0.8 s (crouch to pelvis 0.765 m, arms to the
clip's pose), then left foot half-way, right foot all the way, left foot the rest (0.6 s weight
shift + 0.7 s swing each, 40 mm lift), home 0.6 s. Legs by damped least squares on `g1_arm_fk.fk_all`.
A single 0.22 m step is out of reach for the trailing leg at that height.

- 265 walk-in steps + 1,459 of the clip = 1,724 steps; every old step number is +245.
- Tool gates: foot error 0.00 mm, no sliding while planted, joint limits pass, 4.15 rad/s peak,
  joins the clip's frame exactly.
- Static (`static_support_check.py`, friction 0.5): every walk-in window holds on the feet alone,
  single-foot phases at 37-44% ankle effort. With the pelvis only 85% of the way over the stance
  foot the three single-foot phases had no static solution; 1.3 is out of leg reach.
- Torso: CLEAN (as rebuild 74).
- Frozen holds in S02: uninformative on the floor, as before (a pose with no balance controller
  leans within a second; the spawn pose itself drifts 291 mm in 3 s).
- Trainer: loads and trains for two iterations locally from the iteration-4,800 checkpoint with
  `task.max_episode_length=1800`. `ladderman_port/koa/lm_train_edge75.sbatch` written, NOT submitted.

## What the free-standing ladder does during the mount (2026-11-21)

`record_climb.py --trace 1`, Dex3, seed 1, clip start pose. The ladder is still (0.0 mm) until the
left hand lands on tread 4 at step 37 with a two-step impact (84 N down, 46 N forward on the
ladder). Over the next 17 steps it tips steadily to 2.27 deg about its cross axis, its origin
18-19 mm up, while the hands read only 18-25 N down and about 10 N forward; it then rocks by 1-2 deg
through the mount, and the robot is down by step 215. Ladder in the simulator: 10.90 kg, centre of
mass at (0, -0.012, 0.826) in its own frame, i.e. midway between front and rear feet. Those hand
forces cannot hold a 107 N ladder tipped about either pair of feet (about 58 N m is needed), so the
cause of the sustained tilt is NOT explained. A contact sensor on the ladder's root prim reads 12 N
at rest and nothing from the robot, so it does not cover the ladder's colliders and settles nothing.

## Start randomisation for the walk-in (2026-11-21)

The benchmark's reset (`SubtaskEventCfg`): root x and y +-0.05 m and yaw +-0.1 rad about the spawn
pose, every joint +-0.05 rad, zero velocity. LadderMan's own start: a random phase of the clip,
root x/y +-0.02 m, z +-0.01, roll/pitch/yaw +-0.1 rad, joints +-0.1 rad. So yaw and joints are
already covered; root x/y is not, and only 265 / 1,724 = 15% of episodes start in the walk-in at all.

`ladderman_port/command_spawn.py` (deployed to LadderMan as
`active_adaptation/envs/mdp/commands/hdmi/command_spawn.py`): `RobotTrackingSpawn`, a subclass that
forces `start_zero_prob` of training episodes to step 0 and gives starts inside `floor_steps` the
root range `floor_pose_range`. Mid-climb starts keep LadderMan's range (5 cm off a 68 mm tread is
not a benchmark start). `lm_train_edge75.sbatch` uses 0.3, 265 and x/y +-0.05 m. Two-iteration
local smoke test passes; NOT submitted.

## A real S02 success on the free-standing ladder, saved (2026-11-21)

`record_climb.py --video` (new: fixed camera, one frame per step). Dex3, free-standing ladder,
layout seed 0, seed 16, fresh process, clip start pose: `success` at step 1,342. It was the 8th of
8 attempts in that batch (seeds 2, 8, 11-16): 2 fell in the mount, 5 near the top (steps 1,277-1,359).
Video, bag and a note saying it is not a valid S02 start were copied to the user's Desktop.
Free-standing fresh runs so far: 4 of 28.

## Free-standing ladder in the trainer, first version (2026-11-21)

`+task.command.ladder_config.free=true` (`ladderman_port/ladder_fiatlux.py`, two hooks in
`ladder_env.py`): the collection's boxes become collision-free markers; one dynamic body
`{ENV_REGEX_NS}/LadderBody` carries the same boxes as colliders (stringers and rear legs stopped
45 mm short of the floor, on four flat feet), 10.9 kg, centre of mass (-0.2116, 0, 0.826), inertia
(3.57, 3.927, 1.318), friction 1.5 / 1.2, solver settings as the benchmark's; after every physics
step each marker is written to the body's pose composed with its own offset; the body is put back
at every reset.

Iteration-4,800 teacher from the ground on rebuild 74, 64 robots, 400 steps (`play_probe_trace.py`):
54 still up at step 400, about what the fixed ladder gives. The body moves a median 1.5-2.3 mm and
tilts 0.00 deg (max 50 mm in single environments once robots are on it; one environment's ladder
was 189 mm off in the first 20 steps, not explained).

That is NOT how the benchmark's ladder behaves: there it tips 2.3 deg when a hand lands. So mass,
centre of mass, inertia and friction do not account for the rocking; what is left is how it
stands on the floor. The benchmark asset's colliders are three SDF meshes, two of them named
WheelsFront and WheelsBack. Next: use that asset itself as the trainer's free body.

## S02's success gate fires before the feet are on the platform (2026-11-21)

`climbed_to_ladder_top`: pelvis above `platform top (1.18) + TOP_STANCE_PELVIS_OFFSET (0.787) -
LADDER_TOP_HEIGHT_SLACK (0.15)` = 1.817 m, within 0.6 m of the top point, under 1.5 m/s. The
reference crosses 1.817 m at step 1,346; in S02 the gate fired at steps 1,341-1,348 in every
success. At step 1,342 of the reference the left ankle is on the platform (1.188 m) and the right
ankle is at 1.028 m, in the air between tread 4 and the platform; both feet are on the platform
from about step 1,380. `success` is a termination, so the episode (and a recorded demo) ends there.
`record_climb.py --past_success` (diagnostic) disables the termination, reports the step at which
the real gate is true, and lets the run continue.

## Rebuild 76: rebuild 75 plus a tail that ends in the next stage's start pose (2026-11-21) -- NOT trained

S03's start, measured (`record_climb.py --task FIATLUX-S03-RemoveOldBulb-v0 --dump_spawn`, layout
seed 0, Dex3): root at clip (-0.082, 0.013, 1.961), yaw -2.6 deg, legs in the default pose, arms at
zero; the on-ladder tier resets with +-0.02 m, +-0.05 rad, joints +-0.02 rad. The clip ended at
pelvis (-0.170, 0.037, 1.942) with the arms slightly forward: 9 cm short, off centre.

`make_topout_suffix.py --clip <rebuild 75> --target spawn_s03_L0_measured.json --over 1.0 --ahead 0.0
--walk_z 1.915` (new, the walk-in tool's machinery): shift, left foot 108 mm, shift, right foot
86 mm, home 0.8 s with the arms lowered. The target is lowered 20 mm so the feet stay on the
platform's real surface (S03 stages against a 1.18 m top; the collider's is 1.152).

- 1,724 + 180 = 1,904 steps. Foot error 0.00 mm, no sliding, joint limits pass, 2.95 rad/s peak.
- With the legs straight (the clip's final height) the legs cannot reach; at `--over 1.15` the
  left ankle roll passes its limit. Stepping at pelvis 1.915 m with `--over 1.0` passes both.
- Static: both single-foot phases hold on the feet alone (50% and 42% ankle effort).
- Torso: CLEAN. `ladderman_port/koa/lm_train_edge76.sbatch` (episode limit 2,000), NOT submitted.

## Benchmark gate change committed; what live S02 runs show at the top (2026-11-21)

Commit `12d91be` (benchmark code, not climb tooling): the climb gate for S02 and S10 now needs both
feet up at the platform and still, every corner of both soles over the platform's footprint, held
0.5 s; and every subtask's episode runs 2.0 s past its success (`mdp.sustained(linger_seconds=)`,
`SUCCESS_LINGER_SECONDS`). Checks are in the commit message. Issues filed for two things the checks
found that predate it: #269 (S08's gate tests the bulb against the default bench position) and
#270 (`verify_move_ladder.py`'s hands_free check).

Live, Dex3, free-standing ladder, clip start pose, 17 fresh runs on the final gate, iteration-4,800
teacher: 0 pass. 11 fell in the mount, 3 on the way up, 3 reached the top:

| Seed | At the top | Heels against the platform's open edge (from the bag) | Then |
|---|---|---|---|
| 24 | pelvis 2.00 m | left 100-110 mm over (40-60 mm inside after step 1,483), right 130-190 mm over | fell at step 1,563 |
| 25 | pelvis 1.93 m | left 104-124 mm over, right 170-196 mm over | fell at step 1,517 |
| 29 | pelvis 1.94 m | no closed episode in the bag | stood to the 1,700-step limit; gate never met |

`gate_soles_within_ladder_top` is false on every row of seeds 24 and 25. The reference itself ends
with the heels 42-44 mm inside; the robot stands 10-19 cm behind it. The clip ends at step 1,479 and
the teacher has never been trained to stand after that. For the next rebuild: end the clip with a
held stance of about 3 s, and find out whether the lag at the top exists in the trainer too.
On-ladder start stances of S03, S04, S11, S12 (layout seeds 0 and 3): all soles inside the
footprint, heels 126-130 mm in, toes 31-35 mm in, sides 30-43 mm.

## Why the robot ends on the platform's edge: the reference slides its feet (2026-11-21)

Trainer, iteration-4,800 teacher on rebuild 72, medians over 49 robots still up (`play_probe_trace.py`):
the root is within 10 mm of the reference at step 1,380 and then falls behind steadily, -148 mm
forward and -94 mm sideways at step 1,470. So the lag seen in S02 is the teacher's, not the
benchmark's.

The reference, steps 1,370-1,478: both ankles stay at 1.188 m (planted on the platform) while they
travel from x -0.343 / -0.352 to -0.202 / -0.200 and the left one from y +0.039 to +0.156. The
last 2 s of the clip SLIDE both planted feet 15 cm along the platform (a blend into the base
clip's final pose, like the slide in its first 20 steps). A robot cannot follow that, so it stays
where the feet landed: ankles about 5 cm OUTSIDE the platform's open edge (edge at x -0.304), only
the front 7 cm of each sole on it.

- Rebuild 77 = rebuild 76 + `--hold 3.0` (3 s standing still at the end): 2,054 steps. Built; it
  still contains the slide, so it is superseded.
- `make_topout_suffix.py --cut_at 1627 --steps left:0.5 right:1.0 left:1.0` (cut the clip before
  the slide, step to the S03 stance from where the feet really land): NOT feasible as tried. With
  the pelvis 9 cm ahead of the stance ankle the legs cannot reach; at 7 cm the right leg cannot
  reach and the single-foot phases have no static solution -- the stance foot has 7 cm of sole on
  the platform. The landing itself has to change: the top-out step of the base builder must put
  the feet fully on the platform.

## Repairing the top-out landing; the left ankle is past its limit on every left step (2026-11-21)

`fix_topout_landing.py` (new): for each foot's last swing, from the moment the sole is clear of the
platform the foot travels to the clip's FINAL foot pose in the air, touches down there and stays;
only that leg is re-solved, the pelvis, waist and arms are the clip's. On rebuild 74: left foot
340 mm in steps 1,310-1,325 (peak 2.0 m/s), down by 1,333; right foot 279 mm in steps 1,356-1,371
(1.7 m/s), down by 1,379; foot error 0.00 mm; over the last 1.5 s the ankles move 0.00 mm (were 161
and 132 mm). Landing nearer the open edge than the final stance is not reachable with the clip's
straight-legged ending. NOT yet checked: static support with the new landing, legs against the
ladder, and that the hands are still on the handrail while a foot is in the air (the clip releases
them at about step 1,359, with the right foot up until 1,375).

Found by that tool's joint-limit gate, and NOT caused by it: in rebuild 67's legs the LEFT ankle
pitch is outside its -50 deg limit on every left step up -- steps 99-112 (-56.1 deg), 537-552
(-54.1), 950-965 (-54.1), 1,127-1,142 (-55.3), 1,306-1,319 (-50.6). The builder's limit gate reads
the arms only. Those are the windows of the left-shin contact every rebuild since 67 has failed on
(542-547, 955-960, 1,132-1,135), and the first is in the mount.

`clamp_leg_limits.py` (new): eases an offending leg joint onto `limit - margin` and recomputes that
leg's bodies. On rebuild 74 (`--allow_planted`: 13 of the 174 frames are at toe-off, foot nearly
still): ankle pitch now -49.0 deg at worst, the ankle link moves 2.2 mm at most, but the TOE comes
down by up to 15 mm in the swing (step 104: 28 mm over tread 1's top becomes 14 mm). So clamping
alone trades an impossible ankle for a thinner toe clearance; the swing needs lifting to match.

## Rebuilds 80 and 81; the benchmark's ladder asset in the trainer; S03's start stance is inside the ladder (2026-11-21)

Chain, each step a tool in this folder:
`fix_topout_landing.py --land_shift -0.04 0.0 --settle 0.16 --end_at 1400` (left foot lands with its
heel just inside the platform's edge, right foot at the clip's final spot; the clip's straight-legged
ending is dropped) -> `clamp_leg_limits.py --allow_planted --keep_toe --margin 3.0` (left ankle
pitch inside its limit, swing lifted up to 19.4 mm, toe within 4.0 mm of where it was) ->
`make_approach_prefix.py` (walk-in) -> `make_topout_suffix.py --hold 3.0` (tail and a 3 s stand).

- With the landing at the clip's final spot for BOTH feet the one-foot phase is 64 mm off static
  balance (the foot is ahead of the body); with the left foot 4 cm nearer the edge it is 26 mm, the
  original was 20 mm. From both feet down it balances at 34% effort.
- Rebuild 80 (tail to S03's start stance, 1,976 steps): all 29 joints inside their limits, soles
  over the platform for the last 7.3 s -- but the S02 whole-body scan reads 6,000-21,000 N on the
  hips, knees and pelvis from step 1,740 to the end. REJECTED.
- Why: S03's own start stance is inside the ladder. `FIATLUX-S03-RemoveOldBulb-v0`, Dex3, layout
  seed 0, zero action: the ladder presses on left_hip_roll_link with 16,500-18,600 N and on
  left_hip_pitch_link with 5,000-13,600 N from the first step to the sixtieth, and the pelvis is
  pushed up from 1.969 to 1.986 m.
- Rebuild 81 (tail to the climb's own final stance, `ladderman_port/stance_platform_clear.json`:
  pelvis (-0.170, 0.037, 1.942), arms lowered; one 40 mm step of the left foot; 3 s stand; 1,906
  steps): all 29 joints inside limits; soles over the platform for the last 5.9 s, heels 42-44 mm
  inside, toes 141-143 mm, sides 16 and 91 mm. Full suite running.

Benchmark ladder asset as the trainer's free body (`+task.command.ladder_config.free=asset`,
`assets_fiatlux/AlumStep_D/` = the `_collision_rigid` USD and its two layers, spawned as the
benchmark spawns it): loads, 10.90 kg, 3 collision shapes, markers in place (object 0 at x -0.588).
Iteration-4,800 teacher from the ground, 64 robots: 44 up at step 400; the body moves a median
0.3 mm and tilts a median 0.00 deg (max 0.8 deg while hands land; one environment 15 deg at step
180). So the asset does not reproduce the 2.3 deg tip of the single traced S02 run either. Not yet
compared: the floor, the solver settings (the benchmark runs with stabilisation on, the trainer
off), and how typical that one S02 run was.

## Rebuilds 82-84; the pelvis shift for the shin contact (2026-11-21)

Scans are now every 5 steps (`--body_scan 5`); every 10 stepped over a contact.

| Step (new numbering = old + 245) | Rebuild 74, the clip the teacher learned | Rebuild 82 | Rebuild 84 |
|---|---|---|---|
| 790 (left onto tread 2) | under 400 N | left thigh 689 | 689 |
| 1,205 (left onto tread 3) | shin 2,328, thigh 1,011 | shin 3,315 | 3,315 |
| 1,380 (left onto tread 4) | thigh 1,843, shin 1,449 | thigh 1,904, shin 1,476 | the same |
| 1,565 (left onto the platform) | nothing | left_hip_roll against pelvis_contour 6,345 | nothing |

- Rebuild 81: suite found three things of its own -- left thigh and shin 12 kN at step 1,380 (the
  ankle repair's 19 mm lift under the platform's lip), the two hands 3.7 mm apart at step 27 of the
  walk-in, right thigh 10 mm inside the ladder in the tail (steps 1,672-1,722).
- Rebuild 82 = 81 with `clamp_leg_limits.py --skip_lift 1115 1155`, `make_approach_prefix.py
  --arm_stagger 0.35`, `make_topout_suffix.py --over 0.8`. All three gone; frozen 8 of 8; torso
  clean; right leg clean; hands: only the inherited 0.5 mm finger clearance on the handrail (step
  1,169). But the every-5 scan shows the landing repair's own defect at step 1,565: carried
  forward at the clip's swing height the left knee reaches 109 deg and the thigh meets the pelvis.
- Rebuild 84 = 82 with `fix_topout_landing.py --low_travel` (the foot crosses the platform 3 cm
  above it): knee 95 deg at most, the 1,565 entry gone. Remaining scan entries are the left-leg
  contacts rebuild 67 has always had. Rest of the suite running.

`shift_pelvis_windows.py` (new: pelvis moved in chosen windows, both ankles and both wrists held
at their world poses by re-solving all four limbs). 2 cm back in the four left-landing windows on
rebuild 82: limbs held to 0.00 mm, no joint outside limits, left arm changes up to 20 deg. Left
shin inside a tread 26 -> 13 mm and 1.9 -> 0.8 s; left thigh 19 -> 9 mm. Static balance in the
one-foot moments after each left landing: 51 -> 63 mm behind the foot (mount), 27 -> 40, 22 -> 37.
NOT adopted: half the contact for a quarter more imbalance where most runs are lost.

On-ladder start stances, zero action, ladder force on the robot: S03 with Dex3, left_hip_roll_link
16.5-18.6 kN from the first step and the pelvis pushed up 2 cm; S03 with Inspire, S04 / S11 / S12
with Dex3 and S12 with Inspire start clean, a hip resting on the handrail at 16-113 N. So the
stance is not inside the ladder in general (an earlier entry said so); something specific to S03
with Dex3 is. Not filed.

## Toe raise on the trailing foot; rebuilds 85-87 (2026-11-21)

`shift_pelvis_windows.py --dz 0.04 --heel_raise right 20 --ramp 0.3` over the four left landings
(windows 366-384, 797-807, 1,209-1,219, 1,392-1,400): the right foot goes up on its toes about the
front of its sole (ankle +44 mm), the pelvis rises 40 mm, all four limbs re-solved (held to
0.01 mm, no joint outside limits). Nothing moves backward.

| | Rebuild 84 | With the toe raise |
|---|---|---|
| Left knee at landing: mount, treads 2, 3, 4 | 94, 94, 98, 90 deg | 85, 84, 88, 79 deg |
| Left shin inside a tread: mount, 2, 3, 4 | 26, 23, 22, 21 mm | 21, 4, none, 4 mm |
| Time more than 3 mm inside | 1.9 s | 0.6 s |
| Left thigh, worst | 19 mm | 14 mm |
| Scan (overlap indicator; every 5 steps) | steps 790, 1,205, 1,380 | step 1,380 only |
| One-foot moment after each left landing, static | 51 / 27 / 22 mm behind the foot | 50 / 26 / 22 mm |
| 0.3 s before each left landing (right foot on its toes, 30 mm of sole) | balances with the hands | does not, on 2 of 4 treads |

The raise cannot start at touchdown: the right foot lifts 0.1-0.2 s after the left one lands.
A stronger raise on the mount (60 mm, 28 deg; rebuild 86) left the mount's 21 mm unchanged: that
contact is after the right foot has left the floor.

Rebuild 87 = rebuild 84's body + tail `--over 0.95` (the 4 cm step balances at 47% ankle effort;
0.8 and 0.9 do not) + the toe raise at 40 mm / 20 deg on all four landings. Suite: all 29 joints
inside limits; frozen 8 of 8; torso clean; scan one entry (step 1,380, left thigh 1,922 and shin
1,592); hands only the inherited 0.5 mm finger clearance (step 1,169); left shin 21 / 4 / 5 / 4 mm
(0.6 s), left thigh up to 14 mm; right thigh up to 6 mm in the tail (steps 1,675-1,720, the price
of the 0.95 shift; 10 mm at 1.0); static as rebuild 84 plus the pre-landing toe moments.
Scan numbers are what the simulator reports one step after a pose is staged with an overlap:
overlap indicators, not loads. In real S02 runs the left shin read 39-119 N.

## Why the benchmark's ladder rocks: its stabilization threshold (2026-11-21)

From 29 recorded S02 runs (free-standing ladder, iteration-4,800 teacher), ladder tilt from the
bags: 0.1 deg before the hands touch it; 2.5 deg median (1.7-3.3) as they land, in every run;
about 1.5 deg through the mount in runs that survive it, about 2.7 (to 4.8) in runs that fall there.
Trainer, same ladder asset, same teacher: 0.0 deg median, 0.8 at most.

`ladder_push_test.py` (benchmark, robot pinned) and `ladderman_port/play_probe_push.py` (trainer):

| | Benchmark | Trainer, same asset |
|---|---|---|
| 84 N down + 46 N forward for 2 steps, then 20 + 10 N held | 0.00 deg | 0.00 deg |
| 300 N forward for 1 step at tread 4 | 5.3 deg, peak 0.45 s later, 1 s to settle | 1.0 deg, peak 0.1 s later |
| 500 N forward for 1 step | 10.0 deg | 3.2 deg |

Rigid-body estimate with the feet where they are (0.54 m from the centre of mass): 1.1 deg, 0.1 s.
(First version of the test left the robot free: with no policy it falls onto the ladder at step 32
and tips it 5.9 deg. That was not the push.)

Slow ramp, 0 to 80 N sideways at tread-4 height, robot pinned, layout seeds 0 and 2: the benchmark
ladder starts to tip at 10-12 N in either direction (a support line 92-106 mm from the centre of
mass). Expected from its weight and foot spacing: 57 N. One setting changed at a time:

| Change | Tips at |
|---|---|
| as shipped | 12 N |
| ladder `stabilization_threshold` 0.001 -> 0.0 | 59 N |
| ladder `sleep_threshold` (already 0.0) | 12 N |
| `sim.physx.enable_stabilization` False | 59 N |
| min iterations 1/0, bounce threshold 0.5 (with stabilization off) | 59 N |
| ladder spawned 10 mm higher; max depenetration 10 m/s | 12 N |

At rest for 2 s the ladder is as quiet with the threshold at 0 (4.3 mm/s, 0.004 rad/s, 0.05 mm
drift) as with 0.001 (4.2 mm/s, 0.0075 rad/s, 0.04 mm). The teacher's hands push forward with
10-13 N on tread 4. `record_climb.py --ladder_stab_threshold 0.0` (diagnostic) applies the change
to a run without touching the benchmark; twelve fresh S02 runs with it are in progress.

## With the ladder's stabilization threshold at 0, no S02 run is lost in the mount (2026-11-21)

Dex3, free-standing ladder, iteration-4,800 teacher (rebuild 72), clip start pose, +-0.03 rad,
fresh process per run, `--ladder_stab_threshold 0.0`, seeds 1-8 and 11-14, final gate (12d91be):

| | As shipped (20 fresh runs earlier) | Threshold 0 (12 runs) |
|---|---|---|
| Lost in the mount (steps 200-250) | 11 | 0 |
| Ladder tilt as the hands land, from the bags | 2.5 deg median (1.7-3.3) | 0.00 deg median, 0.02 max |
| Ladder tilt through the mount | 1.5-2.7 deg | 0.00 deg median, 0.04 max |
| Lost at steps 1,279-1,312 (pelvis 1.71 m: the step onto the platform) | 6 of 20 | 6 |
| Reached the top (pelvis 1.90-1.93 m) | 3 | 6: three stood to the 1,700-step limit, three fell at steps 1,423, 1,553, 1,557 |
| Passed the gate | 3 (the older, looser gate) | 0: the teacher stands with its heels over the edge, as before |

Every run got at least to tread 4. What is left is the top-out, which is the old clip's ending.

## The ladder after an operator lets go of it: recorded S01 takes re-simulated (2026-11-21)

`ladder_release_replay.py` (new): for each of the 13 recorded Dex3 VR takes of S01 (10 scored 1.00),
the ladder's own state (pose, both velocities) at the last step a hand gripped it is written to a
ladder standing alone, robot pinned, and left to settle for 4 s. "Quiet" = first 0.5 s under 10 mm/s.

Method check, the four takes whose last release is the end of the carry: the shipped-setting
replay's peak tilt against the bag's own next steps -- 2.46 / 2.41, 1.39 / 1.52, 1.25 / 1.28,
1.20 / 1.22 deg.

| Ladder `stabilization_threshold` | Quiet after (12 takes; one take is the ladder already lying down) |
|---|---|
| 0.001 (shipped) | 2.1-3.2 s in seven takes; not within 4 s in five |
| 0.0 | 0.04-1.8 s in all twelve |

Peak tilt after release is the same or lower with 0.0 (e.g. 3.05 -> 1.56, 2.36 -> 0.50, 0.47 -> 0.16
deg); none topples with either. In the recorded takes themselves the ladder is still tilted
1.0-2.2 deg and moving at 80-390 mm/s in the last two seconds before success in nine of ten.

## 2026-11-21 — ladder stabilization fix merged to main and brought here

- The ladder fix (`LADDER_STABILIZATION_THRESHOLD = 0.0` on both ladder spawns) merged to main as PR #271
  and is cherry-picked onto this branch as `ea5af32`. `record_climb.py --ladder_stab_threshold 0.0` is
  no longer needed; the free-standing ladder now has this value by default.
- Issue #121's evidence take (VR S01, seed 65) was re-simulated from its tilted states (5.85, 9.92,
  10.25 degrees). The frozen tilt in that take was NOT reproduced, with the ladder's sleep threshold at
  0.0 or at the pre-fix 0.005, at the recorded floor position or the layout's own. The stabilization
  threshold only slows the return: 1.2 s to drop from 9.92 degrees at 0.001 against 0.4 s at 0.0.
- `ladder_release_replay.py` gained `--release_step` (bags without `grip_force`), `--keep_xy` and
  `--sleep_threshold`.

## Rebuild 87: local checks and training script (2026-11-21) -- NOT submitted

- Benchmark gate table in the S02 scene (`verify_climb_gate.py --old <rebuild 74> --new <rebuild 87>`):
  12 of 12; the clip's last frame meets every conjunct and the termination fires at hold 25 + linger
  100 steps.
- Iteration-4,800 teacher on rebuild 87 in S02 (free ladder with the stabilization fix, Dex3, clip's
  first pose): aligned to 0.000 m, fell at step 102. In the trainer, same clip, free asset ladder, 64
  robots from step 0: 47 up at step 100, 10 at 200, 4 at 300. It never saw the 265 walk-in steps.
  Not measured: the same teacher started after the walk-in.
- Trainer smoke run, 64 environments: rebuild 87 + `ladder_config.free=asset` + `RobotTrackingSpawn`
  (`start_zero_prob=0.3`, `floor_steps=265`) + episode limit 2,100 loads, iterates and writes a
  checkpoint. 4,096 environments not run.
- `ladderman_port/koa/lm_train_edge87.sbatch`: continues lm-edge72 iteration 4,800; `ITERS` is the
  total (default 12,300 = 7,500 more); `FAMILY` separates a test run's folder and resume lineage from
  the real one; refuses to start if the free-ladder trainer files are not installed; 120 G.
- On Koa: clip, ladder asset, `command_spawn.py` and the script are uploaded (checksums match). The
  two trainer files that differ (`ladder_env.py`, `ladder_fiatlux.py`) are STAGED beside the trainer,
  not installed, because the running bulb job would pick them up if it were requeued.

## Bulb rebuild 17; the trainer on the free ladder at full scale; training started (2026-11-22)

### Bulb clip `bulb17` (and `bulb18`, rejected)
- `bulb15`, the clip `lm-bulb15` learned, is rebuild 74 with only the right arm's seven joints changed:
  legs, pelvis, waist and left arm identical to 0.0 deg. So rebuild 87's chain applies unchanged.
- Start and end poses measured in the benchmark (`record_climb.py --dump_spawn`, layout seeds 0 and 2):
  S10 spawns on S02's floor spot (root x -1.036 / -1.060) with the right arm in the carry pose
  (shoulder_pitch -19.4 / -20.5, elbow +20.3 / +19.3, wrist_roll +89.8 / +90.0 deg); S11's start stance
  is root x -0.110 / -0.086, z 1.961, same joints. `ladderman_port/spawn_s10_nominal.json`,
  `stance_s11_nominal.json`, `stance_platform_clear_bulb.json` (rebuild 87's end stance + the carry arm).
- Chain on `bulb15`: `fix_topout_landing.py --land_shift -0.04 0.0 --settle 0.16 --end_at 1400
  --low_travel` -> `clamp_leg_limits.py --allow_planted --keep_toe --margin 3.0 --skip_lift 1115 1155`
  -> `make_approach_prefix.py --spawn spawn_s10_nominal.json --over 1.15 --ahead 0.0 --arm_stagger 0.35`
  -> `make_topout_suffix.py --target stance_platform_clear_bulb.json --steps left:1.0 --over 0.95
  --ahead 0.0 --walk_z 1.915 --hold 3.0` -> `shift_pelvis_windows.py --dx 0.0 --dz 0.04 --heel_raise
  right 20 --ramp 0.3 --avoid_limits 3 --window 366 384 --window 797 807 --window 1209 1219 --window
  1392 1400` = `bulb17`, 1,906 steps.
- `shift_pelvis_windows.py --avoid_limits DEG` (new, off by default): the 4 cm pelvis rise drove the bulb
  arm's wrist_roll to 122.7 deg (limit 113); a null-space term now uses the 7-joint arm's spare degree of
  freedom to stay inside, the wrist pose still held to 0.00 mm / 0.00 deg. Wrist_roll peaks at 111.9.
  Without the flag the tool reproduces rebuild 87 byte for byte.
- `bulb17` against rebuild 87: every joint that is not the right arm identical, pelvis identical.
- Suite: joint limits pass; frozen holds all stay up; torso clean (bulb forearm +14.5 mm at step 547);
  bulb never touches the ladder (closest 17.3 mm, as `bulb15`); squeeze 3 N at most; legs as rebuild 87
  (shin 21 / 4 / 5 / 4 mm, thigh up to 14 mm, right thigh 6 mm in the tail); scan (overlap indicator,
  every 5 steps): left knee steps 395 and 1,380, left hip_yaw 1,380. Right-arm scan at 20 N: nothing
  (`bulb15`: wrist_roll and wrist_yaw up to 156 N at steps 965, 1,405-1,435).
- With the benchmark's cup (`--fingers spawn`) the bulb leaves the hand in the tread 3->4 and top holds,
  as it did on earlier bulb clips.
- NOT valid for the bulb hand: `check_hand_collision.py` (assumes an open gripping hand; 11 "contacts")
  and `static_support_check.py` on the top-out rows (assumes the right hand holds the handrail).
- `bulb18` = the same with the tail ending in S11's measured stance (`--steps left:1.0 right:1.0`,
  1,976 steps): left knee up to 35 mm inside the ladder at steps 1,737-1,802, and the end stance itself
  flags both hip_roll links and the right ankle for the whole stand. Rejected; S11's own stance rests a
  hip on the handrail.
- Videos (kinematic): four angles under the videos folder, `bulb17_ref/`. The ladder stands in a corner
  in S10 layout 0: no front or left view.

### Trainer at 4,096 environments with the free ladder (Koa, ece605)
- Three runs in one job hung twice in start-up, every process in Lustre I/O wait (`cl_sync_io_wait`).
  First blamed on the shared simulator caches (runs then started one at a time: the third still hung).
  Cause: the node ran out of memory -- 2 GB of 251 available. One run takes about 80 GB of the node, of
  which ~50 GB shows as the process's own (MaxRSS 52-54 GB); the rest is shared memory (20 GB) and memory
  the job accounting does not see. Two runs fit (37 GB available with both training).
- The free ladder overflows PhysX's collision stack at the default 64 MiB ("collisionStackSize buffer
  overflow ... at least 100 MB ... Contacts have been dropped", 4,909 lines in 30 iterations); the fixed
  ladder does not. `ladder_env.py` takes `+task.sim.physx_collision_stack=<bytes>`; free-ladder runs set
  256 MiB. Zero overflow lines in the two-run test.
- Trainer's free ladder: `sleep_threshold` was unset (PhysX default) and `stabilization_threshold` 0.001;
  both now 0.0, as the benchmark after PR #271.
- Speed: 13.5-14.7 s per iteration with the free ladder (fixed: 7-10 s). 7,500 iterations is about 29 h,
  past the 24 h limit: one resubmission resumes each run from its newest checkpoint.
- `koa/lm_train_edge87.sbatch` (one run; CLIPNAME, LADDER, SEED, EEF_R, SEEDCKPT, FAMILY from the
  environment) and `koa/lm_train_edge87_multi.sbatch` (several runs as steps of one job, one GPU each).
- SUBMITTED: job 15569375, two runs, 7,500 iterations each: `edge87free` (rebuild 87, free ladder, from
  lm-edge72 iteration 4,800) and `bulb17free` (`bulb17`, free ladder, from lm-bulb15 iteration 11,400).
  Not run: rebuild 87 on the fixed ladder, and rebuild 87 without the toe raise (clip `edge87n`).
- `lm-bulb15` (job 15562958) cancelled at iteration 6,769 of 7,500. From the ground, 64 robots, fixed
  ladder: 34 full climbs at checkpoint 9,600, 44 at 10,500, 45 at 11,400 (pinned in `teacher/keep/`).

### `record_climb.py`: a start-pose bug, and the bulb in S10
- BUG, fixed: with `--rsi_noise` the staged joint velocity was (next reference pose - NOISY pose) * 50,
  i.e. every joint was handed -noise * 50, up to 1.5 rad/s at 0.03. Every earlier `--rsi_noise` run
  (the free-ladder S02 counts included) started with that kick. Velocity is now the clip's own.
- `--right_curl C` (new): right hand staged in the benchmark's cup, the bulb seated on the hand as it
  actually is after staging, the curl ramped to C over 15 steps; reports the bulb's distance from its seat
  and the hardest squeeze. Before this the recorder opened both hands flat on the step ladder.
- First attempt (bulb seated on the REFERENCE wrist, hand pre-curled): thrown out in 5 steps or squeezed
  past 100 N in 3. Staging error, not a result.
- `lm-bulb15` checkpoint 11,400 in S10, free ladder, Dex3, clip's first pose, seeds 1-3, curls 0.667 /
  1.0 / 1.25: the bulb leaves the hand at step 9-11 in all nine, squeeze 3-6 N. Same without start noise.
  Curl makes no difference.
- Cause: the teacher's own first move. Step 1: right elbow 11 rad/s, its target 39 deg past the clip,
  right shoulder_roll 9.4 rad/s; the wrist moves at 0.9-1.0 m/s where the clip has 0.02. The trainer
  shows the same teacher doing the same from step 0 (64 robots, median: right elbow +44 deg off the clip
  by step 3, left elbow +37, torso pitched 11 deg). Not the adapter. `bulb15`'s first second is the
  stretch already known to slide the feet.
- So whether the benchmark's grip holds the bulb through the climb is still unmeasured: the bulb never
  survives the start on this teacher. First check on `bulb17free` checkpoints: the first ten steps.

## Descent rebuild 1: `desc01` (S12, hands free) and `descbulb01` (S04, bulb) (2026-11-22) -- NOT trained

A descent is a climb played backwards (`reverse_clip.py`, new: poses reversed, velocities reversed and
negated, hold labels reversed), with its own start and its own ending.

### What the benchmark fixes
- Start (`record_climb.py --dump_spawn`, layout seeds 0 and 2): S12 and S04 both start in the stance S11
  starts in, root x -0.110 / -0.086 in the clip's frame, legs at the default pose; S12 arms at zero, S04
  right arm in the carry pose. Left to settle with zero action the robot comes to rest with its ankles
  at 1.181 m (root 1.933, pitched 2.2 deg forward): there is no lip under the feet; the clip uses 1.188.
  `ladderman_port/stance_s12_nominal.json`.
- End: `descended_from_ladder` wants the pelvis under 0.94 m, within 0.6 m of the ladder's ROOT, slower
  than 1.5 m/s. The climb's mount stance is 0.614 m out and S02's spawn spot 0.85 m. So the walk-in is
  NOT reversed: the clip keeps the climb's steps 265 onward, and after stepping down takes two short
  steps TOWARD the ladder to a stance with the pelvis 0.56 m out (`stance_floor_descent.json`).

### Leaving the platform's front
The start stance is 7.6 cm further forward than where rebuild 87 ends, where the ladder's top frame
narrows. A full sideways weight shift over the forward foot puts the upper thigh (hip_yaw link, y 0.205)
into the rail. Forward tail from rebuild 84's body to the S12 stance, `make_topout_suffix.py --steps
left:1.0 right:1.0`, leg overlap in the tail (`check_leg_collision.py --stride 3`):

| --over / --ahead / --walk_z | left thigh | left knee |
|---|---|---|
| 0.95 / 0.0 / 1.915 (as `bulb18`) | 35 mm | 9 mm |
| 0.95 / 0.0 / 1.930 | 35 | 8 |
| 0.95 / -0.03 / 1.915 | 36 | 7 |
| 0.95 / -0.05 / 1.920 | 36 | 8 |
| right foot first, 0.95 / -0.03 / 1.920 | 32 (right 34) | 34; right knee past its limit |
| 0.7 / -0.04 / 1.925 | 10 | 0 |
| 0.6 / -0.04 / 1.925; 0.6 / -0.02 / 1.930; 0.5 / -0.04 / 1.925 | 0 | 0 |

It is the SIDEWAYS shift, not the crouch or the fore-aft position. Used: `--over 0.6 --ahead -0.02
--walk_z 1.930 --phases 0 0.6 0.4 0.8` (swing shortened to 0.4 s: at 0.6 the body's centre is outside
the stance sole, so the step is a dynamic one). The same forward tail would let a CLIMB end in the next
stage's exact stance without the overlap that rejected `bulb18`; not built.

### Chain
Forward: `<rebuild 84's body, walk-in on>` (`edge84c`; bulb: `bulb_rebuild/b3_walkin`)
-> `make_topout_suffix.py --target stance_s12_nominal.json` (bulb: `stance_s11_nominal.json`) with the
step parameters above and `--hold 0.5` -> `shift_pelvis_windows.py` as rebuild 87 (bulb: `--avoid_limits
3`) = a climb of 1,811 steps ending in the descent's start stance.
Then `reverse_clip.py --keep 265 1811` (1,546 steps) -> `make_topout_suffix.py --target
stance_floor_descent.json` (bulb: `stance_floor_descent_bulb.json`) `--steps left:1.0 right:1.0 --over
0.95 --ahead 0.0 --walk_z 0.765 --hold 3.0 --arms_first 1.2 --arm_stagger 0.35` = 1,876 steps.
`make_topout_suffix.py --arms_first / --arm_stagger` are new and off by default; without them the tool
reproduces rebuild 87's tail byte for byte. `descbulb01` against `desc01`: every joint that is not the
right arm identical.

### The ending took four tries
1. Arms lowered in the last segment: the steps toward the ladder carried the reaching hands into tread 4
   (fingers 0.1 mm off, hands 3.7 mm apart). -> arms first, one after the other.
2. Spawn arm pose (forearms forward) at 0.56 m: fingertips AT tread 4's front edge. The end pose is free
   (S12 is terminal; S05 starts from its own spawn) -> arms hang at the sides.
3. Hanging straight down: each hand inside the hip (simulator scan: thumb and fingers against the hip,
   kN-scale overlap indicators). `check_hand_collision.py` does not test hand against hip and had passed
   it. -> shoulder_roll 0.4 outward.
4. Bulb arm: in the benchmark's carry pose the glass is 27.8 mm inside tread 4 at this distance. Retracted
   with the forearm kept level (shoulder_pitch +0.3, elbow -0.3): 93 mm from the ladder.

### Suite, final clips
- Joint limits pass. Torso clean; right wrist_roll +1.1 mm at step 177 (hands-free), which is rebuild
  87's own +1.1 mm at its step 1,633.
- Legs: as rebuild 87 in the reversed body (left shin 21 mm at the dismount, steps 1,407-1,420; left
  thigh up to 14 mm); the start steps and the ending are clean.
- Scan (overlap indicator, every 5 steps): 365 of 376 poses clean. Flagged: the start stance itself, steps
  0-35 and 60-65 (ankles 1.2-1.5 kN, hip_roll links 0.6-0.8 kN) -- the benchmark's own stance, whose hips
  rest on the handrail (16-113 N measured in live S04/S11/S12); and step 430 (left thigh and knee), which
  is rebuild 87's step 1,380.
- Frozen holds, the poses rebuild 87 was tested at plus the start and the first step off the front: 9 of
  9 stay up (both clips).
- Hands (hands-free): 6 flags, all 0.8-2.3 mm clear, none inside, at steps 219, 581 and 1,412 where a hand
  is arriving at or leaving a hold. The same poses are in rebuild 87; its run sampled other frames. Not
  confirmed at full sampling.
- Bulb: never touches the ladder (closest 17.3 mm offline, 0 N in the scan); squeeze 3 N at most. With the
  benchmark's cup the bulb leaves the hand in two holds (top, and rebuild 87's step 1,305), as on every
  bulb clip.
- Static balance: row for row rebuild 87's (same poses; the check does not know the direction of time).
  The end stance balances at 14-16% effort. This says nothing about the dynamics of stepping DOWN.

### A test that does not apply
`physics_segment_test.py --modes hold` on a pose standing free on the floor: the robot is a PD statue
with no balance control and topples in about a second. The end stance "fell" at 0.5 s with the hands in
the hips, then "stayed up" by toppling onto the ladder (pelvis 198 mm off the reference, tilt 15 deg,
bulb squeezed to 49-56 N, bulb thrown out). Neither says anything about the stance. Floor poses are
judged by the static check and the scan.

### The gate (`verify_descent_gate.py`, new; S12 and S04, layout seed 0)
- Last frame: `descended_from_ladder` true, pelvis 0.560 m from the root (limit 0.6), height 0.789 m;
  robot_standing and ladder_near_vertical true.
- `descended_from_ladder` is FIRST true at clip step 1,410 of 1,876, while the robot is stepping off tread
  1 (pelvis under 0.94 m and 0.41-0.6 m out); false again at the mount stance (step 1,546, 0.614 m); true
  from about step 1,700. The gate has no hold and no feet-on-the-floor condition, so with the 2 s linger an
  episode would end around step 1,510 with the success latched mid-step.
- S12's `bulb_seated` read true on the staged poses; this does not exercise the locked bulb of #143.
  S04's `payload_held` read false only because the bulb is not staged.

Videos (kinematic): `desc01_ref/` side and rear, `descbulb01_ref/` rear-right, under the videos folder.

## First checkpoints of the free-ladder runs in the benchmark; bulb rebuilds 19 and 20; the trainer's body (2026-11-22)

### Job 15569375 (`edge87free`, `bulb17free`), checked early and stopped at 10 h 55 min
- Trainer, 64 robots from the clip's start (`play_probe_trace.py`), reaching step 1,800: `edge87free` 27 at
  iteration 5,400, 60 at 6,900; `bulb17free` 5 at 12,000, 33 at 13,200. The bulb arm did not improve with
  it: right elbow 24 deg off the clip by step 3 at both checkpoints, 44-83 deg off at steps 1,200-1,650.
- Benchmark S02 (Dex3, free ladder, clip's first pose, noise 0.03), `edge87free`:

  | checkpoint | seed 0 | seed 1 | seed 2 | seed 3 |
  |---|---|---|---|---|
  | 6,900 | platform, stands | falls 450 | platform, stands | falls 446 |
  | 6,900 with `--trace` | platform | falls 446 | falls 446 | falls 446 |
  | 7,200 | falls 520 | falls 1,477 | falls 1,494 | - |

  One seed changes outcome with `--trace` on; one flip does not separate a side effect of the trace from
  run-to-run variation. The runs that reach the platform do NOT score: `soles_within_ladder_top` stays
  false, both heels over the open edge (left 60-68 mm, right 142-143 mm, toes 17-82 mm inside).
- Stopped on the user's word: the two runs train a body the benchmark does not have (below). Newest
  checkpoints pinned on the cluster: `edge87free` 7,500, `bulb17free` 14,100.
- Earlier in the day a wait loop of mine matched its own command line (`pgrep -f`) and the first probes
  never ran for five hours. Wait on an output marker or a PID, not on a pattern the waiter contains.

### `record_climb.py`: three opt-in readouts and a soft start
- `--gate_table N`: every N steps, each conjunct of the task's success gate on the live state, and the
  margin of each of the eight sole corners inside the platform footprint the gate uses.
- `--residual_ramp N --ramp_joints REGEX`: the teacher's correction on the reference faded in over the
  first N steps for the matching joints. Not for the legs: their correction carries the gravity
  compensation, and ramping it let the robot sag 28 mm and jolt the hand.
- The squeeze readout now takes the harder of the two hand sensors, as the crush gate does.

### Why the bulb is lost in S10, in the order the causes were removed
All S10, Dex3, free ladder, clip's first pose, `--right_curl` cup, one run per row unless said.
1. The teacher's first action. Step 1: right shoulder_roll target 30 deg from where it is, waist_pitch
   5.5 rad/s, wrist 1.2 m/s (the clip: 0.02). Bulb out at step 5 (26 with start noise), squeeze under 3 N.
   A 100-step soft start on waist and both arms removes it.
2. `bulb17`'s own opening. Steps 20-45 move the bulb hand from the spawn carry pose to the clip's: 9 cm,
   0.55 m/s, palm tipped 14 deg on the way. Played with the root pinned and no teacher
   (`physics_segment_test_bulb.py --pin_root --modes track`, cup fingers): the bulb ends 136 mm off its
   seat. With a soft start on, the bulb slides 65 mm toward the palm by step 30 and jams between thumb and
   finger at step 35 (39 N, then `bulb_crushed`), at every curl from 0.667 to 1.0.
3. The cup at the benchmark's curl 0.667 does not grip: squeeze 0-1 N. At 0.9-1.0 it reads 2-3 N and the
   bulb stays 4-34 mm from its seat through the walk-in (three runs of four; curl 0.8 lost its seat on a
   second seed).
4. The landing of the bulb wrist on tread 4's near edge (the clip's wrist rest). Clip: 0.83 m/s, arriving
   step 300. Teacher: 1.8 m/s, arriving 285-289, elbow 21-24 deg off. Hand force on the last step 88 N up
   (`fresh_bulb_struck`: the bulb's velocity changed by over 2.4 m/s in a step) or 26 N back
   (`bulb_crushed`).
5. Leaning on that wrist during the first step up (steps 453-477 of `bulb20`): 21-70 N backward on the
   right hand; ends `bulb_crushed`, `bulb_dropped` or both rules on one step.
Nine firm-grip runs on `bulb19`/`bulb20`: crushed 3, struck 2, both 1, dropped 1, fell 2.

### Bulb rebuild 19 (`bulb19`) and 20 (`bulb20`) -- `bulb20` in training
- `make_approach_prefix.py --right_arm_span S --level_palm` (new, off by default; without them the tool
  reproduces `bulb17`'s walk-in byte for byte): the right arm takes S seconds from the spawn pose to the
  clip's, and the three wrist joints are solved so the hand's orientation goes straight between the two
  (both level). `bulb19` = `bulb17`'s chain with `--right_arm_span 5.0 --level_palm`. First 300 steps: hand
  speed relative to the pelvis 0.10 m/s (0.55), palm off level 1.2 deg (14.1). Only the right arm, only
  steps 0-250, differs from `bulb17`.
- Pinned-root playback of steps 0-300: bulb 12 mm off its seat at the end, 27 at most (`bulb17`: 136).
- `dilate_window.py` (new): one stretch of a clip in slow motion, whole body, positions by cubic through
  the original frames, velocities the new clip's own derivatives, the added time a whole number of steps
  and frames. `bulb20` = `bulb19` with steps 262-302 at one third speed: 1,976 steps (+70 after step 302),
  right wrist peak in the window 0.39 m/s (1.03). Before step 258 and after the window it equals `bulb19`
  to 0.0 (shifted 168 frames). Limits pass, torso CLEAN (+14.5 mm), bulb never under 17.3 mm from the
  ladder. Physics scan and frozen holds NOT rerun (every pose is one of `bulb17`'s).
- `retime_arm_window.py` (new) stretches ONE arm's move instead. Rejected for this reach: the reach is
  coordinated with the torso leaning in, and the arm running ahead put the bulb 23.1 mm inside the ladder
  at step 276. Kept for moves that are not coordinated with the body.
- Teacher (`bulb17free` 13,200) on `bulb20`, soft start, curl 0.9-1.0, two seeds each: all four pass the
  tread-4 landing that ended every `bulb19` run and end at steps 463-518, in the first step up.

### The trainer's body was not the benchmark's
- Self-collision is on in both. Shapes, benchmark read from the live S02 stage (Dex3), trainer from its
  URDF, link frame, compared by extent:

  | link | trainer | benchmark |
  |---|---|---|
  | shoulder_pitch (L/R) | none | cylinder r 30, 50 long |
  | shoulder_roll (L/R) | none | cylinder r 30, 30 long |
  | shoulder_yaw (L/R) | capsule r 35, 200 long | hull 81 x 75 x 145 |
  | hip_pitch (L/R) | none | hull 83 x 99 x 117 |
  | pelvis | sphere r 70 | none; pelvis_contour_link hull 120 x 136 x 124 |
  | logo_link | none | hull 149 x 149 x 29 |

  (mm.) Hip roll / yaw, knee, torso mesh, head, elbow and the three wrist links agree to under 3 mm. The
  trainer's torso also carries three neck cylinders inside the head's hull. In the benchmark only a
  joint's own two links are excluded from colliding, so torso against shoulder_roll collides.
- Measured in S02: shoulder_roll links 349-435 N on the torso at step 300, left shoulder_roll motor at 100 %
  torque, the joints about 20 deg inside the clip -- in the run that climbs and in the ones that fall.
- `ladderman_port/make_fxbody_urdf.py` (new) + `fx_body_collision_points.npz`: a sibling URDF,
  `g1_29dof_dex3open-complexfeet-fxbody`, with those links' shapes replaced by the benchmark's (convex
  hulls of the benchmark colliders' points; cylinders as hulls because the import turns URDF cylinders
  into capsules). Inertials, joints, names unchanged; checkpoints load.
- The same teachers on the two bodies, trainer, 64 robots from the clip's start, stock rewards, robots
  reaching the clip's end: `edge87free` 6,900: 56 -> 48 (15 lose a hand hold, median step 436);
  `bulb17free` 13,200 on `bulb20`: 29 -> 4 (42 lose a hand hold, median step 508, all by 631). The
  benchmark's failures sit in the same windows (S02 falls 446-450, S10 ends 463-518).

### Opt-in trainer terms (`ladderman_port/bulb_hand.py`, registered by one import line)
- LadderMan's arm tracking is loose: 0.1 m / 0.1 rad free, fall-off 1.0 m / 0.5 rad, averaged over six
  links or eight joints. A 24 deg elbow error costs about 8 % of one term.
- Rewards: `tight_arm_joint_pos_tracking`, `tight_arm_pos_tracking`, `tight_arm_ori_tracking` (both arms);
  `carry_hand_linvel_tracking`, `carry_hand_contact_force_l2` (over 15 N), `carry_hand_accel_l2` (over
  5 m/s2). Terminations: `carry_hand_force_exceeded` (50 N), `carry_hand_velocity_jump` (2.4 m/s), both
  exempting an episode's first 25 steps. They go into existing reward groups: a new group changes the
  critic's size and a checkpoint no longer loads. There is NO bulb in the trainer; these read the wrist.
- `play_probe_terms.py` (new): which termination ends each robot and when, each reward's running mean, the
  hand's contact force and velocity change per step, and (`+event_force=N`) where the hand is when loaded.
- `bulb17free` 13,200 on `bulb20`, old body, terms on, no grace: all 64 end by step 500, 52 on the force
  rule (median step 327), 5 on the velocity jump. The earliest (from step 80) are the bulb hand against the
  robot's own right thigh (200-355 N) and then the left hand: the teacher carries the hand 14 cm below and
  7 cm behind the clip. The offline suite does not test hand against thigh or hand against hand.
- `edge87free` 6,900, hands, old body: 23-33 N median on the ladder; spikes 99th percentile 195 N on tread
  1 -> 2, worst 682 N; over 15 N in 11 % of robot-steps before the hands should touch anything.
- Without the grace, at 4,096 environments, the mean episode with the terminations on was 4.7 steps (30
  without them): random-phase starts with the wrist already on a tread. With 25 steps of grace: 30.5.

### Scale test and the job now running
- Job 15574012, two runs, 4,096 environments, benchmark body, 30 iterations: 12.6 and 10.5 s per iteration
  (13.1-13.2 on the old body), MaxRSS 58.6 / 58.9 GB (52-54), zero overflow, exit 0.
- Job 15574462 (`koa/lm_train_fxbody_multi.sbatch`, new; the edge87 scripts are untouched): `edge87fx`
  from `edge87free` 7,500 with the tight arm terms; `bulb20fx` from `bulb17free` 14,100 on `bulb20` with
  the tight arm and carry-hand terms, velocity-jump termination on, force termination OFF (penalty only).
  7,500 new iterations each.
- Seen in every start's first log line, old runs included: `actor_std/mean` 0.50, where `edge87free` had
  annealed to 0.14 by iteration 7,150. The action noise restarts on each resume. Cause not yet looked at.

Videos: benchmark runs of the early checkpoints and of `bulb19` under the videos folder, `bench_early/`.

## Bulb rebuild 21 (`bulb21`): no tread-4 wrist rest, more room from the right stringer (2026-11-22) -- in training

### A test that plays the clip itself with the bulb loose: `physics_segment_test_bulb.py --follow_root`
- The robot is a puppet of the clip: its gravity off, the ladder kinematic, and at every PHYSICS step the root
  pose and velocity and the 29 joints' positions and velocities written from the clip (interpolated inside a
  control step). The bulb and the fingers stay dynamic. Hand off the clip: median 1-2 mm.
- `--pin_root` is only right while the body barely moves: on climbing windows it read the palm 65 deg off
  level where the clip has 28 at most, and "lost" the bulb in windows the puppet holds.
- Checked on the two openings with known answers (curl 0.667): `bulb17` loses the bulb at +1.08 s, `bulb19`
  keeps it (76 mm off its seat at most).
- A segment that starts while the hand is moving (0.5 m/s at step 1,300) drops a resting bulb into a moving
  hand and can lose it within 7-21 steps. Start segments at slow moments, or run long ones: `bulb21` steps
  1,200-1,975 in one run holds the bulb throughout.
- Contact forces are overstated: any touch pinches the bulb against an immovable hand. Counts and places are
  what to read.

### The grip, measured with it
- `bulb20`, whole clip in 150-step segments, segments losing the bulb by `--right_curl`: 0.667 7 of 14,
  0.75 2, 0.8 3, 0.85 2, 0.9 0. Every curl under 0.9 loses it in steps 300-450. At 1.0 this tool reads a
  222 N squeeze where `record_climb.py --right_curl 1.0` read 2-3 N: the two map "curl" to finger targets
  differently. NOT reconciled; 0.9 here is the tested value.
- At curl 0.9 the bulb's root sits at (0.112, 0.076, -0.128) in the right wrist frame, 22 mm further out of
  the palm than the loose-cup seat `check_bulb_clearance.py` assumes (0.1192, 0.0540, -0.1310). With the
  measured seat `bulb20`'s glass is -2.1 mm from the platform's front lip at step 343 (approach to the
  tread-4 rest), +8..+10 mm just before and after the rest, +20..+25 mm during it, and +21..+28 mm from the
  right stringer over steps 650-1,250.
- Tilt and acceleration alone were a false alarm: `bulb20` has the palm up to 28 deg off level and the hand
  at up to 8.3 m/s2, and holds the bulb through both at curl 0.9.

### Why the tread-4 rest goes
- Platform underside 1.109 m, tread 4's top 0.944 m. At the rest the glass tops out at 1.072 (loose cup) or
  1.094 m (curl 0.9): 15 mm under the platform. On the approach (step 340) it is at 1.115 m, into the lip.
  Moving the wrist along the tread keeps the bulb under the platform; in front of it the wrist is 10 cm
  short of the tread.
- Rejected, all by the puppet map at curl 0.9:
  - `clear_bulb_path.py` on `bulb20` (wrist pushed away from the nearest ladder point): 180 touching steps,
    452 N, two segments lost. It only sees the glass; away from the lip is down, into tread 4.
  - Variant A, `--pitch -6 8 8 --lift 0.02` (hand tipped fingers-down at the rest, low approach): bulb lost
    in one segment, 96-99 N touches, 226 N squeeze.

### `bulb21`
- `make_bulb_wristrest_clip.py --rests 0 0 372 890 0 0` (no tread-4 rest; the rest of `bulb15`'s arguments),
  `bulb15`'s left-hand labels copied in (30 rows, steps 851-880, that a fresh build labels free), then
  `bulb20`'s chain unchanged, then `clear_bulb_path.py --target 0.045 --max_shift 0.05`: wrist moved up to
  24 mm over steps 660-975, 13 mm at 1,205-1,230, 14 mm at 1,391-1,458, orientation held to 0.011 deg.
  Chain check: started from `bulb15` it reproduces `bulb20` to 0.0, and `edge74` with `bulb15`'s arguments
  reproduces `bulb15`'s motion to 0.0.
- `make_approach_prefix.py --level_palm` now holds the three wrist joints 0.5 deg inside their limits (the
  no-rest variant had right wrist_roll at 113.58 deg for steps 149-193, limit 113).
- Against `bulb20`: 1,976 steps; every joint that is not the right arm identical; right arm differs in steps
  2-1,484; right-hand rest steps 230 (420); left labels identical.
- Offline: limits pass; torso CLEAN; glass never under 25.4 mm from the ladder at the measured seat, 28.6 mm
  at the loose-cup seat.
- Puppet maps, curl 0.9, three per clip with the re-seat points shifted by 50 steps:

  | clip | segments | bulb lost | steps touching the ladder | hardest squeeze while held |
  |---|---|---|---|---|
  | `bulb20` | 40 | 1 (step 344, the lip) | 78 (340-451, 674-881; up to 29 N) | 31 N |
  | `bulb21` | 39 | 2, both the re-seat artefact above | 3 (1,042-1,043, 1,154; up to 6 N) | 11 N |

- Frozen holds, cup at curl 0.9, nine ladder-supported poses, 3 s each: 9 of 9 stay up on both clips. Pelvis
  off the reference at the mount pose 44 mm (`bulb20`: 2 mm) -- what the wrist rest was carrying -- and at
  the first-step-up pose 55 (58), on tread 1 46 (85); the six higher poses equal. Bulb in the hand in all
  nine on `bulb21` (furthest 114 mm); `bulb20` loses it at tread 3->4 (196 mm) and at the top (183 mm).
- Body scan (overlap indicators, not loads; every 5 steps): bulb against the ladder 0 N at every pose,
  squeeze 3 N; left knee at steps 465 and 1,450, left hip_yaw at 1,270 and 1,450. `bulb17` had the knee at
  the same two poses (395, 1,380 before the +70) and hip_yaw at one; the legs are `bulb20`'s to 0.0, and
  `bulb20` was not scanned, so the extra hip_yaw pose is unexplained.
- NOT tested: the dynamics of the first step up without the right wrist. A frozen hold says the pose is
  holdable, not that the teacher can make the step.

### Descent clips, checked against today's findings -- not rebuilt
- Benchmark body: none of `edge87`, `bulb20`, `desc01`, `descbulb01` enters the shoulder, hip or thigh
  colliders (shoulder links 5-11 mm clear of the torso hull, wrist 60-132 mm from thigh and hip).
- `descbulb01`, puppet map: curl 0.9 loses the bulb in 0 of 13 segments (one touch, steps 1,509-1,519,
  16 N: the reversed tread-4 rest); curl 0.667 in 4 of 13. It needs `bulb21`'s arm (reversed), and its
  opening is the mirror of the move `bulb19` slowed (0.55 m/s in the first 2 s).
- For training a descent: the climb scripts' start randomisation (x, y +-0.05 m over the first 265 steps)
  is for the floor walk-in; a descent starts on the platform.

### Koa
- Job 15574462 (ece605) unchanged: `edge87fx`, `bulb20fx`. Action noise restarts at 0.50 on a resume and is
  back to 0.13-0.18 within about 500 iterations.
- `bulb21fx`: job 15575977 on `kill-shared` (preemptible, requeues from its own checkpoints), one run, from
  `bulb20fx` 14,400, `lm_train_fxbody.sbatch` with `ALLOW_PARALLEL=1` (new: skips the stand-down that keeps
  two jobs off one lineage). Submitted BEFORE the holds and the scan above were run, on the user's word.
  - A value with commas cannot go through `sbatch --export=`: export it in the calling shell.
  - V100 nodes: `CUDA error: no kernel image is available` with this trainer's PyTorch. Excluded, with the
    ece605 node (two runs fill its memory), the sliced-GPU nodes and the 2080 Ti / 2070 nodes; the RTX 5000
    nodes showed 59-77 GB actually free against the ~80 GB a run needs.

## Descent rebuild 2: `descbulb02` (S04, bulb) on `bulb21`'s arm; `desc01` (S12) kept (2026-11-22) -- NOT trained

- Chain as descent rebuild 1 with the forward clip replaced by the no-tread-4-rest bulb climb at its walk-in
  stage (variant B, `bulb15`'s left labels). Check: from rebuild 1's forward clip the chain reproduces
  `descbulb01` to 0.0. Then `clear_bulb_path.py --target 0.045 --max_shift 0.05` (wrist up to 14 mm at steps
  422-489 and 650-676, 24 mm at 906-1,220), `dilate_window.py --steps 610 680 --factor 2.5` (the approach to
  the platform rest) and `--steps 2 62 --factor 3` (the opening). 2,086 steps; the trainer's episode limit
  is 2,100.
- Against `descbulb01`: every joint that is not the right arm identical before the two slowed stretches.
  One right-hand rest, the platform, steps 882-1,116 (the reversed tread-4 rest is gone). Bulb hand in the
  first 3 s 0.18 m/s (0.55); in the second before the rest 0.64 m/s (0.87) -- that window holds a foot step,
  so slowing the whole body buys less there.
- Offline: limits pass; torso CLEAN; glass never under 25.4 mm from the ladder at the curl-0.9 seat (28.6 at
  the loose-cup seat).
- Puppet (`--follow_root`, curl 0.9): 0 of 27 segments lose the bulb over two maps; one continuous run of
  all 2,086 steps holds it (71 mm off its seat at most, squeeze 11 N, no touch). In the segmented maps the
  bulb touches the ladder in 19 steps, all at 883-1,018, up to 22 N: the landing on the platform rest and
  the first seconds of it. `bulb21` (climb), which settles onto the same rest slowly, has 3.
- Frozen holds, cup at curl 0.9, nine poses: 9 of 9 stay up. The bulb leaves the hand in three (the pose
  with both feet just off the platform, pelvis 207 mm off the reference and 23 deg of tilt; the tread 3->4
  pose; the mount pose). A frozen robot sags and tilts there; the puppet, which holds the clip's own poses,
  keeps the bulb at all of them.
- Body scan (overlap indicators, not loads): bulb against the ladder nothing, squeeze 3 N. Ankles and
  hip_roll flagged at steps 0-100 and 160-175: the benchmark's own S04 start stance, as in rebuild 1 (now
  over more steps because the opening is slowed). Left knee at 540 and 1,620-1,625, left hip_yaw at 540.
- Gate (`FIATLUX-S04-DescendWithBulb-v0`, layout seed 0): last frame passes `descended_from_ladder` (pelvis
  0.560 m from the root, limit 0.6); first true at clip step 1,620 of 2,086, mid-dismount, as before.
- `desc01`: unchanged. Clears the benchmark body's shoulder, hip and thigh colliders; hands land at
  0.80-0.82 m/s, nothing fragile in them.
- For training either descent: benchmark body, tight arm terms, and the start randomisation of the climb
  scripts switched off for the platform start (`floor_steps`).

Videos (kinematic): `descbulb02_ref/` side and rear-right, under the videos folder.

## First checkpoints on the benchmark body; foot terms; why a start on the platform throws the robot (2026-11-22)

### `edge87fx` 8,400 (900 iterations on the benchmark body, tight arm terms)
- Benchmark S02, Dex3, free ladder, seeds 0-3: all four reach the platform and stand (the old run at
  6,900: two of four, the other two falling at steps 446-450). None scores: `soles_within_ladder_top` false
  on every seed, the other four conjuncts true. Seed 0: left heel 49 mm over the open edge, right heel
  109-114 mm (old run: 60-68 and 142-143).
- Trainer, benchmark body, 64 robots from the clip's start: 57-60 reach the clip's end (48 when the old
  checkpoint was first put on this body).
- `tight_feet_pos_tracking`, `tight_feet_ori_tracking` (new, `bulb_hand.py`): the two ankle links held to the
  clip's WORLD pose (sigma 0.06 m / 0.3 rad). LadderMan's leg terms are pelvis-relative or averaged over
  several links with a 0.5 m fall-off. On checkpoint 8,400 the position term pays 0.07 of 0.5.
  `lm_train_fxbody.sbatch` FEET=1; multi-script profile suffix `feet`. `edge87fx` restarted with them from
  8,400 (job 15576907, one run).

### The bulb teachers, early
- Trainer, benchmark body, robots reaching the clip's end: `bulb20fx` 15,000 on `bulb20` 54 of 64 (4 when
  first put on this body); `bulb21fx` 14,700 on `bulb21` 26 of 64 after 300 iterations (26 lose a hand hold,
  median step 1,195).
- Benchmark S10, curl 0.9, soft start on waist and arms, two seeds each:
  - `bulb21fx` 14,700: the robot falls at steps 507-512 (the first step up, now without the right wrist's
    rest); the bulb is in the hand, squeeze 3.5 N at most.
  - `bulb20fx` 15,000: reaches tread 1; bulb out of the hand at 507 / 704 with a 37-49 N squeeze, ends
    `bulb_dropped` / `fresh_bulb_struck` at 639 / 740 (the wrist rest under the platform lip).
- `bulb20fx` stopped at 15,000 (superseded clip).

### Descents: three failed starts and their causes
1. `desc01fx`, 4,096 environments, stock pair buffers x1.5: 24 PhysX errors in 68 iterations
   (`foundLostAggregatePairsCapacity`, up to 171.6M asked for, "the simulation will miss interactions").
   A log check that greps for "overflow" does not see these: grep for `PhysX error`.
2. Buffers x3 (`CAPACITY_SCALE`, new): the asked-for size kept growing, 330M then 1.2 billion, and the run
   stalled at iteration 114. On a 16 GB card x3 does not fit at all (CUDA out of memory at start-up).
3. `descbulb02fx` on an H100 node: log frozen for 28 minutes at the URDF-to-USD conversion, 1.9 GB resident.
   It had started 6 minutes after `desc01fx`, which converts the same robot file to the same place. Start
   one job at a time (what the multi script already does for its own runs). Not shown to be the H100.
- Cause of 1 and 2: a reset ON THE PLATFORM throws the robot. Trainer, 64 robots, ankle links, first steps:

  | start | 99th percentile | worst | largest velocity jump |
  |---|---|---|---|
  | `edge87`, on the floor | 0.4 kN, 2.2 kN while settling | 2.2 kN | 1.8 m/s |
  | `desc01`, on the platform | 10-12 kN | 25-28 kN | 12-14 m/s |
  | the same, `init_joint_pos_noise=0` | 5.9 kN | 7.6 kN | 5.5 m/s |
  | the same, `reset_lift=0.03` | 1.5-2.6 kN | one robot at 22-25 kN | 2-3 m/s |
  | `edge87`'s own end stance as a start | 10 kN | 11 kN | 8 m/s |

  The hands read 0 N at step 0; the ankles carry it. The clips keep the soles 1 mm above the platform, and
  LadderMan adds +-0.1 rad to every joint at a reset, in evaluation too. Against the ladder's triangles the
  foot shapes are 0.0 mm inside at frame 0; the trainer still reads 5.9 kN with no noise, unexplained.
  A climb starts at the top in a small share of episodes; a descent in 30 % plus every early phase.
- `command_spawn.py` `reset_lift` (new, default 0): every reset places the root that much higher (the
  benchmark's own on-ladder spawns are 28 mm high). With 0.03: robots ending in the first 100 steps about 5
  of 64 (15 without); the rest end at step 235 (`desc01`) or 345 (`descbulb02`) on the first hand hold, the
  seed being a climb teacher. `lm_train_fxbody.sbatch` RESET_LIFT, FLOOR_STEPS, CAPACITY_SCALE.
- Resubmitted with `RESET_LIFT=0.03`, buffers x3, large-memory GPUs only, one after the other.

## Descents, fourth failure and the cause: thrown robots cross the environment grid (2026-11-22)

- `desc01fx` with `RESET_LIFT=0.03` and pair buffers x3 (job 15577182, L40): clean for about 250 iterations,
  then 476 PhysX errors (466 `collisionStackSize buffer overflow`, the rest pair buffers, up to 227 M asked
  for) and a stall at iteration 396. The lift moved the failure later; it did not remove it.
- The `gpu` partition (`-A uh -p gpu`) is open to this account and not preemptible, but a job there waited
  with a start estimate eight days out. `sbatch --test-only` printing "Preempts: ..." is not a promise to
  start. Withdrawn. `sinfo %e` / FreeMem is free memory without the page cache, not what is available:
  node exclusions made on it were too cautious.
- `bulb21fx` was preempted on `kill-shared` after 3 h 37 min and requeued from checkpoint 15,300.
- `play_probe_resets.py` (new): resets as training makes them (random phase, pose and joint noise,
  exploration noise), 256 robots; a "launch" is a body over 6 m/s within six steps of an episode's start.

  | clip, setting | launched starts | ankle force, 99th pct | fastest body | furthest from own origin |
  |---|---|---|---|---|
  | `edge87` (climb), as trained | 3.4 % | 35 kN | 37 m/s | 2.25 m |
  | `desc01`, no lift | 4.2 % | 33 kN | 42 m/s | 4.39 m |
  | `desc01`, lift 0.03 | 1.9 % | 31 kN | 40 m/s | 4.46 m |
  | `desc01`, lift 0.03, joint noise 0.03 | 2.3 % | 14 kN | 39 m/s | 8.93 m |
  | `desc01`, lift 0.03, speed limit 6, spacing 6 | 0 | 34 kN | 6.0 m/s | 1.88 m |
  | `descbulb02`, the same | 0 | 36 kN | 6.0 m/s | 2.00 m |

  Environments are 3.0 m apart. The CLIMB throws its robots just as often and trains: they stay inside
  their own cell. From the platform a thrown robot is in the air for the 25 or more steps the stock
  terminations need and lands in its neighbours' space. Environments do not collide with each other, but
  the broadphase counts the overlapping pairs, which is the number that grew to 1.2 billion.
  Launches happen at every phase of both clips (the +-0.1 rad joint noise at a reset), not only on the
  platform; earlier entries that blamed the platform start alone were incomplete.
- `bulb_hand.py` `body_speed_exceeded` (new termination, no grace): ends the episode on the step any body
  is over a limit. `lm_train_fxbody.sbatch` SPEED_LIMIT, ENV_SPACING (`task.viewer.env_spacing`).
- Not done: the same check at 4,096 environments (only the cluster can), and whether the climb runs would
  also gain from the speed limit (their robots get within 0.75 m of the next cell).

### Foot terms hurt the climb; stance-only foot terms and a gated mount term (Nov 23 UTC)

- Whole-clip foot terms (`FEET=1`, `tight_feet_*`), added to `edge87fx` to bring the heels inside the
  platform, made S02 in the benchmark worse: checkpoint 8,400 (before them) reaches the platform and stands
  on 4 of 4 seeds, failing only `soles_within_ladder_top` (heels 49-114 mm over the open edge);
  8,700 / 9,000 / 9,300 (after them) reach it on 0 of 4. That run is stopped. 8,400 is pinned.
- `bulb_hand.py` `stance_feet_pos_tracking` / `stance_feet_ori_tracking` (new): the same two ankle terms,
  counted only from `start_step` on and neutral (1.0) before it, so the climb itself is left as trained.
  `lm_train_fxbody.sbatch` `FEET=stance`, `FEET_START` (default 1450); profile suffix `stance` in the multi
  script. Run `edge87st` (job 15579037, from 8,400, `FEET_START=1500`, to 15,900) is training.
- `bulb21fx` (S10 clip `bulb21`, benchmark body, bulb profile). Trainer, clip start, 64 robots: 26, then 8
  just after the pre-emption restart, then 34 complete the clip. Benchmark S10, checkpoints 14,700 / 15,600
  / 15,900: every seed falls at steps 502-528, the first step up with one hand on the ladder. No bulb run
  has passed the first tread.
- Where it goes wrong: at clip steps 380-420 the benchmark robot is 6-8.5 cm behind the clip (4-5 cm in the
  trainer), the left shoulder_roll motor is at its torque limit from then on, and the robot falls backward
  at steps 470-500, when the pelvis should travel forward and up. In the trainer 61 of 64 get through.
  Why the benchmark robot ends 2-3 cm further back than the trainer's is not explained.
- `bulb_hand.py` `mount_root_pos_tracking` (new): the pelvis' world position held to the clip between
  `start_step` and `end_step`, neutral elsewhere. `lm_train_fxbody.sbatch` `MOUNT_ROOT=FROM:TO`. The walk-in
  before it stays free, so a randomised spawn is still absorbed there. Local smoke on checkpoint 15,900 with
  `300:520`: a 3-iteration training run exits 0; in a 530-step probe the term pays 0.287 of 0.5 and 58 of
  64 robots are still up at step 530. That is the term loading and being unsatisfied, not evidence it helps.
- `bulb21fx` restarts from its newest checkpoint with `MOUNT_ROOT=300:520`; checkpoints 16,500 and 16,800
  (before the term) are pinned. To test: S10 on 3 seeds at about 300 and 600 iterations after the restart,
  distance behind the clip at step 400 and whether tread 1 is reached.
- Descents: `desc01fx` (job 15578517) with lift 0.03, speed limit 6, spacing 6 is training at 4,096
  environments with no PhysX errors in its first 92 iterations; the earlier failures came at about 250.

### First checkpoints after the mount and stance terms; the paired job hung (Nov 23 UTC, morning)

- `bulb21fx` with `MOUNT_ROOT=300:520` (job 15579275, from 16,800). S10, 3 seeds, curl 0.9, soft start:

  | checkpoint | outcome | behind the clip at step 400 |
  |---|---|---|
  | 15,900 (before the term) | all fall at 502-528 | 60-85 mm |
  | 17,100 | fall 504, `bulb_crushed` 443, fall 535 | 71, 35, 67 mm |
  | 17,400 | fall 531, 543, 539 | 35, 38, 55 mm |

  The robot arrives closer to the clip and falls a little later, but still falls backward on tread 1.
- Fine trace of 17,400 seed 0 (`--trace 2`): from step 460 the left upper arm presses on the torso with
  130-175 N (`torso_link` and `left_shoulder_yaw_link` read the same force) while left shoulder_roll is at
  its torque limit; at steps 480-484 both feet take 90-120 N of horizontal force pushing the robot away from
  the ladder; the left hand's force goes to zero at step 482 and the robot tips back.
- Not the cause: the actuators. Shoulder effort limit (25 N m), stiffness (100), damping (2) and armature
  are the same in the trainer's `g1_fiatlux.py` and the benchmark's `robots/g1.py`. And not the clip's arm
  pose: `bulb21`'s left upper arm clears the torso hull by 5 mm at its closest (step 478). The left
  shoulder_roll sits 22-35 deg off the clip at steps 360-440 in the TRAINER as well, so the arm-on-torso
  press follows from where the body is, not from the reference.
- Trainer, same checkpoint, left wrist contact through the mount: almost purely vertical, 6-35 N. The hand
  rests on the ladder; it does not pull the robot in. The trainer's robot is 35-78 mm AHEAD of the clip at
  steps 470-500, the benchmark's falls behind from +19 to -89 mm over the same steps.
- The trainer randomises friction on the ankle and wrist links: static 0.3-7.2, dynamic 0.3-3.6
  (`cfg/task/base/ladder.yaml`). The benchmark's ladder grip material is 1.5 / 1.2. Not tested as a cause.
- The trainer's contact sensor covers only the ankle and wrist links, so arm-on-torso force cannot be read
  there with the existing probes.
- `edge87st` (stance foot terms from step 1,500, from 8,400) at 8,700, S02, 4 seeds: two stand on the
  platform with heels 24-115 mm over the open edge, two fall at steps 1,659-1,662. 8,400 stood on 4 of 4.
  In `edge87` the left foot swings onto the platform at steps 1,545-1,580, the right at 1,585-1,620, and the
  left re-steps 4 cm forward at 1,690-1,715; the falls come with both feet already up.
- Two runs in one ece605 job (job 15579646: `edge87st` + `desc01fx`, 12 cores, 220 GB): `edge87st` trained
  18 iterations, then went silent the moment the descent began its start-up beside it, and the descent
  never reached an iteration in 28 minutes. No PhysX errors, no memory limit hit (cgroup 112 GB of 220,
  no OOM events), GPU utilisation 1-2 %. Cause not found. Cancelled; `edge87st` restarted alone from 8,700
  (job 15579998) and `desc01fx` put back on `kill-shared` (job 15579999, 12 h limit).
- `lm_train_fxbody_multi.sbatch`: descent settings (lift, speed limit, spacing) now go only to a descent
  run in a pair (`DESC_LIFT`, `DESC_SPEED`, `DESC_SPACING`); `CPUS_PER_RUN`.
- Pending descents changed in place to a 12 h limit and released from the single-node pin.
