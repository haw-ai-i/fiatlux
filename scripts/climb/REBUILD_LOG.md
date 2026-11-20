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
