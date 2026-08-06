# Climb Reference-Motion Pipeline

## Summary

Produce reference ladder-climbing motion for the G1, expressed against the fiatlux ladder and the fiatlux robot, and land it in this repo as a data-collection track alongside teleoperation and asset generation. The output is a versioned motion format plus the loader and replay tooling that consume it.

A clip is one retargeted motion: a directory holding `motion_50hz.npz` and `meta.json`.

Two downstream consumers, in priority order:

1. **A tracking reward for `FIATLUX-Climb-v0`.** The current climb policy learns from shaping alone (`climb_env_cfg.py`: `climb_height_progress`, `ladder_contact_fraction`, sway and wobble penalties) and reaches 18.0% success at a 24.0% fall rate (climb baseline table, `sections/4. results.tex`). No reference motion exists anywhere in this codebase. A tracking target converts climbing from an exploration problem into a tracking problem, which is the structure LadderMan uses.
2. **Demonstration data for imitation pre-training and VLA fine-tuning** (roadmap item 3,
   `sections/5. discussion.tex`). Every VLA row in the replacement baseline table is currently 0.0%.

## LadderMan

LadderMan is the only published work so far that performs this exact task on this exact robot: a Unitree G1 climbing real ladders, zero-shot from simulation.

It also diagnoses the current climb baseline. Its ablation compares hybrid motion tracking against DeepMimic-style tracking and against variants with contact tracking or the climbing reward removed: plain tracking fails on any geometry other than the captured one, and dropping either contact tracking or the task reward degrades learning. Our 18% at a 24% fall rate comes from shaping with no reference motion at all, which is upstream of even the weakest arm of that ablation.

The critical finding is that the method needs one reference motion, not a dataset. LadderMan collects a single OptiTrack capture on a ladder at 65.5° inclination with 24.8 cm rung spacing, then learns expert policies for different ladder geometries from that one clip. Real-world success across three physical ladders: 9/10 on the captured geometry, 6/10 and 7/10 on two others. A single good capture is therefore a viable deliverable, which is what makes this effort tractable inside the schedule.

Where we diverge is how that clip is obtained. We have no motion-capture stage, so the goal of this effort is to produce LadderMan's one reference motion from an ordinary recording — RGB video, and where it helps the LiDAR depth and camera poses an iPhone can capture alongside it — and retarget that onto the G1. Nothing downstream cares how the motion was captured, only whether it is metrically accurate; _Capture without mocap_ below argues that a climb filmed against a measured ladder is recoverable without a suit. This substitution is the effort's central bet and its largest technical risk.

Two design consequences, both of which shape this spec:

- **Contact events are annotated once, by hand,** for the four end-effectors on the reference motion. They become the reference contact indicator the contact-tracking reward consumes.
- **Contact targets are generated from the ladder geometry,** not baked into the clip: for a given ladder, predefined contact locations on the target rung are assigned per hand and foot. The motion is never warped to fit the ladder; the reward pulls the limbs to geometry-derived targets.

That is cheaper than retiming a clip per ladder, and it is what makes one capture generalize.
This effort therefore owes one good clip, a hand annotation of its four contact channels, and a loader — not a corpus, and not a per-ladder warp.

## OmniRetarget As The Retargeting Engine.

OmniRetarget builds an **interaction mesh** spanning the human, the terrain, and any manipulated object, then minimizes Laplacian deformation between the human and robot meshes while enforcing **hard** kinematic constraints: foot sticking, non-penetration, and joint limits. It preserves the spatial and contact relationships that generic joint-level retargeting discards. Demonstrated on a Unitree G1: dynamic climbing onto a 0.9 m platform (70% of robot height), slope crawling, and a 30-second parkour course, all zero-shot sim-to-real. The RL recipe on top is deliberately minimal — 5 rewards, 4 domain-randomization terms, purely proprioceptive observations, no curriculum, shared across tasks.

Three consequences for this spec:

- **It supersedes GMR for stage 2.** GMR is one of OmniRetarget's three open-source baselines (alongside PHC and VideoMimic), and is outperformed on penetration, foot skating, and contact preservation. GMR stays as the Phase 1 path and the fallback, because it works today.
- **It absorbs the feasibility pass.** Non-penetration and foot sticking are hard constraints at retarget time rather than defects to repair afterward. What remains is a validation check, not a pipeline stage, and the SUGAR refiner drops out of the plan.
- **It generates ladder-geometry variation from a single clip.** Augmenting to new terrains and object configurations requires modifying only the interaction-mesh keypoint correspondences and the robot collision model. That is exactly the per-geometry reference data LadderMan's experts need, generated rather than captured, which pulls geometry randomization out of Phase 3 and into Phase 2.

Code ships as [`amazon-far/holosoma`](https://github.com/amazon-far/holosoma) (Apache-2.0, G1 supported) with the retargeting module as `holosoma_retargeting`, plus a released dataset. It is a full framework spanning IsaacGym / IsaacSim / MuJoCo, so the integration question is how much to take — see _Decision: holosoma integration_.

Inputs are SMPL parametric data (via forward-model keypoints) or skeleton hierarchies, so a video-derived SMPL-X source composes. Contact handling needs verification against `holosoma_retargeting` before Phase 2 commits: the interaction mesh needs to know which correspondences to preserve, so the hand annotation likely becomes an **input** to retargeting as well as a reward signal, and the `contacts` array in the output can then be derived geometrically from limb-to-rung proximity rather than transcribed.

## Capture without mocap

LadderMan used OptiTrack. We do not have a mocap stage, which is the gap this section closes.

Of the four inputs the hybrid-tracking reward consumes, only one actually depends on the capture modality:

| Input                                 | Source                                      | Needs mocap?                         |
| ------------------------------------- | ------------------------------------------- | ------------------------------------ |
| Reference contact indicator           | hand-annotated once                         | No — LadderMan annotates by hand too |
| Target contact positions              | generated from ladder geometry              | No — never touches the capture       |
| Upper-body reference pose             | capture, deliberately relaxed by the method | Barely                               |
| Lower-body joints and root trajectory | capture                                     | **Yes — the real dependency**        |

That last row is where monocular video is weakest: metric scale and global root drift. Two properties of ladder climbing close the gap:

1. **The ladder is a ruler.** A rigid object of known dimensions is in frame throughout. Solving camera pose and scale against it pins the metric scale that monocular pose estimation can only infer from a body-size prior.
2. **Contacts are anchors.** When a limb is annotated in contact with rung _k_, its 3D position is known in the ladder frame. Every contact frame is a hard constraint on the root-trajectory solve, which is what kills drift.

Capture is an iPhone 14 Pro, which has LiDAR. Record3D or Stray Scanner export RGB, depth (ARKit `sceneDepth`, 256×192, usable to ~5 m), and **metrically-scaled per-frame camera poses with a gravity vector**. Depth does not survive occlusion — a limb behind a rail is hidden in depth too — but it makes scale and "up" exact, and a scan of the physical ladder measures rung spacing and inclination off the point cloud.

Note that the depth and camera poses make the _world-frame_ capability of stage-1 estimators largely redundant: we solve global trajectory ourselves from depth plus contact anchors. The stage-1 selection criterion is therefore per-frame accuracy under occlusion, not world-frame quality.

The Pico 4 Ultra headset plus ankle trackers (arriving ~2026-08-13) are sparse metric capture — head and both ankles in 6-DoF, which is most of what OptiTrack was providing for a climb. Closer to LadderMan's setup than video.

## Scope

- **Lives in this repo as a folder**, not a submodule and not a separate repository. Following the pattern Brian established for the asset pipeline: only driver scripts are committed, and those scripts clone the upstream repositories and download model weights at setup time.
- **Stage 1 is PromptHMR** (3D human mesh recovery; outputs SMPL-X 55-joint pose), unchanged from the winter stack in `bushuyeu/iolani-g1-workshop`. Candidates for a Phase 2 bake-off are listed under _Stage-1 alternatives_.
- **Stage 2 is OmniRetarget**, with GMR as the Phase 1 path and the standing fallback.
- **Embodiment is fiatlux's G1**, `G1_INSPIRE_CFG`, not a retarget-tool-native URDF. The Inspire hand can grasp a rail, and the reference names the bodies and joints this repo already uses.
- **Ladder is the fiatlux climb ladder**: `STEP_LADDER_USD` (kinematic `_collision` variant), 0.68 × 1.11 × 1.75 m, at `LADDER_POSITION = (1.6, 0.0, 0.0)`, `LADDER_YAW_DEG = 90.0`. Its collision mesh is also the interaction-mesh input, so it must be exportable from USD.
- **50 Hz**, matching the family control rate (`sim.dt = 1/200`, `decimation = 4`).
- **The existing shaping reward stays.** A tracking term is added alongside `climb_height_progress` and never replaces it, because the 18% result is the paper's published comparison point.
- **Clips are committed to the repository**, not synced from GCS. A 10 s clip is roughly 400 kB (500 frames × ~205 float32 values), three orders of magnitude below the binary USD assets that motivated `assets/download_assets.sh`. Revisit past ~50 clips.

Out of scope:

- **Reproducing SUGAR.** Its retargeting stage and sim-to-sim pipeline are both unreleased, it is 29-DoF with a non-grasping rubber hand, and its contact representation is a single scalar boolean where this repo already has four filtered limbs. OmniRetarget now covers the one idea worth borrowing (making kinematic motion physically consistent), so SUGAR's remaining contribution is the motion schema, which is shared with beyondmimic anyway.
- **A converged tracking policy.** This spec delivers the clips, the loader, and the reward term. Training to convergence is separate work on PSC Bridges-2.
- **Per-ladder motion warping.** Superseded twice over: by geometry-derived contact targets, and by OmniRetarget's augmentation.

### Stage-1 alternatives

- **SAM-Body4D** — training-free video HMR over SAM-3 segmentation with explicit occlusion recovery. The most on-target candidate for a subject disappearing behind rails.
- **SAM 3D Body** — promptable full-body HMR, strong on occlusion and hard viewpoints. Uses the Momentum Human Rig rather than SMPL-X, so it needs a conversion before stage 2.
- **VideoMimic** — reconstructs motion _and_ terrain from video for human-terrain traversal, limited to static-scene interactions. Our ladder is kinematic, so that limitation is not binding. OmniRetarget benchmarks against it and criticizes its artifacts, but as a combined stage-1-and-2 path for a static scene it is the closest published match to this task and deserves an hour before it is dismissed.

## Deliverable: the motion contract

```
assets/motions/climb/<clip_id>/
  motion_50hz.npz
  meta.json
```

`motion_50hz.npz` holds float32 arrays unless noted, over `T` frames at 50 Hz:

| Array                              | Shape                      | Notes                                                                          |
| ---------------------------------- | -------------------------- | ------------------------------------------------------------------------------ |
| `joint_pos`                        | `(T, 29)`                  | body joints, fingers excluded. Provisional, see _Decision: DoF scope_          |
| `joint_vel`                        | `(T, 29)`                  | finite-differenced, then low-pass filtered                                     |
| `root_pos_w`                       | `(T, 3)`                   | pelvis, world frame, ladder at its configured pose                             |
| `root_quat_w`                      | `(T, 4)`                   | wxyz, the Isaac Lab convention                                                 |
| `root_lin_vel_w`, `root_ang_vel_w` | `(T, 3)`                   |                                                                                |
| `body_pos_w`, `body_quat_w`        | `(T, NB, 3)`, `(T, NB, 4)` | tracked bodies, below                                                          |
| `body_lin_vel_w`, `body_ang_vel_w` | `(T, NB, 3)`               |                                                                                |
| `contacts`                         | `(T, 4)` bool              | per-limb contact, derived from limb-to-rung proximity in the retargeted motion |

The layout deliberately mirrors the beyondmimic and SUGAR motion formats. That is what a tracking reward wants to read, and it leaves the option of borrowing upstream tracking code without a translation layer.

If _Decision: DoF scope_ resolves to 53, `joint_pos` and `joint_vel` become `(T, 53)` and the finger columns are filled from a scripted grip pose rather than from the retarget.

**Tracked bodies** (`NB = 10`) are the four climb limbs plus the chain needed to define a pose: `pelvis`, `torso_link`, both `*_knee_link`, both `*_ankle_roll_link`, both `*_elbow_link`, both `*_hand_base_link`.

**Contact columns are addressed by name, never by position.** The `ladder_contact` sensor resolves `.*(ankle_roll|hand_base)_link` through Isaac Lab's body ordering, which is not guaranteed stable across asset revisions. `meta.json` records the resolved body-name list, and the loader indexes by name.

`meta.json` records: `clip_id`; source video path or capture session; the ladder geometry the clip was retargeted against (asset constant plus rung spacing and inclination, since augmented variants differ); whether the clip is captured or augmented, and from which parent `clip_id`; robot variant (`inspire`); retargeting tool and commit SHA; `fps`; frame count; duration; the contact body-name order; and free-text notes on known defects.

## Pipeline

```
iPhone RGB-D + ARKit poses
  └─ PromptHMR ──────→ SMPL-X, 55 joints
       └─ grounding ─→ metric root trajectory (depth + ladder calibration + contact anchors)
            └─ annotation ─→ 4-channel contact events (manual, once per capture)
                 └─ OmniRetarget ─→ G1 motion, contacts preserved, penetration-free
                      ├─ validation ──→ motion_50hz.npz + meta.json
                      └─ augmentation ─→ variants over rung spacing / inclination
```

**Grounding** solves the metric root trajectory using iPhone depth, the ARKit camera pose and gravity vector, the measured ladder geometry, and the annotated contact frames as anchors. This replaces the naive rigid-transform alignment of earlier drafts; it is the step that substitutes for mocap.

**Annotation** labels, for each of the four end-effectors, the frames where that limb is in contact. Manual, once per capture. It feeds both the grounding solve and the interaction mesh, and it is the one step with no automated substitute.

**OmniRetarget** consumes the grounded SMPL-X motion plus the ladder collision mesh and produces G1 joint trajectories with contacts preserved and penetration and foot skating constrained away.

**Validation** replaces the old feasibility pass. Rather than repairing the motion, measure it and reject: penetration depth and duration, foot-skating duration and peak velocity — the metrics OmniRetarget itself reports — plus contract conformance. A clip either passes or goes back.

**Augmentation** re-runs retargeting against ladder variants (rung spacing, inclination) to produce the per-geometry reference set LadderMan's experts consume. Each variant records its parent `clip_id`.

## Repository integration

```
scripts/retarget/
  setup.sh             # clones PromptHMR + holosoma, downloads weights
  capture_notes.md     # capture protocol (framing, standoff, lighting, ladder scan)
  run_pipeline.py      # capture → motion_50hz.npz + meta.json
  annotate_contacts.py # frame-by-frame contact labelling, writes the contacts array
  augment_geometry.py  # ladder variants → additional clips
  validate_clip.py     # penetration / skating / contract checks
  replay_motion.py     # kinematic playback of a clip in the Climb-v0 scene, --record
source/fiatlux_task/fiatlux_task/motions.py   # loader; owns the tracked-body and contact contract
```

The loader lives in the package because the environment imports it. The pipeline lives in `scripts/` because the environment must never import PromptHMR, holosoma, or any capture dependency.

## Embodiment reconciliation

| Aspect       | `FIATLUX-Climb-v0`                                       | Retargeter output            | Resolution                                                   |
| ------------ | -------------------------------------------------------- | ---------------------------- | ------------------------------------------------------------ |
| DoF          | 53, `joint_names=[".*"]`, Inspire fingers                | 29                           | _Decision: DoF scope_                                        |
| Hand bodies  | `left/right_hand_base_link`                              | keypoint correspondence only | semantic hand-to-hand mapping in the interaction mesh        |
| Contact      | 4-limb filtered force matrix against `/Ladder`           | preserved by construction    | derived geometrically, verified against the sensor at replay |
| Ladder       | USD collision mesh                                       | interaction-mesh input       | export the mesh once; reuse for augmentation                 |
| Control rate | 50 Hz                                                    | varies by source             | resampled to 50 Hz                                           |
| Action       | position targets, `scale=0.5`, `use_default_offset=True` | joint angles                 | the tracking reward compares joint positions, not actions    |

## Phases

**Phase 1 — first results** Measure and scan the physical ladder (rung spacing, inclination) — this gates grounding and is the cheapest task on the list. Capture one climb in RGB-D. Take it through PromptHMR and **GMR** — the working path, not OmniRetarget — to a 29-DoF trajectory, and replay it kinematically in the Climb-v0 scene with a recording. Expected outcome: joint-space retargeting works; contacts do not yet coincide with the rungs. Depth and ARKit poses are banked, not yet used.

**Phase 2 — usable reference, 2026-08-08 to 2026-08-15.** Stand up `holosoma_retargeting` against the exported ladder mesh; grounding solve; contact annotation; the motion contract frozen and `motions.py` landed; `validate_clip.py`; a tracking reward term added alongside the existing shaping; geometry augmentation producing the first variant set. One validated capture satisfies the method — additional _captures_ are optional, but augmented _variants_ are the point. This window decides whether the paper gets a fine-tuned baseline row.

**Phase 3 — from 2026-08-13, overlapping Phase 2's tail; gated on tracker delivery.** Tracker-based capture with Yujin Chen on the Pico 4 Ultra setup, for poses monocular video cannot recover: single-view occlusion behind an A-frame is severe, with hands wrapping rails and the body edge-on. Then per-geometry experts and distillation on the augmented set, the stage-1 bake-off, and demonstration export for VLA fine-tuning. The camera-based-mocap-without-a-suit contribution belongs here.

## Open decisions

**Decision: DoF scope.** Stage 2 emits 29 body joints; `FIATLUX-Climb-v0` actions cover all 53 including fingers. Either lift the reference to 53 with fingers driven by a scripted grip pose, or restrict the tracked joint subset to the 29 body joints and leave fingers to the existing `joint_deviation_fingers` term. Recommendation: the latter, because finger motion recovered from SMPL-X is unreliable and the Inspire grip is better scripted than tracked. This changes the environment configuration, so it needs agreement before Phase 2 builds on it.

**Decision: holosoma integration.** `amazon-far/holosoma` is a full framework (IsaacGym, IsaacSim, MJWarp, MuJoCo) and we need one module from it. Options: clone-at-setup and call `holosoma_retargeting` as a subprocess, keeping it out of the fiatlux environment; or vendor the retargeting module. Recommendation: clone-at-setup, matching the scope rule above and avoiding a second simulator stack in this repo. Confirm the module runs standalone before Phase 2 depends on it.

## Risks

- **holosoma integration cost** (high, new). A full multi-simulator framework absorbed in one week, during the freeze. Mitigation: Phase 1 deliberately ships on GMR, so Friday does not depend on it; GMR remains the fallback if the module does not run standalone.
- **Grounding accuracy** (high). Metric root trajectory from monocular-plus-depth is the substitute for mocap, and it is unproven here. Mitigation: contact anchoring against measured rung positions; the ankle trackers as a Phase 3 backstop.
- **Occlusion** (high). A person on a ladder is a hard case for every stage-1 method, and depth does not see through rails. Mitigation: oblique framing, multiple takes, occlusion-focused stage-1 candidates in the Phase 2 bake-off.
- **Schedule collision** (medium). Phase 2 lands in the same week as the infrastructure freeze. Mitigation: the motion contract is what unblocks everyone else, so freeze it early even with a single clip.
- **Augmentation fidelity** (medium, new). OmniRetarget's augmentation is demonstrated on platforms and slopes, not ladders; A-frame rung contact may be harder. Mitigation: validate augmented variants with the same penetration and skating checks as captured clips, and treat variant count as a result rather than a plan.

## Acceptance criteria

Phase 2 is complete when:

- the motion contract is frozen and documented;
- at least one captured clip validates against it;
- `validate_clip.py` reports penetration and foot-skating within the thresholds OmniRetarget reports for its own G1 outputs;
- kinematic playback shows limbs meeting rungs, confirmed against the `ladder_contact` sensor rather than by eye;
- at least one augmented geometry variant validates;
- the tracking term trains without destabilizing the existing shaping;
- the 18% baseline re-measures unchanged with the tracking term disabled.

Verification commands, runnable once the corresponding pieces land:

```bash
conda activate env_isaaclab && export PYTHONPATH=$PWD/source/fiatlux_task

# contract conformance: shapes, fps, wxyz ordering, contact names resolve
python -m pytest source/fiatlux_task/fiatlux_task/tests -k motions

# penetration / foot-skating / contract checks on a clip
python scripts/retarget/validate_clip.py --clip <clip_id>

# kinematic playback in the climb scene, recorded to logs/runs/retarget
python scripts/retarget/replay_motion.py --clip <clip_id> --task FIATLUX-Climb-v0 --record

# no regression in the climb scene
python scripts/verify_scene.py --headless --task FIATLUX-Climb-v0 --num_envs 1

# tracking reward smoke test
python scripts/rsl_rl/train.py --task FIATLUX-Climb-v0 --num_envs 32 --max_iterations 3 --headless
```

## Tickets

Spec-local IDs (`R1`…), not GitHub numbers. Size is S (half a day or less), M (one to two days), L (three or more). The critical path is R2 → R4 → R9 → R11 → R13; R9 is the long pole and the one to start early.

One thing to settle while filing: the physical ladder we film is almost certainly not the BEHAVIOR-1K asset the sim uses. That is fine and expected — film against whatever ladder exists, measure _it_, ground against _its_ geometry, and let augmentation carry the motion to the sim ladder. It is the same A-to-B generalization LadderMan demonstrates. But it must be deliberate, not discovered in week two.

### Phase 1 — by 2026-08-07

**R1 — Tracking issue: climb reference-motion pipeline.** S. No dependencies.
The umbrella issue: scope, priorities, a link to this spec, and a checklist of R2–R19.
_Done:_ issue filed, linked from this spec, both open decisions flagged for comment.

**R2 — Ladder geometry package.** S. Blocks R6, R9, R13.
Measure the physical ladder (rung spacing, inclination, rail geometry); scan it with the iPhone; export the collision mesh from `STEP_LADDER_USD` for use as interaction-mesh input.
_Done:_ measurements recorded in `capture_notes.md`, scan and exported mesh stored, and the delta between the physical and simulated ladder written down rather than assumed away.

**R3 — Repo scaffold and `setup.sh`.** S. Blocks R5.
`scripts/retarget/` skeleton; clone-at-setup for PromptHMR with weight download.
_Done:_ a clean checkout reaches a runnable PromptHMR with one command.

**R4 — Capture protocol and first RGB-D capture.** S–M. Depends on R2.
Write `capture_notes.md` (standoff, oblique framing, full-body-at-top, lighting, empty-ladder head frames), then shoot several takes and keep one.
_Done:_ one clean ascent stored with RGB, depth, and ARKit poses; protocol written so a second person could repeat it.

**R5 — `run_pipeline.py` v0: PromptHMR → GMR.** M. Depends on R3, R4.
The working path only — no grounding, no contacts, no OmniRetarget. RGB in, 29-DoF trajectory out.
_Done:_ one capture produces a joint trajectory at 50 Hz.

**R6 — `replay_motion.py`.** M. Depends on R2, R5.
Kinematic playback of a trajectory in the Climb-v0 scene, with `--record`.
_Done:_ a recording exists showing the G1 executing the retargeted climb, contacts not yet expected to land.

### Phase 2 — 2026-08-08 to 2026-08-15

**R7 — Freeze the motion contract; land `motions.py` and its tests.** M.
The loader, the tracked-body set, name-indexed contact resolution, and contract conformance tests.
_Done:_ `pytest -k motions` passes against a real clip; the contract is documented and no longer moving.

**R8 — `annotate_contacts.py`.** M. Depends on R4.
Frame-by-frame labelling of the four end-effectors, writing the `contacts` array.
_Done:_ one capture fully annotated; annotation round-trips through the loader.

**R9 — Grounding solve.** L. Depends on R2, R4, R8. **Highest risk.**
Metric root trajectory from depth, ARKit poses and gravity, measured ladder geometry, and contact anchors.
_Done:_ root ascent is monotonic and matches the measured climb height to within a tolerance recorded in the ticket; annotated contact frames place the limb at the correct rung.

**R10 — holosoma spike.** S–M. Gates _Decision: holosoma integration_.
Can `holosoma_retargeting` run standalone against our exported mesh, without pulling a second simulator stack into this repo?
_Done:_ a written answer plus a minimal working invocation, or a documented reason to stay on GMR.

**R11 — Stage 2 swap to OmniRetarget.** M. Depends on R9, R10.
Replace GMR in `run_pipeline.py`, keeping it behind a flag as the fallback. Resolve how contact annotation feeds the interaction mesh.
_Done:_ one clip retargeted through OmniRetarget; GMR still selectable.

**R12 — `validate_clip.py`.** M. Depends on R7, R11.
Penetration depth and duration, foot-skating duration and peak velocity, contract conformance, with thresholds taken from OmniRetarget's reported G1 numbers.
_Done:_ pass/fail on a clip with the metrics printed; the captured clip passes.

**R13 — `augment_geometry.py`.** M. Depends on R2, R11, R12.
Re-retarget across ladder variants (rung spacing, inclination), including the sim ladder if it differs from the filmed one. Each variant records its parent `clip_id`.
_Done:_ at least one variant validates through R12.

**R14 — Tracking reward term.** M. Depends on R7.
A motion-tracking term in `climb_env_cfg.py` alongside the existing shaping, plus the agent-cfg wiring, with a switch to disable it.
_Done:_ PPO smoke test iterates; the term is off by default until R15 clears.

**R15 — Baseline regression re-measure.** S. Depends on R14.
Re-run the climb baseline with the tracking term disabled.
_Done:_ 18.0% success / 24.0% fall rate reproduce within noise, confirming the paper's number still stands.

### Phase 3 — from 2026-08-13

**R16 — Tracker capture path.** M. Yujin Chen and Pavel Bushuyeu.
Pico 4 Ultra plus ankle trackers into the same contract, for poses video cannot recover.
_Done:_ one tracker-sourced clip validates through R12.

**R17 — Stage-1 bake-off.** M.
SAM-Body4D, SAM 3D Body, and VideoMimic against our own capture, judged on monotonic ascent, foot-height agreement with measured rungs, and behaviour through occlusion.
_Done:_ a recommendation with the comparison recorded, and stage 1 either swapped or explicitly kept.

**R18 — Per-geometry experts and distillation.** L. Depends on R13, R14.
LadderMan's second stage on the augmented set.
_Done:_ climb success rate reported against the 18% baseline.

**R19 — Demonstration export for VLA fine-tuning.** M. Depends on R12.
Clips into whatever format the imitation and VLA work consumes.
_Done:_ a fine-tuning run ingests the export without bespoke conversion.

## References

- OmniRetarget: Interaction-Preserving Data Generation for Humanoid Whole-Body Loco-Manipulation and Scene Interaction. Yang, Huang, Wu, Kanazawa, Abbeel, Sferrazza, Liu, Duan, Shi (Amazon FAR, MIT, UC Berkeley, Stanford, CMU). <https://omniretarget.github.io>, copy at `journal/references/omniretarget.pdf`. Code: <https://github.com/amazon-far/holosoma> (Apache-2.0). Dataset: <https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset>.
- LadderMan: Learning Humanoid Perceptive Ladder Climbing. Zhao, Zhang, Lu, Abbeel, Duan, Sreenath, Wang, Liu, Shi (Amazon FAR, USC, UC Berkeley, Stanford, CMU). <https://ladderman-robot.github.io>, copy at `journal/references/ladderman.pdf`.
- SUGAR: A Scalable Human-Video-Driven Generalizable Humanoid Loco-Manipulation Learning Framework. arXiv:2605.20373, copy at `journal/references/sugar.pdf`. Retained for the motion schema only.
- GMR: General Motion Retargeting (ICRA 2026). <https://github.com/YanjieZe/GMR>. Phase 1 path and fallback.
- Stage-1 candidates: SAM 3D Body (arXiv:2602.15989, <https://github.com/facebookresearch/sam-3d-body>), SAM-Body4D (arXiv:2512.08406), VideoMimic.
- `fiatlux-report-2026-q3` (separate repository) — `sections/4. results.tex` for the climb baseline, `sections/5. discussion.tex` for roadmap item 3.
- `journal/syncs/` — 2026-07-22 (LadderMan review), 2026-07-31 (fine-tuning cutoff), 2026-08-03 (trackers, repository renewal), 2026-08-05 (folder not submodule).
- `bushuyeu/iolani-g1-workshop` — the PromptHMR → GMR stack being renewed.
