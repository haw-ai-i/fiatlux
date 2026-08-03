# Bulb Attach/Detach: Bayonet Socket Mechanic

Issue: https://github.com/haw-ai-i/fiatlux/issues/54
PR: https://github.com/haw-ai-i/fiatlux/pull/65

Status: IMPLEMENTED AND VERIFIED (2026-08-03, RTX 3090 / Isaac Sim 5.1 / Isaac Lab
2.3.2): `verify_attach.py` 21/21 + `--check-ranges` 1/1; `verify_scene.py` base
31/31, Install 34/34, Remove 34/34, Replace 43/43.

---

## 1. Problem statement

`FIATLUX-Replace-v0` requires the old bulb to start "screwed into" the fixture yet be
removable, and the fresh bulb to become "screwed in" once installed. Plain rigid-body
physics provides neither:

- The old bulb was a kinematic stand-in. No policy could move it, so every
  removal/disposal score channel and the `success` termination were
  scored-but-unachievable.
- Fresh-bulb "insertion" was a purely geometric seating check. A bulb could be waved
  through the success zone or knocked out of it after scoring.

An attachment mechanic must make both events real. It must be driven by the **bulb's
motion**, not by robot state: any proxy signal (e.g. wrist pose) can be satisfied
without the bulb actually moving, which rewards a gesture instead of a manipulation.

## 2. Requirements

The socket is modeled as a **bayonet mount**, driven entirely by the bulb's pose
relative to the socket:

- **Install**: insert, then rotate — the bulb, not the wrist.
- **Remove**: rotate back, then eject.
- While the bulb travels along the socket axis it **cannot rotate**.
- Once it has started rotating (at full depth) it **cannot travel along the axis**.
- `insertion_depth` and `rotation_angle` are **parametric** and per-env randomizable
  for domain randomization.

The two motion regimes are mutually exclusive, so the required order is structural,
not reward-shaped: no sequence of pushes can free a locked bulb, and no fresh bulb
counts as installed until it has bottomed out and turned through the lock angle. How
the robot produces bulb rotation (finger friction, palm contact) is left to contact
physics — the mechanic constrains only the bulb.

Non-functional requirements:

- Compatible with the GPU-vectorized pipeline: per-env state, no runtime USD edits.
- Throughput regression at standard env counts within noise.
- No coupling to robot morphology or joint naming.
- Existing score-channel and telemetry contracts (term names) unchanged.

## 3. Asset geometry

Bulb and socket are the two halves of the Omniverse Sample-Scenes `LightBulb`
(`assets.py`: `LightBulb_bulb_z_rigid.usda` / `LightBulb_socket_z_static.usda`;
regenerate with `scripts/omniverse/omniverse_bulb_rigid.py`).

- The socket hole is genuinely open: exact-triangle-mesh colliders
  (`physics:approximation = "none"`), so a lowered bulb nests and rests by contact.
  Exact meshes are illegal on dynamic bodies, so the socket stays static/kinematic.
- Both halves are authored assembled at identity: seated ⇔ bulb root pose == socket
  root pose; `SOCKET_SEAT_OFFSET == BULB_PLUG_OFFSET == (0, 0, 0.036259)`.
- The mating axis is explicit: `BULB_PLUG_AXIS == SOCKET_SEAT_AXIS == +Z`;
  `verify_scene` re-measures the offsets from geometry at runtime.
- Both bulbs spawn dynamic; authored masses (bulb 35 g, fixture 0.30 kg).

Contact geometry alone already provides retention (a bulb rests in the hole, even
inverted — measured 1.9 mm settle in a ceiling mount). The mechanic adds the ordering
constraint that makes removal an unscrew rather than a pick-up.

## 4. Design: state machine

Per bulb, per env. Persistent state: `phase` and the lock angle `theta` (current
rotation toward locked, in `[0, rotation_angle]`). The axial coordinate is read from
the live pose each step, never stored.

```
             enter channel                    bottomed + twist begins
   FREE  ────────────────────────▶  AXIAL  ─────────────────────────▶  ROTATING
    ▲    aligned, socket empty        │ ▲                                  │
    │                                 │ │      theta back to 0             │
    └─────────────────────────────────┘ └──────────────────────────────────┘
             travel past depth
             (ejected)
```

- **FREE** — unconstrained rigid body. Physics owns it entirely.
- **AXIAL** — the insertion channel. The bulb keeps only its axial coordinate,
  clamped to `[0, insertion_depth]` measured from the seat; lateral offset and all
  rotation relative to the socket are projected away. Entered from FREE when the plug
  point is inside the channel mouth, the bulb is aligned within tolerance, and the
  socket holds no other bulb; entered from ROTATING when `theta` returns to 0.
- **ROTATING** — the lock groove. Position pinned at full depth; the bulb keeps only
  twist about the socket axis, tracked as `theta` and clamped to
  `[0, rotation_angle]`. Entered from AXIAL when the bulb is at full depth and its
  twist moves in the locking direction.

Attachment is derived, not stored:

- `old_bulb_attached` = `phase == ROTATING` (resets there with
  `theta = rotation_angle` — locked).
- `fresh_bulb_attached` = `phase == ROTATING and theta >= rotation_angle`.

Removal is therefore ROTATING → (theta→0) → AXIAL → (travel past depth) → FREE;
installation is the exact reverse.

### Parameters (all on the event term)

| name                    | default   | meaning                                               |
| ----------------------- | --------- | ----------------------------------------------------- |
| `insertion_depth`       | `0.034` m | seat-to-mouth travel (measured socket geometry)       |
| `rotation_angle`        | `π/2` rad | released → locked twist (quarter turn)                |
| `rotation_sign`         | `+1`      | which twist direction locks                           |
| `radial_tolerance`      | `0.015` m | channel-entry lateral tolerance (= seating tolerance) |
| `orientation_tolerance` | `0.2` rad | channel-entry axis-alignment tolerance                |
| `seat_tolerance`        | `0.004` m | "bottomed" gate: locking may begin within this axial distance of the seat (contact stops the bulb slightly short of exact zero) |

`insertion_depth` and `rotation_angle` accept a scalar or a `(low, high)` range;
ranges are sampled independently per env at every reset. This is the domain
randomization hook — no other code changes are needed to randomize the socket
geometry.

## 5. Design: enforcement by per-step pose projection

Alternatives considered and rejected for the GPU-vectorized pipeline:

| mechanism                    | verdict        | why                                                                                                        |
| ---------------------------- | -------------- | ---------------------------------------------------------------------------------------------------------- |
| breakable joints             | reject         | break events are CPU callbacks; not a supported Direct-GPU type; broken-joint resurrect bug (PhysX #200)   |
| runtime joint create/destroy | reject per-env | USD stage edit + physics re-parse; impractical at RL scale                                                 |
| `kinematic_enabled` toggling | reject         | spawn-time property; per-env runtime pose writes to kinematic bodies broken on GPU (Isaac Lab #3646/#2069) |
| Factory-style SDF threads    | out of scope   | physically faithful, but needs thread-quality collision meshes + heavy contact tuning                      |

All joints except D6 are unsupported by the PhysX Direct-GPU pipeline; a persistent
per-env D6 with tensor-toggled drives remains a possible physical backend, deferred
pending an on-GPU spike.

Enforcement is therefore **projection**: an every-step event term
(`mode="interval"`, `interval_range_s=(0.0, 0.0)`) that

1. reads the bulb pose in the socket frame (axial coordinate, lateral offset, twist),
2. advances the state machine,
3. writes back the projected pose and velocity through the tensorized
   `write_root_pose_to_sim` / `write_root_velocity_to_sim` — pose components the
   current phase forbids are removed; velocity is projected onto the allowed axis
   (linear-axial in AXIAL, angular-twist in ROTATING, with a hard stop at
   `theta = rotation_angle`).

No USD edits, no joints, no per-env stage state — the same API surface episode resets
already use, valid per-env on GPU.

## 6. Design constraints

- **No robot state.** The state machine reads two rigid-body poses (bulb, socket) and
  nothing else — no palm bodies, wrist joints, grasp radii, or roll accumulators.
- **One code path for both bulbs.** Old and fresh bulb differ only in reset phase
  (`ROTATING`+locked vs `FREE`): same advance function, same projection, applied to
  different state slices.
- **No hidden bookkeeping flags.** `theta` integrates against the twist of the pose
  the projection last wrote (or the spawn pose), so transition and reset steps need
  no special-casing.
- **Scoring reads state, not geometry.** `fresh_bulb_inserted` and `success` gate on
  `fresh_bulb_attached`; the old-bulb channels gate on `phase != FREE`. Because
  `ManagerBasedRLEnv.step` computes rewards/terminations before interval events, a
  mid-step shove of a constrained bulb is visible to scoring before projection
  corrects it; while constrained, the old-bulb distance channels therefore report the
  seat pose, preventing the best-progress latches from paying transient displacement.

## 7. Implementation scope

- `mdp/attach.py` — the `bulb_attachment` event term (FSM + projection), derived
  attachment predicates, and the constrained-aware old-bulb score channels.
- `replace_env_cfg.py` — event wiring with `BAYONET_INSERTION_DEPTH` /
  `BAYONET_ROTATION_ANGLE` (+ sign, tolerances); reward and termination term names
  unchanged, so telemetry/recording contracts are untouched.
- `scene_cfg.py` — docstrings (the old bulb already spawns dynamic).
- `scripts/verify_attach.py` — policy-free verification driving bulb poses with robot
  actions zero (§8).
- Docs (`task_spec.md`, `roadmap.md`, module docstrings).

Out of scope: porting the mechanic to Remove/Install (their bulbs still lift straight
out), and any joint-based (D6) backend.

## 8. Verification

`scripts/verify_attach.py`, policy-free, robot actions zero, driving bulb poses
directly (GPU host required):

1. Old bulb starts attached; pulling it along the axis while locked produces no
   axial displacement.
2. Rotating the locked old bulb to `theta = 0`, then pulling: it travels, and cannot
   be rotated while traveling (applied twist during AXIAL is rejected).
3. Reversing mid-unlock re-locks (theta clamps, no state corruption).
4. Past `insertion_depth` the old bulb is FREE (moves freely under physics).
5. A fresh bulb never engages an occupied socket, nor a misaligned entry.
6. Fresh bulb aligned into the empty socket: engages AXIAL; twist during travel is
   rejected; at full depth, twist through `rotation_angle` → `fresh_bulb_attached`;
   with the old bulb in the crate, the `success` termination fires.
7. `(low, high)` parameter ranges sample per env and re-sample on reset
   (`--check-ranges`; a second `ManagerBasedRLEnv` build hangs Isaac Sim in-process,
   so this check runs as its own invocation).

Plus `scripts/verify_scene.py` across presets (regression) and ruff.

## 9. Acceptance criteria

- The old bulb can be freed only via rotate-then-eject, then carried and dropped in
  the crate → `old_bulb_removed`, `old_bulb_disposed` fire.
- The fresh bulb counts as installed only via insert-then-rotate →
  `fresh_bulb_inserted` and `success` (`attached_replacement_success`) are reachable
  end-to-end in `FIATLUX-Replace-v0`.
- `insertion_depth` / `rotation_angle` randomize per env from ranges with no code
  changes.
- No per-env USD edits at runtime; throughput regression within noise.
- The verification checklist in §8 passes on a GPU host.

## 10. Open questions

- **Tolerance while constrained**: entry uses `radial_tolerance`, but once in AXIAL
  the projection is exact (lateral ≡ 0). Acceptable for v1; a compliant channel
  (project only the excess) is a possible refinement if the hard writes fight the
  solver.
- **Escape velocity**: projection zeroes forbidden velocity components each step, so
  a violent yank cannot accumulate escape speed — but the bulb also cannot be
  "broken out". Accepted: breakage is not part of this task's contract (fragility is
  scored via hand contact force separately).
- **`rotation_sign` handedness** vs the G1's preferred wrist direction — to be chosen
  from teleop/scripted attempts; it is one parameter.

## 11. References

- PhysX Direct-GPU API (D6-only): https://nvidia-omniverse.github.io/PhysX/physx/5.4.0/docs/DirectGPUAPI.html
- Isaac Lab discussions/issues: #4189 (runtime attachment), #883 (startup FixedJoint),
  #3700 (SurfaceGripper CPU-only), #3646 / #2069 (kinematic pose writes broken on GPU)
- PhysX joints & breakage: https://nvidia-omniverse.github.io/PhysX/physx/5.1.0/docs/Joints.html ;
  broken-joint resurrect bug: https://github.com/NVIDIA-Omniverse/PhysX/issues/200
- Factory (SDF screwing, no attach events): https://developer.nvidia.com/blog/advancing-robotic-assembly-with-a-novel-simulation-approach-using-nvidia-isaac/
- OmniGibson `AttachedTo`: https://github.com/StanfordVL/OmniGibson/blob/main/OmniGibson/omnigibson/object_states/attached_to.py
- In-repo: `journal/specs/task-family-unification.md` Phase 4. Earlier revisions of
  this file (git history) hold the original platform investigation, the superseded
  wrist-roll design, and the asset-migration analysis.
