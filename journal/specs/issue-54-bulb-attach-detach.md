# Issue #54 — Bulb Attachment/Detachment Mechanics (Phase 4)

Status: DRAFT spec — findings from a three-agent investigation (codebase survey, Isaac Lab
/ PhysX platform research, BEHAVIOR-1K reference research), 2026-07-17. Branch:
`54-bulb-attach-detach-mechanics`. Issue: https://github.com/haw-ai-i/fiatlux/issues/54.

## 1. Problem statement

`FIATLUX-Replace-v0` is physically unsolvable:

- The **old bulb** is spawned with `kinematic_enabled=True`
  (`scene_cfg.py:773-782`, inside `apply_replace_preset`), locking it in world space. No
  policy can remove it, so `old_bulb_removal`, `old_bulb_disposal_progress`,
  `old_bulb_removed`, `old_bulb_disposed`, and therefore `full_replacement_success` /
  the `success` termination are scored-but-unachievable (acknowledged in
  `replace_env_cfg.py:31-34` and `docs/task_spec.md:24-27`).
- The **fresh bulb**'s insertion is a purely geometric check — `bulb_seated`
  (`mdp/rewards.py:112-120`): plug-vs-seat position error < 0.015 m AND orientation error
  < 0.2 rad. No physical bond is ever created; the bulb can be knocked out after
  "success".

The design intent already exists: `journal/specs/task-family-unification.md:170-182`
(Phase 4, DEFERRED) calls for a shared `mdp/attach.py` util that makes/breaks a
`FixedJoint` between bulb and socket seat pose, **gated by alignment + accumulated wrist
roll ("screw")** — explicitly NOT threaded geometry. Per the Phase 3.5 regrouping, the
same mechanic must anchor at the table lamp's seat pose for the bench tasks
(Remove/Install) and the elevated fixture for Replace.

## 2. Current state of the codebase (survey findings)

### Scene & assets

- Bulb = B1K `kfmkwd`, socket/lamp = B1K `ehjsdz` — **this exact pair is a registered
  attachment pair in BEHAVIOR-1K** (`attachment_combinations.json`: `light_bulb-kfmkwd` ↔
  `table_lamp-ehjsdz`), and was manually verified insert/detach-able via the API
  (`journal/syncs/2026.07.11.transcript.md:98`).
- `spawn_b1k_single_body` (`scenes.py:40-48`) deactivates all `meta__*` prims — including
  the B1K **attachment metalinks** whose frames are the natural joint anchor points. Their
  transforms survive as baked constants: `SOCKET_SEAT_OFFSET=(0,0,0.0326)`,
  `BULB_PLUG_OFFSET=(0.0635,0,-0.0225)` (`assets.py:52-53`).
- Everything clones via `{ENV_REGEX_NS}` with `replicate_physics=True`
  (`replace_env_cfg.py:384`); `@clone` spawn funcs author env-0's template **before**
  replication — so anything authored at spawn time replicates for free, while per-env
  runtime USD edits fight the shared physics view.
- Existing pattern for authoring physics schemas at spawn: `_spawn_usd_as_rigid_body`
  (`scene_cfg.py:244-278`); `PhysxSchema` confirmed importable
  (`scripts/omniverse/omniverse_ladder_playground.py:243,257`).

### MDP terms & grasping

- Seat/plug frame helpers: `_seat_point_w`, `_plug_point_w`, `_old_bulb_plug_point_w`
  (`rewards.py:56-67, 317-321`). Success chain: `bulb_seated` → `fresh_bulb_inserted` +
  `full_replacement_success` (= seated AND old bulb within 0.25 m of the crate);
  removal gate: fixture clearance > 0.10 m.
- **No grasp mechanic exists** — grasping is emergent friction (12 Inspire finger DoF,
  torque capped at 2.0 N·m, `g1.py:168`). The `hand_contact` sensor
  (`scene_cfg.py:371-375`) has no `filter_prim_paths_expr` against the bulb; the climb
  task's filtered `ladder_contact` sensor (`scene_cfg.py:518-532`) is the pattern if a
  real bulb-contact channel is wanted.
- Timing hazard: `distance_progress` (`rewards.py:346-395`) captures `d0` at reset; the
  `away_threshold` branch for `old_bulb_removal` assumes the bulb starts seated (d0≈0) —
  only holds if the attach mechanic keeps it seated until deliberately released.
- Isaac Lab pinned at **2.3.2.post1** (`pyproject.toml:18`). No joint/attach code exists
  anywhere in `source/`.

## 3. Reference design — BEHAVIOR-1K / OmniGibson `AttachedTo`

(Read from `omnigibson/object_states/attached_to.py` + `utils/usd_utils.py:create_joint`.)

- **FSM shape (transferable):** per step while unattached: contact scan → for the first
  compatible male(child)/female(parent) metalink pair whose poses satisfy
  **pos_diff < 0.05 m AND orn_diff < 15°**, teleport child so frames coincide,
  `keep_still()` both, create a `UsdPhysics.FixedJoint` (child male link ↔ parent female
  link, `excludeFromArticulation=True`) with `physics:breakForce=5000 N` /
  `physics:breakTorque=10000 N·m` (tunable; their demo uses 500). Detach is PhysX's own
  `JOINT_BREAK` event → delete joint prim → wake bodies.
- **No screwing exists in OmniGibson.** Screw-type and snap-type attachments collapse to
  the same aligned-→-instant-fixed-joint model. The wrist-roll "screw" gate in our
  unification spec is an addition on top, not a port.
- **Mechanism does NOT transfer:** their attach path stops/plays the sim, edits the USD
  stage at runtime, and carries an in-code warning that `create_joint` crashes under
  multi-GPU when triggered from contact callbacks. Built for ~1 CPU-ish env. What
  transfers is the FSM shape and the threshold constants, not the joint plumbing.

## 4. Platform constraints — Isaac Lab / PhysX on GPU (verdicts on the issue's options)

The decisive fact (PhysX 5.x Direct-GPU API docs): **all joints except the D6 joint are
unsupported by the Direct-GPU (tensorized) pipeline.** Consequences:

| Issue option | Verdict | Why |
| --- | --- | --- |
| 1. Breakable joints | **REJECT** | Break events are CPU callbacks; no tensorized per-env break query; breakable fixed/revolute joints not a supported Direct-GPU type; known PhysX bug (#200): re-added broken joints resurrect unbroken → fragile episode resets. Unconfirmed whether breakForce is honored at all under the GPU solver. |
| 2. Runtime joint create/destroy | **REJECT for per-env use** | Works as a one-off via `PhysxSchema.PhysxPhysicsAttachment` + `PhysxAutoAttachmentAPI` (Isaac Lab discussion #4189; avoids the snap-to-init-pose bug of `physx_utils.createJoint`, since USD poses go stale during GPU sim). But it is a USD stage edit + physics re-parse — impractical per-env at RL scale. Fine only for startup/global authoring. |
| 3. Kinematic toggling | **REJECT** | `write_root_pose_to_sim` fails on kinematic bodies mid-sim on GPU (Isaac Lab issues #3646, #2069); `kinematic_enabled` is a spawn-time USD property with no documented per-env runtime tensor toggle. This is exactly the operation that is broken. |

Also rejected: Isaac Sim `SurfaceGripper` (CPU-only as of 5.0/5.1 — kills GPU throughput).

**The two mechanisms that DO work at vectorized-GPU scale:**

- **Factory-style — no attach event at all.** NVIDIA's Factory/AutoMate tasks (nut-bolt
  screwing, insertion) model threading as pure SDF collision + friction on GPU; screwing
  is emergent contact. Physically faithful, zero USD edits, but high sim-tuning cost and
  needs thread-quality collision meshes our B1K assets don't have.
- **Persistent per-env D6 joint with tensor-toggled drives.** Author one D6 joint
  (socket seat frame ↔ bulb plug frame) on the env-0 template at spawn; represent
  attached/detached by raising/zeroing the joint's drive stiffness/damping/maxForce via
  GPU tensors (shape `(num_envs, …)`) — no stage edits, no stop/play, per-env divergence
  for free. The D6 is the one joint type with a GPU constraint shader.

## 5. Proposed design

### Approach A (primary): per-env D6 "virtual screw" joints + explicit FSM

One pre-authored D6 joint per bulb per env (old bulb ↔ socket, fresh bulb ↔ socket),
frames at the baked seat/plug offsets, all 6 axes free (no limits), drives OFF by
default. A new tensorized attach manager (`mdp/attach.py`) owns a per-env boolean state
per joint and flips drive gains:

- **ATTACHED** (old bulb at reset; fresh bulb after successful install): high
  stiffness/damping/maxForce on all 6 drive axes, drive target = seat pose → the bulb is
  held rigidly-ish in the socket, yet remains a dynamic body (rewards/obs unchanged,
  `d0≈0` assumption in `distance_progress` preserved).
- **DETACHED**: all drive gains/maxForce zeroed → joint imposes no constraint; bulb is
  free to carry/dispose.

Gates (per unification spec Phase 4, thresholds seeded from B1K):

- **Detach (unscrew) gate**, old bulb: hand within grasp proximity of the bulb (position
  check, optionally a filtered bulb contact sensor) AND **accumulated wrist roll** past a
  threshold (e.g. ≥ 2π of rolling motion in the unscrew direction while in proximity) →
  zero the drives. PhysX break events are NOT used; the "break" is our own gate.
- **Attach (screw-in) gate**, fresh bulb: `bulb_seated`-style alignment (reuse existing
  0.015 m / 0.2 rad tolerances — tighter than B1K's 5 cm / 15°, keep ours) held for N
  consecutive steps AND accumulated wrist roll → raise the drives. `bulb_seated` success
  then additionally requires the ATTACHED state, closing the "knock it out after
  success" hole.

Implementation shape:

1. **`mdp/attach.py` (new)** — `BulbAttachmentManager` (or event-term pair): per-env
   state tensors, wrist-roll accumulator, gate predicates, drive-gain writes; `reset()`
   restores old-bulb=ATTACHED / fresh-bulb=DETACHED. Exported via `mdp/__init__.py`.
2. **`scene_cfg.py`** — spawn `old_bulb` **dynamic** (drop `kinematic_enabled=True`);
   author both D6 joints on the template at spawn (extend the `_spawn_usd_as_rigid_body`
   pattern or a dedicated spawn func; `PhysxSchema` available). Same change mirrored in
   `apply_remove_preset` / `apply_install_preset` (bench lamp seat pose).
3. **`replace_env_cfg.py`** — wire the manager (EventCfg interval term or custom
   manager, `EventCfg` at 157-186 is the hook point); re-verify `old_bulb_removal`
   `away_threshold`, `old_bulb_dropped`, and disposal terms now that the bulb is
   genuinely free; gate `bulb_seated` on ATTACHED.
4. **`remove_env_cfg.py` / `install_env_cfg.py`** — promote to `ManagerBasedRLEnvCfg`
   with the shared util (their `:35` TODOs), as separate follow-up issues if needed.
5. Optional: filtered bulb contact sensor (clone the `ladder_contact` pattern) if the
   proximity gate proves too weak a grasp proxy.

### Approach B (fallback): boolean attach + per-step pose slaving

If free-axis-zero-drive D6 joints turn out to still constrain or destabilize the solver:
no joints at all; "attached" is a per-env boolean, and the attach manager writes the
seated pose/zero velocity to the (dynamic) bulb every step via `write_root_state_to_sim`
while attached. Same gates, same FSM. Less physical (no compliance, no force feedback
through the fixture), known-working tensor path.

### Out of scope (recorded for the roadmap)

Factory-style SDF thread simulation — the physically-faithful end state, but requires
thread-geometry collision meshes and heavy contact tuning; revisit if/when realism of the
screw interaction itself becomes a benchmark goal.

## 6. Validation spikes (do these first)

1. **Zero-drive D6 is truly free**: 2-env headless scene, bulb + socket + D6 with zeroed
   drives → bulb must fall/behave as unconstrained; then raise gains per-env via tensor →
   one env's bulb holds seated, the other falls. This validates the whole of Approach A.
2. **Template-authored joints replicate** under `replicate_physics=True` and survive
   `reset()` (watch for PhysX issue #200-adjacent resurrect/state bugs — we never
   delete joints, so we expect to dodge it; confirm).
3. **Drive-gain writes are per-env addressable** in Isaac Lab 2.3.2's tensor API for a
   non-articulation D6 between two `RigidObject`s (this is the least-documented part —
   if the manager-based API only exposes articulation joint drives, we may need
   `omni.physics.tensors` views directly).
4. **Wrist-roll accumulator** signal quality: log accumulated roll during scripted/replay
   motion to pick the unscrew threshold.

## 7. Acceptance criteria

- Old bulb: dynamic at reset, held seated; a policy (or scripted motion) that grasps and
  rolls the wrist past threshold frees it; it can then be carried and dropped in the
  crate → `old_bulb_removed`, `old_bulb_disposed` fire.
- Fresh bulb: aligning it to the seat and rolling the wrist engages ATTACHED; `success`
  (`full_replacement_success`) is reachable end-to-end in `FIATLUX-Replace-v0`.
- No per-env USD stage edits at runtime; throughput regression at standard env counts
  within noise.
- `verify_scene` and existing reward/obs tests updated; the
  "scored-but-not-yet-achievable" caveats removed from `replace_env_cfg.py` docstring,
  `docs/task_spec.md`, `docs/roadmap.md`.

## 8. Open questions

- Does Isaac Lab 2.3.2 expose drive-gain tensors for a loose (non-articulation) D6
  between two RigidObjects, or do we drop to raw `omni.physics.tensors`? (Spike 3.)
- Drive target pose while ATTACHED: fixed seat pose vs. current-pose-at-attach (B1K
  teleports child to exact alignment; we can snap via drive target instead — decide
  during spike 1).
- Should detach also have a force-based escape hatch (pull hard enough = B1K's
  5000 N intent) in addition to wrist roll, so non-screw strategies aren't dead ends?
- Unscrew direction sign convention for the accumulated-roll gate (per-hand handedness).

## 9. Key sources

- PhysX Direct-GPU API (D6-only): https://nvidia-omniverse.github.io/PhysX/physx/5.4.0/docs/DirectGPUAPI.html
- Isaac Lab discussions/issues: #4189 (runtime attachment), #883 (startup FixedJoint),
  #3700 (SurfaceGripper CPU-only), #3646 / #2069 (kinematic pose writes broken on GPU)
- PhysX joints & breakage: https://nvidia-omniverse.github.io/PhysX/physx/5.1.0/docs/Joints.html ;
  broken-joint resurrect bug: https://github.com/NVIDIA-Omniverse/PhysX/issues/200
- Factory (SDF screwing, no attach events): https://developer.nvidia.com/blog/advancing-robotic-assembly-with-a-novel-simulation-approach-using-nvidia-isaac/
- OmniGibson `AttachedTo`: https://github.com/StanfordVL/OmniGibson/blob/main/OmniGibson/omnigibson/object_states/attached_to.py
  (+ `utils/usd_utils.py` `create_joint`, `examples/object_states/attachment_demo.py`)
- In-repo: `journal/specs/task-family-unification.md` Phase 4; issue
  https://github.com/haw-ai-i/fiatlux/issues/54
