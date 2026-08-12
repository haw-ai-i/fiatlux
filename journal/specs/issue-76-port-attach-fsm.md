# Port the Bulb Attach/Detach State Machine to Remove-v0 and Install-v0

Issue: https://github.com/haw-ai-i/fiatlux/issues/76
Follows: #54 (`journal/specs/issue-54-bulb-attach-detach.md`)

---

## 1. Problem

`mdp.bulb_attachment` is the bayonet finite state machine delivered by #54. It makes bulb
removal an unscrew and bulb installation an insert-then-rotate, by projecting the bulb onto
mutually exclusive axial and rotational motion channels each step.

On `main` it is wired into `FIATLUX-Replace-v0`. It is not wired into `FIATLUX-Remove-v0` or
`FIATLUX-Install-v0`. In those two tasks the bulb is an ordinary dynamic rigid body: it lifts
straight out of the socket and drops straight back in, and a zero-action rollout can score
Remove's removal channel.

This surfaced on 2026-08-10, when twist-and-pull removal was tested under teleoperation and
reported as not working. The task under test was Remove-v0, which never had the mechanic.

Six subtasks also wire the term — S05, S06, S07, S13, S14 and S15 — but that work lives on the
unmerged branch `origin/subtask-rediscretization` and is not on `main`. Commit `5779501` first
wired it at the mate tier; `1993f13` then moved it up to the whole balance tier, because every
on-ladder subtask starts with a bulb already seated in the inverted fixture and nothing else
keeps it from falling out under gravity.

### Why it was invisible: `bulb` is an unmarked name

`mdp/attach.py` resolves scene entities by hardcoded name, and the names carry different
meanings in different tasks:

| Entity role | `attach.py` reads          | Replace-v0 | Remove-v0               | Install-v0 |
| ----------- | -------------------------- | ---------- | ----------------------- | ---------- |
| socket      | `scene["socket"]` (L118)   | present    | present                 | present    |
| old bulb    | `scene["old_bulb"]` (L178) | present    | **absent**              | **absent** |
| fresh bulb  | `scene["bulb"]` (L179)     | fresh bulb | **the seated old bulb** | fresh bulb |

`scene_cfg.py` declares `socket` (L406) and `bulb` (L422) unconditionally, and
`old_bulb: RigidObjectCfg | None = None` (L444), populated only in the Replace preset (L990).

`old_bulb` is marked: it announces which bulb it is. `bulb` is unmarked — it reads as "the bulb
in this scene" rather than "the fresh one". A single-bulb task therefore has an obvious, wrong
name available, and Remove took it, inheriting the name from Insert's bench preset
(`apply_remove_preset`, scene_cfg L683) rather than from the role.

The code already disagrees with itself. Remove scores this bulb with Replace's `old_bulb_*`
terms, and its own docstring flags them as pointing at "this scene's `bulb` entity instead of
Replace's `old_bulb`". The scoring layer treats the bulb as the old one; only the entity name
says otherwise.

The codebase has also already chosen the word for the other side: `fresh_bulb`,
`fresh_bulb_attached`, `fresh_bulb_dropped`, `fresh_bulb_inserted`, `fresh_bulb_pose` and
`fresh_bulb_progress`, plus `_FRESH` at seven sites. `attach.py` L179 reads
`fresh_bulb: RigidObject = env.scene["bulb"]` — the local is named for the role, and only the
lookup disagrees.

### Why a missing term is silent

`attach.py`'s `_attachment()` (L283) raises when an attach-aware predicate runs on an env with
no term wired. A task using only the raw predicates never calls it, so nothing fires. Remove-v0
is that case: it runs, the bulb behaves as an ordinary rigid body, and the absence of the state
machine is recorded only in a module docstring.

---

## 2. Tasks

### Step 1 — rename entities by placement

The rule is where the preset puts the bulb, not which task it belongs to. Seated in the socket
makes it `old_bulb`; anywhere else makes it `fresh_bulb`.

| Preset                       | Bulb placement                                | Name         |
| ---------------------------- | --------------------------------------------- | ------------ |
| `workshop`                   | on the floor                                  | `fresh_bulb` |
| `tabletop`                   | table, hand height                            | `fresh_bulb` |
| `at_height` (Climb, Descend) | `PARKED_BULB_POSITION`, out of the way        | `fresh_bulb` |
| `position` (Carry)           | **seated in the ceiling socket, kinematic**   | `old_bulb`   |
| `remove`                     | `TABLETOP_SEATED_BULB_POSITION` = socket pose | `old_bulb`   |
| `install`                    | in the crate                                  | `fresh_bulb` |
| `replace`                    | both                                          | both         |

- [ ] `scene_cfg.py` — make both bulb entities optional, following the pattern
      `old_bulb: RigidObjectCfg | None = None` already establishes.
- [ ] `apply_remove_preset` populates `old_bulb` instead of `bulb` (currently L683).
- [ ] `apply_position_preset` (Carry) populates `old_bulb`.
- [ ] Remaining presets populate `fresh_bulb`.
- [ ] Sweep the call sites: `SceneEntityCfg("bulb")` ×15, `scene["bulb"]` ×16, `scene.bulb.` ×11
      — **42 across 12 files, in `source/` and `scripts/` both**:
      `source/`: `attach.py`, `base_env_cfg.py`, `g1_bulb_env_cfg.py`, `install_env_cfg.py`,
      `recording.py`, `remove_env_cfg.py`, `replace_env_cfg.py`, `rewards.py`, `scene_cfg.py`.
      `scripts/`: `verify_attach.py`, `verify_interactions.py`, `verify_scene.py`.

      `scripts/` is easy to miss and breaks the verification itself. `verify_scene.py` L363 and
      L452 index `scene["bulb"]`, and `verify_interactions.py` has eight lookups including
      `cfg.scene.bulb.init_state.pos = TABLETOP_SEATED_BULB_POSITION` (L433) — a *seated* bulb,
      so that script exercises the same configuration Remove does.
- [ ] `remove_env_cfg.py` — repoint `BULB_ENTITY` (L50), which carries its six use sites with it.

> **Not a global find-and-replace.** Remove's and Carry's `bulb` become `old_bulb`; everyone
> else's becomes `fresh_bulb`. A blanket substitution would rename two seated bulbs to
> `fresh_bulb` and re-create the defect this work removes. Do Remove first, in its own commit,
> then sweep the rest.

**Carry is the case to check carefully.** `apply_position_preset` places its bulb at
`fixture_pos`, which is the socket's own pose, and the preset's comment states the intent:
"Seated == the fixture's own pose". The bulb is seated, so it is an old bulb.

That bulb is also kinematic (`kinematic_enabled = True`), because it is scenery rather than
something the robot handles. It must stay outside the state machine. The projection writes a
pose every step, and per-env pose writes to kinematic bodies are broken on GPU — the reason #54
rejected `kinematic_enabled` toggling (Isaac Lab #3646, #2069). The `old_bulb` name is safe
while Carry wires no attachment term, and today it wires none. If that changes, make the bulb
dynamic before wiring the term.

**Recorded output does not change.** `recording.py` L106 looks up `env.scene["bulb"]`, but the
keys it serializes — `bulb_pos`, `bulb_quat`, `bulb_lin_vel` — are independent string literals.
Update the lookup; leave the keys alone, so existing recordings stay readable.

### Step 2 — port the state machine

- [ ] `attach.py` — resolve entity existence from the scene **config**, in `__init__`, before
      `self.reset()` runs. See below for why config rather than the runtime dict.
- [ ] `attach.py` — reset an absent bulb's row to `FREE` regardless of its role. See below.
- [ ] Wire the event term into `remove_env_cfg.py` and `install_env_cfg.py`.
- [ ] Swap Remove's raw predicates for the attach-aware ones.
- [ ] `install_env_cfg.py` — gate seating on `mdp.fresh_bulb_attached`, not geometric proximity.
- [ ] **Swap Remove's two dense `distance_progress` channels too**, not just the sparse
      predicates. `removal_progress` (L165) uses `mdp.removal_bulb_fixture_clearance` and
      `disposal_progress` (L174) uses `mdp.removal_bulb_disposal_distance` — both raw. Replace
      uses the attach-aware `old_bulb_release_clearance` (L267) and
      `old_bulb_disposal_distance_pinned` (L274). Raw dense distances are read pre-projection
      and can latch false progress, which would violate the "no channel fires before the
      projection has run" criterion in §4. After the rename Remove's entity *is* `old_bulb`, so
      Replace's variants may apply directly and the `removal_bulb_*` wrappers may drop out —
      confirm during implementation.
- [ ] Move `BAYONET_INSERTION_DEPTH`, `BAYONET_ROTATION_ANGLE`, `SEAT_POS_THRESHOLD` and
      `SEAT_ORI_THRESHOLD` out of `replace_env_cfg.py` into a shared module. `mate.py` reached
      across with `from ..replace_env_cfg import ...`; two more tasks doing the same is the
      wrong shape.
- [ ] Confirm Remove's seated bulb spawns with the socket's rotation. `reset()` sets
      `_prev_twist = 0`, relying on `attach.py` L159: "the old bulb's init rot IS the fixture
      rot". `apply_remove_preset` sets only position. If the rotation disagrees, `theta`
      integrates a spurious delta on the first step and the bulb starts partly unlocked.
- [ ] Confirm `SOCKET_SEAT_OFFSET`, `BULB_PLUG_OFFSET` and `SOCKET_SEAT_AXIS` — constants in
      `attach.py` measured from the Omniverse `LightBulb` asset — hold for the bench lamp socket
      Remove and Install use. If those tasks use a different socket, the offsets are wrong and
      nothing else here matters.
- [ ] `remove_env_cfg.py` — delete the "Honesty note" docstring paragraph, which stops being true.
- [ ] `replace_env_cfg.py` — no change expected. Verify rather than assume.

**How to test for an absent entity.** `InteractiveScene` offers two workable mechanisms. Use the
config one.

```python
# __init__, before self.reset()
self._has = {
    _OLD:   getattr(env.scene.cfg, "old_bulb",   None) is not None,
    _FRESH: getattr(env.scene.cfg, "fresh_bulb", None) is not None,
}

# reset() — an absent row starts FREE so socket_empty reads true
self._phase[_OLD, ids] = _ROTATING if self._has[_OLD] else _FREE
self._theta[_OLD, ids] = self._angle[ids] if self._has[_OLD] else 0.0

# __call__ — index only what the config declares
if self._has[_OLD]:
    self._advance(env.scene["old_bulb"], _OLD, ...)
```

The alternative, `env.scene.rigid_objects.get("old_bulb")`, works — `rigid_objects` is a public
`dict[str, RigidObject]` — but it fails silently. A misspelled entity name returns `None` and the
bulb is never governed, which is indistinguishable from "this task has no old bulb". That is the
exact failure this work exists to remove, so it should not be reintroduced in the fix.

Reading the config keeps the loud path. `InteractiveScene.__getitem__` raises
`KeyError: Scene entity with key '...' not found. Available Entities: [...]`, naming every entity
the scene does have. The config is also the authoritative statement of intent: if a task declares
an entity that then fails to spawn, this surfaces the failure instead of masking it as absence.

Confirmed against Isaac Lab v2.3.2 source: `InteractiveScene.__init__` sets `self.cfg = cfg`, and
the scene is built before managers load, so `env.scene.cfg` is populated when the event term is
constructed. That is what makes the `__init__`-time check viable and lets the first `reset()` set
an absent row correctly.

This also follows the repo's existing idiom — `events.py` L209, `viz.py` L119, `verify_scene.py`
L276, `record_run.py` L170 and `verify_attach.py` L107 all test optional entities with
`getattr(cfg, name, None)`.

**The predicate swap.** `ManagerBasedRLEnv.step` computes rewards and terminations _before_
interval events, so the projection that enforces the channel runs after scoring. Raw geometry
therefore reads "removed" for a step during which the bulb has not left the channel. #54 §6
flags this, and commit `5779501` performed exactly this swap when wiring S06.

| Remove today                         | Must become                           |
| ------------------------------------ | ------------------------------------- |
| `mdp.old_bulb_removed` (L183)        | `mdp.old_bulb_removed_after_release`  |
| `mdp.old_bulb_disposed` (L189, L236) | `mdp.old_bulb_disposed_after_release` |
| `mdp.old_bulb_dropped` (L200, L240)  | `mdp.old_bulb_dropped_after_release`  |

**Why an absent row must reset to `FREE`.** `socket_empty` for each bulb reads the _other_
bulb's row, so an absent bulb's row still participates in the gate, and `engage` (the only
`FREE → AXIAL` transition) is false whenever `socket_empty` is false.

- **Remove** — the fresh row resets to `FREE` and is never advanced, so `socket_empty` for the
  old bulb reads true. Correct: no fresh bulb can occupy the socket.
- **Install** — the old row resets to `ROTATING` at the full lock angle, so `socket_empty` for
  the fresh bulb reads **false forever** and the fresh bulb can never enter the channel.

The alternative is to make `socket_empty` ignore rows for absent entities. Either is a few
lines; resetting to `FREE` is simpler. Whichever is chosen, Test 4 covers it, because the
symptom — Install silently never seats — is indistinguishable from a physics or grasp problem.

Nothing else in `attach.py` changes: the `(2, num_envs)` tensors (`__init__` L137–148), the
phase transitions (`_advance` L223–248), the projection (`_advance` L250–279) and the
parameter sampling (`_sample_parameter` L81) all stay as they are.

### Step 3 — close the discoverability gap

- [ ] Warn at env init when a task **scores** bulb removal or seating but wires no
      `bulb_attachment` term.

Key the warning on what a task scores, not on what its scene holds. `scene_cfg.py` L422 declares
`bulb` unconditionally, so Climb-v0, Carry-v0 and Descend-v0 each carry a bulb and socket they
never manipulate, and Base-v0 defines no rewards. A scene-contents trigger fires on those four
every run, and a warning that fires where it does not apply gets filtered out. A scoring trigger
fires on Remove-v0 and Install-v0, the two tasks this work changes.

### Out of scope

`subtask_tiers/mate.py` is out of scope while it's not on `main`.

### File overlap with `subtask-rediscretization`

The unmerged branch `origin/subtask-rediscretization` (issue #66) also wires
`mdp.bulb_attachment`, for S06 and S14. Comparing `main...origin/subtask-rediscretization`:

| File                       | This plan          | That branch     |
| -------------------------- | ------------------ | --------------- |
| `mdp/attach.py`            | modified           | **not touched** |
| `remove_env_cfg.py`        | modified           | not touched     |
| `install_env_cfg.py`       | modified           | not touched     |
| `scripts/verify_attach.py` | modified           | not touched     |
| `scene_cfg.py`             | modified           | **modified**    |
| `mdp/rewards.py`           | modified (sweep)   | **modified**    |
| `replace_env_cfg.py`       | expected unchanged | modified        |

That branch does not touch the state machine, and this plan does not change the state machine's
interface, so there is no conflict over `mdp/attach.py`.

That branch also adds `mdp/attach_terms.py`. Check whether it duplicates anything this plan
touches before either merges.

---

## 3. Tests

`verify_attach.py` takes only `--seed`, `--video` and `--check-ranges`. It exercises Replace-v0
alone, so the first item is a prerequisite for the rest.

- [ ] **Add `--task` to `verify_attach.py`.**
- [ ] **Remove-v0, single bulb.** The bulb starts locked; pulling it axially while locked
      produces no displacement; rotating to `theta = 0` then pulling frees it; twist during
      axial travel is rejected.
- [ ] **Install-v0, single bulb.** A bulb aligned into the empty socket engages the axial
      channel; twist during travel is rejected; twist through `rotation_angle` at full depth
      marks it attached.
- [ ] **The absent-row gate, directly.** On Install-v0, assert `socket_empty` reads true for the
      fresh bulb at reset with no old bulb in the scene.
- [ ] **Scoring cannot fire a step early.** Drive Remove's bulb out of the channel and confirm
      `old_bulb_removed_after_release` stays false until the bulb is genuinely released. Raw
      geometry reports success one step sooner; the test must distinguish the two.
- [ ] **Reset integrity.** Remove's bulb is `ROTATING` with `theta == rotation_angle` at reset,
      and `theta` has not drifted after one step. Catches a spawn rotation that disagrees with
      the socket.
- [ ] **Renames are transparent to observations.** Each task's observation group builds and
      reports the same pose after its entity is renamed.
- [ ] **Recordings still parse.** A run recorded before this work loads afterwards.
- [ ] **Zero-action rollout on Remove-v0 scores nothing.** It can today, because the bulb is
      merely resting.
- [ ] **Sweep completeness, over `source/` and `scripts/` both.** No `scene["bulb"]`,
      `SceneEntityCfg("bulb")` or `scene.bulb.` survives. Scoping this to `source/` alone is how
      the first draft of this plan undercounted by 14 sites and three files — all of them in
      `scripts/`, where a miss breaks the verification tooling itself.

---

## 4. How to verify

Requires a GPU host. Nothing below has been executed.

**The host is `MarkLXXXV`**, reachable over Tailscale at `100.119.136.84` (password auth;
credentials are not recorded here). It carries an RTX 3090 with 24 GB and driver 580.126.09 —
the same machine and GPU #54 cites for its own verification, so the 21/21 baseline is
reproducible on it rather than merely inherited.

**Run under the `student2` account.** It holds the toolchain:

| | |
| --- | --- |
| conda | `/home/student2/miniconda3` |
| env | `env_isaaclab` — the name the commands below assume |
| Isaac Lab | `/home/student2/IsaacLab`, tag **v2.3.1** |

Two gaps to close before the commands run:

- **`fiatlux` is not checked out on the host.** Nothing matching it exists under `/home` or
  `/opt`. Clone it under `student2` first.
- **Version skew.** `pyproject.toml` pins `isaaclab[isaacsim,all]==2.3.2.post1`; the host
  checkout is `v2.3.1`. Confirm that gap is harmless, or align it, before treating a 21/21 result
  as comparable to #54's.

The `pasha` account has no conda, Isaac Lab or checkout. The `ssh iolani` route is down — its
`bastion` ProxyJump (`170.9.60.0`) times out during banner exchange — so Tailscale is the working
route today.

```bash
conda activate env_isaaclab && export PYTHONPATH=$PWD/source/fiatlux_task

# regression gate — Replace-v0 must not move
python scripts/verify_attach.py --task FIATLUX-Replace-v0                 # expect 21/21
python scripts/verify_attach.py --task FIATLUX-Replace-v0 --check-ranges  # expect 1/1

# the port itself
python scripts/verify_attach.py --task FIATLUX-Remove-v0
python scripts/verify_attach.py --task FIATLUX-Install-v0

# scene regression across presets
python scripts/verify_scene.py --headless --task FIATLUX-Remove-v0
python scripts/verify_scene.py --headless --task FIATLUX-Install-v0
python scripts/verify_scene.py --headless --task FIATLUX-Replace-v0

# completeness: the unmarked name is gone, not merely reduced
grep -rn 'scene\["bulb"\]\|SceneEntityCfg("bulb")\|scene\.bulb\.' source/ scripts/ && echo FAIL || echo OK
```

What the regression gate does **not** establish: `verify_attach.py` drives bulb poses directly
with robot actions at zero. Passing 21/21 shows the state machine is self-consistent and that
this port did not break it. It does not show that a robot can operate the mechanic — see §5.

### Done when

- `FIATLUX-Remove-v0` requires unscrew-then-eject, and a zero-action rollout no longer scores
  removal.
- `FIATLUX-Install-v0` requires insert-then-rotate before a bulb counts as seated.
- `FIATLUX-Replace-v0` holds at 21/21 and `--check-ranges` 1/1.
- `old_bulb` means "starts seated and locked" and `fresh_bulb` means "starts free", in every
  task, and no entity is named `bulb`.
- Remove and Install score through the attach-aware predicates, so no channel fires before the
  projection has run.
- A task that scores bulb outcomes without the term says so at init, rather than in a docstring.
- The bayonet parameters live in a shared module; no task imports configuration from another
  task's env cfg.
- Recorded output is unchanged and prior recordings still parse.

---

## 5. Open questions inherited from #54

Both concern whether the mechanic can be operated at all under contact-driven control. Neither
is introduced by this port, but they bound its value: porting a mechanic that cannot be operated
spreads the problem to two more tasks.

- **`rotation_sign` handedness is unresolved.** #54 §10 records it as "to be chosen from
  teleop/scripted attempts". It shipped hardcoded at `+1.0` and was never chosen. Because the
  old bulb resets at the clamp ceiling, twisting in the locking direction produces no state
  change and no feedback of any kind. An operator cannot distinguish a wrong twist direction
  from a broken mechanic.
- **Contact-driven twist is untested.** Twist delivered through finger friction against a body
  whose pose is overwritten every environment step has never been exercised. #54 §10 anticipates
  a compliant channel as the refinement if the hard pose writes fight the solver.

Recommendation: settle both on `FIATLUX-Replace-v0` before this port lands, because the answers
may change what gets ported.

---

## 6. References

- `journal/specs/issue-54-bulb-attach-detach.md` — the mechanic and its design rationale.
- `source/.../mdp/attach.py` — the state machine.
- `source/.../remove_env_cfg.py` — the "Honesty note" this work removes.
- `source/.../subtask_tiers/mate.py` on `origin/subtask-rediscretization` — the closest
  precedent for wiring this term into another task (commit `5779501`, subtasks S06 and S14).
  Read it before starting; it is not on `main` and cannot be run from this branch.
- Issue #66, "Re-discretize subtasks on navigation / balance / grasping boundaries".

## 7. Glossary

- **Finite state machine** — the three-phase bulb model in `mdp/attach.py`: `FREE`
  (unconstrained), `AXIAL` (in the insertion channel, axial travel only), `ROTATING` (bottomed
  out in the lock groove, twist only).
- **Subtask tier** — a shared configuration layer grouping subtasks with the same physical
  demands. Four exist on `origin/subtask-rediscretization`: `balance`, `grasp`, `mate`, `place`.
- **Mate tier** — the tier covering S06 (remove the old bulb) and S14 (screw the fresh one in):
  manipulation performed while balancing on the ladder.
