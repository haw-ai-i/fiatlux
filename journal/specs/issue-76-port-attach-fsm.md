# Port the Bulb Attach/Detach State Machine to Remove-v0 and Install-v0

Issue: https://github.com/haw-ai-i/fiatlux/issues/76
Follows: #54 (`journal/specs/issue-54-bulb-attach-detach.md`)

---

## 1. Problem

`mdp.bulb_attachment` is the bayonet finite state machine delivered by #54. It makes bulb
removal an unscrew and bulb installation an insert-then-rotate, by projecting the bulb onto
mutually exclusive axial and rotational motion channels each step.

On `main` it is wired into `FIATLUX-Replace-v0`. It is not wired into `FIATLUX-Remove-v0` or
`FIATLUX-Install-v0`. In those two tasks the bulb is an ordinary dynamic rigid body: it lifts straight out of the socket and drops straight back in, and a zero-action rollout can score Remove's removal channel.

**This was found by reading the code, not by a failing run.** An earlier draft of this spec
attributed the discovery to a 2026-08-10 teleoperation session in which twist-and-pull removal was
reported as not working, and named Remove-v0 as the task under test. That was wrong: the session
was on **Replace-v0**, confirmed with Yujin on 2026-08-12. Replace *has* the mechanic, so that
report describes a different failure — the mechanic being inoperable under contact-driven control
— and is now tracked as issue #77.

The defect described here is real and independently verified: `remove_env_cfg.py`'s `EventCfg`
wires no `bulb_attachment` term, and the module's own docstring states that the bulb "lifts
straight out". But nothing ever observed it in a run, and that is precisely the point. The gap's
only record is a docstring paragraph, which is what Step 3 exists to fix — a motivation that has
to stand on its own rather than lean on a session that turned out to be about another task.

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

Remove scores this bulb with Replace's `old_bulb_*` terms, and its own docstring flags them as
pointing at "this scene's `bulb` entity instead of Replace's `old_bulb`". The scoring layer
treats the bulb as the old one; only the entity name says otherwise.

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

| Preset                       | Bulb placement                                | Name         | Prim path  |
| ---------------------------- | --------------------------------------------- | ------------ | ---------- |
| `workshop`                   | on the floor                                  | `fresh_bulb` | `/Bulb`    |
| `tabletop`                   | table, hand height                            | `fresh_bulb` | `/Bulb`    |
| `at_height` (Climb, Descend) | `PARKED_BULB_POSITION`, out of the way        | `fresh_bulb` | `/Bulb`    |
| `position` (Carry)           | **seated in the ceiling socket, kinematic**   | `old_bulb`   | `/OldBulb` |
| `remove`                     | `TABLETOP_SEATED_BULB_POSITION` = socket pose | `old_bulb`   | `/OldBulb` |
| `install`                    | in the crate                                  | `fresh_bulb` | `/Bulb`    |
| `replace`                    | both                                          | both         | both       |

#### 1a. This is a scene-construction refactor, not a substitution

The class default `bulb` (L422) is load-bearing: **every preset mutates it rather than creating
it.** `apply_workshop_preset` (L496) is a documented no-op precisely because the class default
_is_ the workshop layout. `apply_tabletop_preset` (L523) sets `scene.bulb.init_state.pos`;
`apply_position_preset` (L525) sets pos, rot and `kinematic_enabled`; `apply_at_height_preset`
(L574) sets pos; `apply_replace_preset` (L970) sets pos and then separately _constructs_
`scene.old_bulb`.

Making both entities optional therefore breaks every one of those, and "point each preset at one
of them" is not a sufficient instruction. Two chained presets make this concrete:

```python
def apply_remove_preset(scene):
    apply_tabletop_preset(scene)      # <- would now create fresh_bulb
    _add_parts_bin(scene)
    scene.bulb.init_state.pos = TABLETOP_SEATED_BULB_POSITION
```

If tabletop creates `fresh_bulb`, Remove inherits one it must **clear**, not merely add
`old_bulb` beside — otherwise the scene carries two bulbs and Remove's `socket_empty` gate reads
the wrong row. `apply_install_preset` chains the same way but keeps tabletop's `fresh_bulb`, so
the two diverge and must be written separately.

- [ ] Add a `_make_bulb_cfg(prim_path, pos, rot=None, kinematic=False)` factory in `scene_cfg.py`
      carrying today's L422 spawn block verbatim — the 32-iteration solver, `contact_offset=0.005`,
      `BULB_MASS_KG`, `activate_contact_sensors=True`. Replace's `old_bulb` block (L990) is the
      same spawn config already duplicated once; the factory absorbs both.
- [ ] `scene_cfg.py` — `bulb` field is **deleted**; declare
      `fresh_bulb: RigidObjectCfg | None = None` and keep `old_bulb: RigidObjectCfg | None = None`.
- [ ] Every preset constructs exactly the bulbs it owns, and explicitly `None`s the other: - `apply_workshop_preset` stops being a no-op — it must now build `fresh_bulb` at
      `BULB_POSITION`. This is the change most likely to be forgotten, because the function
      body is currently empty. - `apply_tabletop_preset` builds `fresh_bulb` at `TABLETOP_BULB_POSITION`. - `apply_at_height_preset` builds `fresh_bulb` at `PARKED_BULB_POSITION`. - `apply_position_preset` (Carry) builds `old_bulb` at the fixture pose, kinematic, and
      sets `fresh_bulb = None`. - `apply_remove_preset` chains tabletop, then **clears `fresh_bulb`** and builds `old_bulb`
      at `TABLETOP_SEATED_BULB_POSITION`. - `apply_install_preset` chains tabletop, then repositions `fresh_bulb` to
      `BIN_BULB_POSITION`; `old_bulb` stays `None`. - `apply_replace_preset` builds both.

#### 1b. The prim path and the contact filter move together

`hand_contact` filters `{ENV_REGEX_NS}/Bulb` (L465), and Replace's `old_bulb` compensates by
appending `{ENV_REGEX_NS}/OldBulb` to that list (L1009). If Remove's new `old_bulb` adopts the
`/OldBulb` prim path without the matching filter edit, its hand-contact observation and
compliance penalty stop seeing the bulb the robot is actually manipulating — silently, since a
zero force reads as "not touching".

**Append-per-preset does not work**, because of the same chaining that complicates 1a: Remove
calls tabletop, which would append `/Bulb`, and then clears `fresh_bulb` — leaving a filter entry
for a prim that no longer exists. Derive the list instead of accumulating it.

- [ ] Add `_sync_bulb_contact_filters(scene)`, called **once after preset construction
      completes**, which rebuilds `scene.hand_contact.filter_prim_paths_expr` from whichever of
      `scene.fresh_bulb` / `scene.old_bulb` is non-`None`. Derivation from final state cannot go
      stale; accumulation during chaining can, and silently.
- [ ] Replace's explicit append at L1009 is then redundant and comes out.
- [ ] `verify_scene.py`'s `env0()` prim resolution and any `/Bulb` string in the verification
      scripts follow the same table.

#### 1c. `rewards.py` splits into two families, and one of them already has the answer

The helper layer already forked, and the rename resolves the fork rather than propagating it.

**Fresh family — hardcoded, no `asset_cfg` anywhere.** Three helpers read `env.scene["bulb"]`
directly — `_plug_point_w` (L81), `_bulb_socket_axis_error` (L97) and `_bulb_socket_ori_error`
(L116) — and everything downstream inherits it: `_bulb_socket_pos_error` (L88),
`bulb_fixture_distance` (L125), `object_socket_distance_exp` (L133), `bulb_seated` (L149).

- [ ] Rename the lookup in all three to `fresh_bulb`. No signature changes — every scoring caller
      is a fresh-bulb task. `recording.py` is the one exception; see 1d.

**Old family — already parameterized.** `_old_bulb_plug_point_w` (L388) takes
`asset_cfg: SceneEntityCfg = SceneEntityCfg("old_bulb")`, and its docstring states exactly why:
_"Remove's standalone scene names its single (dynamic) seated bulb `bulb`, so its cfg passes
`asset_cfg=SceneEntityCfg("bulb")` through every function below."_

That plumbing exists **solely to work around the naming defect this issue removes.** After the
rename Remove's entity _is_ `old_bulb`, which is already the default, so every caller uses the
default and the parameter becomes dead weight.

- [ ] Delete the `removal_bulb_fixture_clearance` / `removal_bulb_disposal_distance` wrappers
      (`rewards.py` L510 is the only non-default route into `_old_bulb_plug_point_w`) and the
      `asset_cfg` pass-through they exist to carry. Remove uses `old_bulb_fixture_clearance` and
      `old_bulb_disposal_distance` directly, as Replace does.
- [ ] `remove_env_cfg.py` — `BULB_ENTITY` (L50) is **deleted**, not repointed. Five of its six use
      sites are scoring params (L184, L191, L202, L237, L241) that disappear entirely when the
      attach-aware functions replace them, since those supply the now-default value. **The sixth
      does not:** L115 is `bulb_pose = ObsTerm(func=mdp.root_pose_w, params={"asset_cfg": ...})`,
      and `root_pose_w` (`observations.py` L22) declares `asset_cfg` with no default. That one
      becomes a literal `SceneEntityCfg("old_bulb")`.

This is a net simplification, and the plan should claim it: the rename does not add indirection,
it retires indirection that existed only because the name was wrong.

#### 1d. `recording.py` needs role resolution, not a renamed lookup

`TrajectoryRecorder` is task-generic — `record_run.py` accepts any registered task — but L106
unconditionally reads `env.scene["bulb"]`, and L127–128 call `_bulb_socket_pos_error(env)` and
`_bulb_socket_ori_error(env)`, which after 1c read `fresh_bulb`. Remove and Carry will have no
`fresh_bulb`, so recording either of them raises `KeyError`.

- [ ] Resolve the role once at recorder construction: `fresh_bulb` if the cfg declares it, else
      `old_bulb`, and pass that entity into the error helpers. This is the one place the fresh
      family needs an `asset_cfg` parameter.
- [ ] Leave the serialized keys `bulb_pos`, `bulb_quat`, `bulb_lin_vel` alone. They are
      independent string literals and old recordings must keep parsing.

Note what this means for the acceptance test: "prior recordings still parse" exercises the _read_
path and would pass with the write path fully broken. The test that matters is **recording a
fresh Remove-v0 run**, which is the case that raises today's `KeyError`.

#### 1e. `verify_scene.py` carries role logic the sweep cannot see

Three sites, none of which is a `scene["bulb"]` reference:

- `TRACKED_CANDIDATES` (L126) is a list of bare strings, currently `[..., "bulb", "old_bulb", ...]`.
- `PRESET_PRESENCE` (L131–140) declares present/absent entities per preset and today asserts a
  bulb role for **`replace` only** — every other preset's expectation set is silent about bulbs.
- The NaN check (L363) and the floor check (L452) index `scene["bulb"]` unconditionally, for
  every preset.

- [ ] `TRACKED_CANDIDATES` — drop `"bulb"`, add `"fresh_bulb"`.
- [ ] `PRESET_PRESENCE` — this is where the rename's invariant belongs. Give every preset an
      explicit bulb expectation from the §2 table, so the role rule is asserted rather than
      assumed: `remove` and `carry` expect `old_bulb` present / `fresh_bulb` absent, `install`,
      `tabletop`, `workshop`, `climb` and `descend` the reverse, `replace` both present.
- [ ] L363 and L452 — resolve the present bulb from the cfg and check **each** bulb the scene
      declares, rather than one hardcoded name. Substituting a single literal breaks half the
      presets whichever name is chosen.

#### 1f. `verify_interactions.py` stays `fresh_bulb`

The earlier draft of this plan read L433 —
`cfg.scene.bulb.init_state.pos = TABLETOP_SEATED_BULB_POSITION` — as evidence that the script
"exercises the same configuration Remove does." **It does not.** The harness is built from
`FIATLUX-Insert-v0` (`build_insert_cfg`, L253; called at L426), and L433 teleports _Insert's_
bulb to a seated pose to test contact stability at rest. The scene has no `old_bulb` and no
Remove preset. Renaming it `old_bulb` dereferences `None`.

- [ ] All eight `verify_interactions.py` lookups become `fresh_bulb`.
- [ ] **False positive to avoid:** L437 assigns `cfg.scene.bulb_socket_contact`, a different
      attribute that merely shares the prefix. A `scene\.bulb` prefix substitution mangles it.

#### 1g. Site count, and the gate that misses some of them

The three documented patterns cover 42 sites across 12 files: `SceneEntityCfg("bulb")` ×15,
`scene["bulb"]` ×16, `scene.bulb.` ×11, in `source/` (`attach.py`, `base_env_cfg.py`,
`g1_bulb_env_cfg.py`, `install_env_cfg.py`, `recording.py`, `remove_env_cfg.py`,
`replace_env_cfg.py`, `rewards.py`, `scene_cfg.py`) and `scripts/` (`verify_attach.py`,
`verify_interactions.py`, `verify_scene.py`).

At least three more do not match any of the three patterns:

| Site                   | Why the patterns miss it                                     |
| ---------------------- | ------------------------------------------------------------ |
| `verify_attach.py:115` | `cfg.scene.bulb` — no trailing dot                           |
| `verify_scene.py:126`  | bare `"bulb"` string in `TRACKED_CANDIDATES`                 |
| `attach.py:75`         | comment: `state row of the fresh bulb (scene entity "bulb")` |

- [ ] Extend the completeness gate to `scene\.bulb\b` and to the bare-literal cases, so it cannot
      report `OK` while `verify_scene.py` is still tracking a `bulb` that no longer exists. The
      first draft of this plan undercounted by 14 sites by scoping to `source/`; the gate as
      written repeats the mistake along a different axis.

> **Not a global find-and-replace.** Remove's and Carry's `bulb` become `old_bulb`; everyone
> else's becomes `fresh_bulb`. A blanket substitution renames two seated bulbs to `fresh_bulb`
> and re-creates the defect this work removes.

**Carry is the case to check carefully.** `apply_position_preset` places its bulb at
`fixture_pos`, which is the socket's own pose, and the preset's comment states the intent:
"Seated == the fixture's own pose". The bulb is seated, so it is an old bulb.

That bulb is also kinematic (`kinematic_enabled = True`), because it is scenery rather than
something the robot handles. It must stay outside the state machine. The projection writes a
pose every step, and per-env pose writes to kinematic bodies are broken on GPU — the reason #54
rejected `kinematic_enabled` toggling (Isaac Lab #3646, #2069). The `old_bulb` name is safe
while Carry wires no attachment term, and today it wires none. If that changes, make the bulb
dynamic before wiring the term.

### Step 2 — port the state machine

- [ ] `attach.py` — resolve entity existence from the scene **config**, in `__init__`, before
      `self.reset()` runs. See below for why config rather than the runtime dict.
- [ ] `attach.py` — reset an absent bulb's row to `FREE` regardless of its role. See below.
- [ ] Wire the event term into `remove_env_cfg.py` and `install_env_cfg.py`.
- [ ] Swap Remove's sparse predicates for the attach-aware ones (table below).
- [ ] **Swap Remove's two dense `distance_progress` channels too**, not only the sparse terms
      gated on predicates. Remove labels the two groups itself: _dense_ channels
      (`mdp.distance_progress`) pay normalized progress every step; _sparse_ ones
      (`mdp.completion_bonus`, terminations) pay once when a predicate flips. The dense case is
      the worse failure — `distance_progress` pays **best-progress increments** against a latched
      `self._best` (`rewards.py` L454–464), so a transient pre-projection displacement banks
      credit the projection cannot claw back, whereas a sparse term merely fires a step early.
      #54 added `old_bulb_disposal_distance_pinned` for exactly this reason. `removal_progress`
      (L165) and `disposal_progress` (L174) must move to Replace's attach-aware
      `old_bulb_release_clearance` (L267) and `old_bulb_disposal_distance_pinned` (L274).
- [ ] ⚠️ **Switch Install's `align_orientation` to axis-only alignment IN THIS STEP.** It reads
      full-frame quaternion error, so once the FSM turns Install's bulb the channel pays the
      policy to hold `theta = 0` and resist the quarter turn. #90 tried to fix it ahead of the
      port and reverted: with no FSM wired nothing turns the bulb, while `seated_bonus` and the
      success DoneTerm both gate on `bulb_seated`, whose threshold is full-frame — axis-only
      alignment would pay a bulb at any clock angle in full while success stayed unreachable.
      The reward, the FSM and an attach-aware success move together, or not at all. Use
      `mate_terms.bulb_axis_alignment_tanh`; `verify_attach.py`'s
      `bayonet:axis_alignment_is_twist_invariant` pins why.
- [ ] **Install has three seating outputs, and all three need a decision.** "Gate seating on
      `fresh_bulb_attached`" names none of them individually:

  | Site                     | Term                                               | Action                      |
  | ------------------------ | -------------------------------------------------- | --------------------------- |
  | `install_env_cfg.py:179` | `seated_bonus` → `bulb_seated`                     | → `mdp.fresh_bulb_attached` |
  | `install_env_cfg.py:217` | `success` (DoneTerm) → `bulb_seated`               | → `mdp.fresh_bulb_attached` |
  | `install_env_cfg.py:177` | `seat_position_exp` → `object_socket_distance_exp` | **decide explicitly**       |

  The DoneTerm is the one that defines the task, so it is not optional.

  **The dense channel keeps its geometry. Replace already settled this**, and Install should copy
  it rather than invent a third answer: `replace_env_cfg.py` keeps a raw geometric dense channel
  for the approach (`fresh_bulb_progress`, L255, over `mdp.bulb_fixture_distance`) and pays the
  mechanic _separately_ through a sparse attach-aware bonus (`fresh_bulb_inserted`, L290, over
  `mdp.fresh_bulb_attached`), whose comment states the split outright: "pays only after full
  insertion and bulb rotation, not on transiting the geometric success zone."
  - [ ] Keep `seat_position_exp` geometric, and update its cfg comment to say it now means
        "aligned and inserted" — approach shaping, not seating.
  - [ ] **No additional bonus term.** Converting `seated_bonus` to `fresh_bulb_attached` _is_
        Install's equivalent of `fresh_bulb_inserted`; adding a second `completion_bonus` on the
        same predicate would pay the same event twice, once through the reward and once through
        the termination that fires on the same step.

  **The lock rotation ends up with no dense gradient, in Install and in Replace alike.** Replace
  ships that way today — `fresh_bulb_progress` shapes the approach, `fresh_bulb_inserted` pays
  the completion, and nothing shapes `theta` in between. Install inherits the property by copying
  the model. That is a deliberate consequence, not an oversight of this port, and it cannot be
  fixed here anyway: shaping on rotation requires reading `theta`, which appears in no
  observation (§5). Flagging it so it is not rediscovered as a training mystery.

  **Why Remove's dense channels _do_ become attach-aware, and Install's does not.** Not because
  the projection ignores approach — it does not; `_advance` (L250–279) clamps lateral position,
  orientation and velocity throughout `AXIAL` travel. The narrower true property is what matters
  for reward: motion **along** the seat axis toward the seat is permitted inside the channel and
  is never clawed back, so an approach-distance metric read pre-projection agrees with the
  post-projection state. Departure while locked in `ROTATING` is zeroed outright, so a departure
  metric read pre-projection can latch best-progress credit for displacement that no longer
  exists a step later. Remove's dense channels measure departure; Install's measures approach.

- [ ] Move `BAYONET_INSERTION_DEPTH`, `BAYONET_ROTATION_ANGLE`, `SEAT_POS_THRESHOLD` and
      `SEAT_ORI_THRESHOLD` out of `replace_env_cfg.py` into a shared module. `mate.py` reached
      across with `from ..replace_env_cfg import ...`; two more tasks doing the same is the
      wrong shape. **`scripts/verify_attach.py:70` imports them from `replace_env_cfg` too** and
      must be repointed.
- [ ] Confirm Remove's seated bulb spawns with the socket's rotation. `reset()` sets
      `_prev_twist = 0`, relying on `attach.py` L159: "the old bulb's init rot IS the fixture
      rot". `apply_remove_preset` sets only position. If the rotation disagrees, `theta`
      integrates a spurious delta on the first step and the bulb starts partly unlocked.
- [ ] Confirm `SOCKET_SEAT_OFFSET`, `BULB_PLUG_OFFSET` and `SOCKET_SEAT_AXIS` — measured from the
      Omniverse `LightBulb` asset — hold for the bench lamp socket Remove and Install use. If
      those tasks use a different socket, the offsets are wrong and nothing else here matters.
- [ ] `remove_env_cfg.py` — delete the "Honesty note" docstring paragraph (L12), which stops
      being true.
- [ ] **`replace_env_cfg.py` does change.** The earlier draft said "no change expected"; that is
      false three times over — the fresh-bulb rename touches it, the constants move out of it,
      and its module docstring (L42–44) states that "Remove/Install do not yet gate on
      attachment", which this work falsifies. Related stale prose to sweep: the task catalog in
      `__init__.py` (L23), `rewards.py` (L35), and the obsolete joint TODOs at
      `remove_env_cfg.py:278` and `install_env_cfg.py:262`.

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

**Swapping the function is not enough — the `params` dicts must be rewritten with it**, or every
converted term raises `TypeError: unexpected keyword argument` the first time it is called. Two
rules cover all of them:

- **Drop `asset_cfg` everywhere.** The attach-aware functions resolve entities internally and
  accept no `asset_cfg`. This is the same `BULB_ENTITY` retirement as §1c, seen from the cfg side.
- **Rename the threshold key where it differs.** `old_bulb_disposed` takes `distance_threshold`;
  `old_bulb_disposed_after_release` (`attach.py` L353) takes **`disposal_threshold`**. Affects
  `success_bonus` (L188) and the `success` DoneTerm (L234). `clearance_threshold` and
  `min_height`/`disposal_threshold` already match on the other two.

Install's conversions drop params rather than rename them: `seated_bonus` (L179) and `success`
(L217) both pass `{"pos_threshold": 0.015, "ori_threshold": 0.2}`, and `fresh_bulb_attached`
(`attach.py` L298) takes `env` alone. Both dicts go away entirely.

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

The earlier draft asked for a warning triggered when a task "scores bulb removal or seating but
wires no term", and asserted such a trigger would select Remove-v0 and Install-v0 only. Both
halves are wrong, and the checkbox was not implementable as written.

**There is no env initializer to host it.** Every task registers
`entry_point="isaaclab.envs:ManagerBasedRLEnv"` (`__init__.py` L108–137). No project `Env`
subclass exists, so there is no universal init hook.

**Inference from score functions cannot work.** Score semantics are arbitrary callables, and the
relevant ones hide inside `predicate_fn` and `distance_fn` params. Worse, any rule broad enough
to catch raw-predicate Remove also catches **Insert-v0**, which scores `object_socket_distance_exp`
(`g1_bulb_env_cfg.py` L229) and `bulb_seated` (L241, L282) and deliberately has no bayonet FSM.
"Scores seating" does not distinguish "should have the mechanic" from "is a simpler task".

Replace it with two checks, both cheap, both honest about what they catch:

**There is also no shared RL base cfg to hang a field on.** `FamilyBaseEnvCfg`
(`base_env_cfg.py` L210) subclasses the non-RL `ManagerBasedEnvCfg`, and `RemoveEnvCfg` (L253),
`InstallEnvCfg` (L237) and `ReplaceEnvCfg` (L407) each subclass `ManagerBasedRLEnvCfg` directly.
So the check ships as a free function, called explicitly — not inherited.

- [ ] **A shared validator.** `validate_attachment_wiring(cfg)` in the `mdp` package, called from
      the `__post_init__` of `RemoveEnvCfg`, `InstallEnvCfg` and `ReplaceEnvCfg`. Introducing a
      shared RL base for three classes is the larger change and is not required here; if one is
      introduced later for other reasons, the call moves into it unchanged.
- [ ] **A declared contract.** `requires_bulb_attachment: bool = False` as a field on those three
      concrete cfgs, set true on all three. If true, `cfg.events` must carry a `bulb_attachment`
      term, else raise. This catches the real regression risk — someone drops or renames the event
      term while the scoring stays. It does **not** catch a brand-new task that forgets both the
      flag and the term; nothing mechanical can, because that is a statement of intent.
- [ ] **A mechanical rule for the attach-aware terms.** If any score term resolves to a function
      that reaches `_attachment()`, the term must be wired. **Mark the functions, do not list
      them elsewhere.** `attach.py` has no `__all__` — nothing in `mdp/` does; public names are
      hand-imported by `mdp/__init__.py` L43 — so there is nothing to derive from, and a list in
      the validator's own file is exactly the enumeration that goes stale on the ninth function.
      Add a one-line `@requires_attachment` decorator in `attach.py` that sets an attribute on
      the function, decorate the eight functions below `_attachment` (L283), and have the
      validator test the attribute. Several of them reach `_attachment()` only transitively, so
      the marker is also the only honest record of which ones do.
- [ ] Back it with a test that walks `attach.py`'s module namespace and asserts every public
      function whose body reaches `_attachment()` carries the marker. Otherwise the decorator is
      just a prettier enumeration.
- [ ] The validator must traverse the **nested** callables, not just each term's `func`: the
      attach-aware functions are usually passed as `predicate_fn` or `distance_fn` params, which
      is exactly where a `func`-only scan would miss them.

This converts today's runtime `RuntimeError` into an init-time failure naming the task, and it
correctly stays silent on Insert.

Scene contents remain the wrong trigger, for the original reason: after Step 1, Climb, Descend
and Carry still carry an untouched bulb they never manipulate, and Base defines no rewards.

### Out of scope

`subtask_tiers/mate.py` is out of scope while it's not on `main`.

### File overlap with `subtask-rediscretization`

The unmerged branch `origin/subtask-rediscretization` (issue #66) also wires
`mdp.bulb_attachment`, for S06 and S14. Comparing `main...origin/subtask-rediscretization`:

| File                       | This plan        | That branch     |
| -------------------------- | ---------------- | --------------- |
| `mdp/attach.py`            | modified         | **not touched** |
| `remove_env_cfg.py`        | modified         | not touched     |
| `install_env_cfg.py`       | modified         | not touched     |
| `scripts/verify_attach.py` | modified         | not touched     |
| `scene_cfg.py`             | modified         | **modified**    |
| `mdp/rewards.py`           | modified (sweep) | **modified**    |
| `replace_env_cfg.py`       | **modified**     | **modified**    |

That branch does not touch the state machine, and this plan does not change the state machine's
interface, so there is no conflict over `mdp/attach.py`. `scene_cfg.py`, `rewards.py` and
`replace_env_cfg.py` all now conflict on both sides; `scene_cfg.py` is the sharp one, because
this plan restructures how presets construct bulbs.

That branch also adds `mdp/attach_terms.py`. Check whether it duplicates anything this plan
touches before either merges.

---

## 3. Tests

`verify_attach.py` takes only `--seed`, `--video` and `--check-ranges`, and exercises Replace-v0
alone. Adding `--task` is a prerequisite for the rest — but it is **not sufficient on its own**.
The harness hardcodes Replace in `build_replace_cfg` (L94) and `make_env` (L126), assumes both
bulbs exist at L115 (`for bulb_cfg in (cfg.scene.bulb, cfg.scene.old_bulb)`), fetches both at
runtime at L298, and imports the bayonet constants from `replace_env_cfg` at L70.

- [ ] **Make `verify_attach.py` task-generic.** `--task`, plus per-task scenario branches, plus
      optional-entity handling everywhere it currently assumes a two-bulb scene.
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
- [ ] **Every preset declares exactly the roles the §2 table prescribes**, no more and no fewer.
      Assert it through `PRESET_PRESENCE` for all eight presets — not just the three tasks under
      test. This is the check that catches Remove inheriting a `fresh_bulb` from its tabletop
      chain, which is otherwise invisible until `socket_empty` reads the wrong row.
- [ ] **Hand contact still sees the manipulated bulb.** For Remove and Replace, a grasp produces
      nonzero `hand_contact` force on the bulb being handled, and the filter list contains no prim
      the scene does not declare. Catches both the prim-path mismatch and the stale-filter case
      from Remove's preset chain, each otherwise silent.
- [ ] **Renames are transparent to observations.** Each task's observation group builds and
      reports the same pose after its entity is renamed.
- [ ] **Recording a fresh Remove-v0 run succeeds.** This is the write path, and it is the one
      that raises `KeyError` today under the naive rename. Recording Install-v0 and Carry-v0 too.
- [ ] **Prior recordings still parse.** A run recorded before this work loads afterwards.
- [ ] **Zero-action rollout on Remove-v0 scores nothing.** It can today, because the bulb is
      merely resting.
- [ ] **Sweep completeness, over `source/` and `scripts/` both**, including the patterns the
      three documented ones miss (§1g). No `scene["bulb"]`, `SceneEntityCfg("bulb")`,
      `scene.bulb` or bare `"bulb"` entity literal survives. Scoping this to `source/` alone is
      how the first draft undercounted by 14 sites and three files — all of them in `scripts/`,
      where a miss breaks the verification tooling itself.

---

## 4. How to verify

Requires a GPU host. Nothing below has been executed.

**The host is `MarkLXXXV`**, reachable over Tailscale at `100.119.136.84` (password auth;
credentials are not recorded here). It carries an RTX 3090 with 24 GB — the same machine and GPU
#54 cites for its own verification, so the 21/21 baseline is reproducible on it rather than
merely inherited. The `ssh iolani` route is down (its `bastion` ProxyJump at `170.9.60.0` times
out during banner exchange), so Tailscale is the working route.

**Run under the `student2` account.** It holds the toolchain; the `pasha` account has no conda,
Isaac Lab or checkout.

|           |                                                      |
| --------- | ---------------------------------------------------- |
| conda     | `/home/student2/miniconda3`                          |
| env       | `env_isaaclab` — the name the commands below assume  |
| Isaac Lab | `/home/student2/IsaacLab`, tag **v2.3.1**            |
| repo      | `~/fiatlux`, assets in place (728 MB, 169 USD files) |

The host cannot reach GitHub and has no `gsutil`, so the repo arrived as a `git bundle` and the
assets by rsync — neither depends on credentials the host lacks. torch 2.7.0+cu128, CUDA
available, `fiatlux_task` imports.

**Version skew.** `pyproject.toml` pins `isaaclab[isaacsim,all]==2.3.2.post1`; the host checkout
is `v2.3.1`. **Run the regression gate on unmodified `main` first.** If 21/21 does not reproduce
there, the gate is meaningless for this work — and that is a finding in its own right.

```bash
conda activate env_isaaclab && export PYTHONPATH=$PWD/source/fiatlux_task

# regression gate — Replace-v0 must not move
python scripts/verify_attach.py --task FIATLUX-Replace-v0                 # expect 21/21
python scripts/verify_attach.py --task FIATLUX-Replace-v0 --check-ranges  # expect 1/1

# the port itself
python scripts/verify_attach.py --task FIATLUX-Remove-v0
python scripts/verify_attach.py --task FIATLUX-Install-v0

# scene regression — every preset the rename touches, not only the three tasks under test.
# --enable_cameras is required by Insert (wrist camera) and Replace (torso camera), per
# verify_scene.py's own docstring, and is harmless on the rest.
for t in Base Insert Carry Climb Descend Remove Install Replace; do
  python scripts/verify_scene.py --headless --enable_cameras --task FIATLUX-$t-v0
done

# completeness: the unmarked name is gone, not merely reduced.
# 45 matches on main today; zero when the sweep is done.
if rg -n --glob '*.py' --glob '!scripts/omniverse/**' \
     "scene\[['\"]bulb['\"]\]|SceneEntityCfg\(['\"]bulb['\"]\)|scene\.bulb\b|['\"]bulb['\"]" \
     source scripts; then
  echo "FAIL: unmarked 'bulb' survives"; exit 1
else
  echo OK
fi
```

Two things the earlier draft's `grep` version got wrong, both worth keeping in mind if the gate is
ever rewritten: `grep -rn ... scripts/` emits paths with a doubled slash (`scripts//omniverse/...`),
so a `grep -v 'scripts/omniverse/'` filter silently fails to exclude anything; and `&& echo FAIL ||
echo OK` inverts the shell's exit status, so a clean tree exits non-zero and cannot be used as a CI
gate. `rg`'s glob excludes and an explicit `if` avoid both. The character class on the quotes is
the third correction: a `"bulb"`-only pattern silently passes a tree containing
`SceneEntityCfg('bulb')`. The `scripts/omniverse/` exclusion is real and not a workaround — those
are asset-authoring scripts where `"bulb"` names a USD half, not a scene entity.

The scene regression must cover all eight presets. Step 1 restructures how _every_ preset
constructs its bulb, so running only Remove, Install and Replace leaves Base, Insert, Carry,
Climb and Descend unverified — and `apply_workshop_preset`, the preset that changes most (empty
function to bulb constructor), is exercised by none of the three.

What the regression gate does **not** establish: `verify_attach.py` drives bulb poses directly
with robot actions at zero. Passing 21/21 shows the state machine is self-consistent and that
this port did not break it. It does not show that a robot can operate the mechanic — see §5.

### Done when

- `FIATLUX-Remove-v0` requires unscrew-then-eject, and a zero-action rollout no longer scores
  removal.
- `FIATLUX-Install-v0` requires insert-then-rotate before a bulb counts as seated, in the
  `success` termination and not only in a reward.
- `FIATLUX-Replace-v0` holds at 21/21 and `--check-ranges` 1/1.
- `old_bulb` means "starts seated in the socket" and `fresh_bulb` means "starts free", in every
  task, and no entity is named `bulb`. (Seated, not _locked_: Carry's `old_bulb` is kinematic
  scenery and deliberately stays outside the state machine, so it is never in `ROTATING`.)
- Every preset declares exactly the bulb roles the §2 table prescribes — no more, no fewer —
  asserted in `PRESET_PRESENCE` for all eight.
- `hand_contact`'s filter list is derived from the scene's final bulb set, so it cannot carry a
  prim no preset created.
- Remove and Install score their **completion** channels through the attach-aware predicates, so
  no success or best-progress-latching channel fires before the projection has run. Install's
  approach shaping stays geometric by design (§2 Step 2), as Replace's does.
- The `removal_bulb_*` wrappers are gone, and `BULB_ENTITY`'s five scoring uses are gone rather
  than repointed; only its observation use survives, as a literal `SceneEntityCfg("old_bulb")`.
- A task that declares `requires_bulb_attachment` without wiring the term fails at init, and so
  does one that scores an attach-aware function without it.
- The bayonet parameters live in a shared module; no task imports configuration from another
  task's env cfg, and neither does `verify_attach.py`.
- Recording works for Remove, Install and Carry; recorded keys are unchanged and prior
  recordings still parse.

---

## 5. Blocked by #77

The two questions #54 left open are no longer hypothetical. Yujin's 2026-08-12 report establishes
that **a human cannot operate this mechanic on the one task that has it**: on Replace-v0, rotating
the old bulb does not release it. Pulling without rotating does not either, but that half is the
lock working exactly as designed — the old bulb resets into `ROTATING`, where the projection
(L250–279) rewrites it to the seat pose with `proj_lin = 0` every step, so no axial pull can move
it. The unlock rotation is what fails.

Three candidate causes, indistinguishable today:

- **`rotation_sign` handedness was never chosen.** #54 §10 records it as "to be chosen from
  teleop/scripted attempts"; it shipped hardcoded at `1.0` (`replace_env_cfg.py` L186). Worse than
  a coin flip, because the wrong face is silent: the old bulb resets at the clamp ceiling, so
  twisting further into the lock changes nothing, and `at_lock_stop` (L271–272) additionally zeroes
  the angular velocity in that direction. One of the two choices leaves the bulb wholly inert.
- **Contact-driven twist may not work at all.** `verify_attach.py`'s `bayonet:bulb_rotation_unlocks_old`
  (L430) proves the unlock logic under *direct pose driving* with robot actions at zero. Finger
  friction against a body whose pose is overwritten every step has never been exercised. #54 §10
  anticipates a compliant channel if the hard pose writes fight the solver.
- **Scene edits.** Yujin repositioned the bulb and socket. If the socket's rotation changed without
  the bulb's matching it, `reset()`'s premise that "the old bulb's init rot IS the fixture rot"
  (L159) breaks and `theta` integrates a spurious first-step delta.

None can be told apart because **`_phase` and `_theta` appear in no observation, telemetry or
recording path.** #77 covers the diagnosis and that observability.

**This is a hard dependency on #77, not a caution.** Porting a mechanic that cannot be operated
spreads the problem to two more tasks. #77 also delivers the `theta` readout §3's own tests
require — reset integrity, the absent-row gate and no-early-scoring all have to read it.

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
