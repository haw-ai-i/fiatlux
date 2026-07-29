# Abstractions for the subtask family

What to factor out, what to leave alone, and why. Companion to `00-foundation.md`.

## The evidence

Six RL task cfgs (`climb`, `descend`, `carry`, `replace`, `install`, `remove`) share **no
inheritance at all** — each declares its own `ActionsCfg`, `ObservationsCfg`, `EventCfg`,
`RewardsCfg`, `TerminationsCfg` from scratch, all named identically, all module-local. Adding 15
more of those is not viable.

But the problem is not volume, it is that the copies have **already drifted**. Only one pair of
blocks in the whole family is byte-identical (`climb`/`descend` `ActionsCfg`). Measured term-set
comparison:

| One concept | Names actually in use | Coverage |
|---|---|---|
| robot fall penalty | `termination_penalty`, `robot_fall` | 2/6 vs 4/6 |
| completion bonus | `success_bonus`, `seated_bonus`, `removed_bonus`, `ladder_ready`, `fresh_bulb_inserted` | five names |
| payload drop termination | `bulb_dropped`, `fresh_bulb_dropped`, `old_bulb_dropped` | three names |
| light randomization | `randomize_light` vs `randomize_sky_intensity` + `randomize_key_light` | two shapes |

Plus outright gaps: `reset_robot_joints` is absent only from Carry; `reset_robot_root` only from
Carry/Install/Remove; `randomize_material_tint` is absent from Descend although
`journal/specs/domain-randomization.md` says all three axes are ON for scaffold envs
(Base/Descend/Remove/Install). The smoothness tail exists in three variants — Climb/Descend/Carry/
Replace carry `com_sway`/`ang_vel_xy`/`ankle_pos_limits`/`joint_deviation_*`, while Install/Remove
instead carry `joint_vel`/`joint_pos_limits` that nobody else has.

**Why the naming drift is a real cost, not an aesthetic one.** `replace_env_cfg`'s own docstring:
"Every channel is its own named reward/termination term, so the per-term episode sums in
`extras['log']` *are* the score breakdown." Two names for one channel means the breakdown does not
aggregate across tasks — you cannot ask "how much did falling cost across the family".

And a live duplication with teeth: Climb and Descend each write their four success thresholds
**twice** — once in `RewardsCfg.success_bonus.params`, once in `TerminationsCfg.success.params`.
Nothing keeps them equal. Drift there produces a reward that pays for a state which does not
terminate.

The codebase already names this work: `climb_env_cfg.py`'s docstring says "A shared RL family base
becomes worthwhile once Descend upgrades to a second RL member". Descend is RL now. The precondition
was met before this issue existed.

## Layer 1 — `SubtaskEnvCfg`: the invariant base

Owns everything that is the same for all fifteen, and owns the **canonical name** for each concept.

- **`ActionsCfg`** — whole-body joint-position targets. One class; 5/6 are already identical.
- **The `policy` observation group — frozen, and never extended by a subclass.** The argument is not
  DRY, it is correctness: the policy group is the *sensor-realizable* mode (IMU, estimated base
  state, proprioception, hand contact, ego RGB, lidar). **Real hardware does not change its sensor
  suite per subtask**, so all fifteen must see the same vector. Consequences worth having: one
  observation space across the family, so a single policy can attempt any subtask and GR00T's
  adapter needs no per-subtask change. This works because the filtered hand-contact channel is
  `force_matrix_w.sum(dim=2)` → `(N, B, 3)`, the same width no matter which objects a subtask
  filters for; only the semantics change.
- **The `privileged` group is the extension point** — per-subtask ground truth, reaches only the
  critic, and rsl_rl's `obs_groups` routing is per-task anyway. Term order sets the flat layout, so
  extending here costs nothing while extending `policy` would silently repack every subtask's
  observation.
- **The shaping reward tail, one canonical name each**: `action_rate`, `joint_acc`,
  `ankle_pos_limits`, `joint_deviation_waist`, `joint_deviation_fingers`, `com_sway`, `ang_vel_xy`,
  `robot_fall`. This is what kills the `termination_penalty`/`robot_fall` split.
- **The termination core**: `time_out`, `fell_below`, `fell_over`, `success` (already 6/6), plus
  **one** `FALL_MIN_HEIGHT`/`FALL_TILT_LIMIT` pair — currently 0.35 (Climb) and 0.40 (Carry) for the
  same robot.
- **The event set, canonical names**: `reset_all`, `reset_robot_joints`, `reset_robot_root`,
  `randomize_sky_intensity`, `randomize_key_light`, `randomize_material_tint`,
  `randomize_hand_material`, plus `randomize_start_state` and `place_payload_in_hand`.
- **Sim/solver block** and **one `disable_randomization()`**, which nulls a declared list of
  randomization terms and never touches `reset_all` (restoring default state is correctness, not
  noise).

### Field declaration order is load-bearing

Event terms run in cfg order, which for a configclass is **field declaration order**, and
inheritance appends subclass fields after the parent's. That is exactly what
`place_payload_in_hand` needs (it must run after `reset_robot_root`), so declare it in the base
after that term. The flip side: a subtask that needs an event to run *before* a base event cannot
get there by adding a field. If that ever comes up, re-declare the whole group rather than fighting
the MRO.

## Layer 2 — one intermediate per mode

Six intermediates, each with ≥2 members and real shared structure:

| Intermediate | Members | Owns |
|---|---|---|
| `NavigateSubtaskCfg` | S01, S03, S08, S10, S12 | `approach_progress`, `arrival_bonus`, the arrival gate shape (radius + facing + speed cap) |
| `GraspSubtaskCfg` | S02, S11 | `reach_progress`, `lift_progress`, contact bootstrap, the lift+hold+sustained gate, fragility penalty |
| `PlaceSubtaskCfg` | S04, S09 | placement progress, release detection, at-rest gate, the longer sustain |
| `ClimbSubtaskCfg` | S05, S13 | `climb_height_progress`, ladder-contact bootstrap, live-ladder-top gate |
| `DescendSubtaskCfg` | S07, S15 | `descend_height_progress`, same, inverted |
| `MateSubtaskCfg` | S06, S14 | seat distance + mating-axis orientation, retained-after-release, fragility |

**This preserves per-task versioning.** Fifteen leaf classes, fifteen modules, fifteen `-v0` ids,
each bumpable — the three-level hierarchy changes none of that. The division of labour is what makes
it safe:

> **Intermediates declare which terms exist and how they are wired. Leaves declare the numbers and
> the predicates.**

Everything routinely retuned — radii, heights, force bounds, weights, episode length, the success
predicate — lives in the leaf, so ordinary tuning never touches shared code. A change to an
intermediate genuinely is a change to all its members and bumps them all, which is the honest
accounting.

Optional simplification, flagged not assumed: `climb_height_progress` and `descend_height_progress`
are two `ManagerTermBase` classes in `mdp/rewards.py` that mirror each other exactly (running max vs
running min). They could be one term with a sign parameter, which would let Climb and Descend share
one intermediate. It touches existing baselines, so it is a separate decision.

## Layer 3 — leaves as data, with a validated contract

The wiring problem to design out: if a leaf tunes the base by reaching into param dicts
(`self.rewards.robot_fall.params["minimum_height"] = x`), a mistyped key is a **silent no-op** and
the parent default quietly persists. That is the same failure shape as every other silent-default bug
on this project.

Instead the base declares a contract the leaf must satisfy, validates it once, and does all the
wiring in one place:

```python
@configclass
class SubtaskEnvCfg(ManagerBasedRLEnvCfg):
    # leaf contract -- ClassVar, so configclass does not treat these as scene/serializable fields
    success_predicate: ClassVar[Callable | None] = None
    progress_distance_fn: ClassVar[Callable | None] = None
    payload: ClassVar[str | None] = None          # "bulb" | "old_bulb" | "ladder" | None

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("success_predicate", "progress_distance_fn"):
            if getattr(self, name) is None:
                raise ValueError(f"{type(self).__name__} must set {name}")
        # ONE wiring site: the success termination and the success bonus cannot drift apart
        self.terminations.success.params["predicate_fn"] = self.success_predicate
        self.rewards.success_bonus.params["predicate_fn"] = self.success_predicate
        self.rewards.approach_progress.params["distance_fn"] = self.progress_distance_fn
```

Two properties this buys. A leaf that forgets a hook **fails at construction** instead of training
against a parent's default. And the `success`-termination / `success_bonus` pair is wired from a
single source, which closes the existing Climb/Descend double-declaration.

`ClassVar` rather than a dataclass field because these are callables and a subtask's identity is its
class, not an instance value — it also sidesteps any question about `configclass`'s `to_dict()`
handling of function objects. Callables inside cfg trees are already normal here (`RewardTermCfg.func`,
`distance_progress`'s `distance_fn` param), but keeping them off the field list costs nothing.
Verify the `ClassVar` behaviour against this Isaac Lab's `configclass` before committing to it; the
fallback is a plain field, which the family already uses for non-entity metadata (`scene_preset`,
`orbit_center`).

## Payload is a parameter, not a mixin

The loaded/unloaded axis is orthogonal to mode: S01 navigate vs S03 navigate-with-ladder, S05 climb
vs S13 climb-with-bulb, S15 descend vs S07 descend-with-bulb. The obvious move is a `PayloadMixin`
and multiple inheritance. **Do not.** `configclass` is dataclass machinery, and multiple inheritance
there means fighting MRO field ordering with defaults — for an axis that is genuinely a parameter,
not a type.

Instead: `payload: ClassVar[str | None]` on the base, with `payload_held`, `payload_dropped` and
`place_payload_in_hand` **declared as fields defaulting to `None`**, populated by the base's
`__post_init__` when `payload` is set. Setting declared fields to `None` is the family's established
idiom (`scene.ladder = None`, `disable_randomization`); adding attributes that were never declared
is not, since manager cfgs are built by walking fields.

## Success gates: a combinator, not a class hierarchy

The highest-value abstraction here, and it is composition rather than inheritance. Every subtask's
success is a conjunction: geometric arrival **and** payload held **and** not fallen **and** not
tipped **and** sustained for *t*. Fifteen hand-written conjunctions is both the largest remaining
duplication and the highest-risk one, because **omitting a conjunct produces a gate that passes
vacuously** — the exact bug behind `fragility:gentle_not_broken` passing at 0.0 N, and the bug that
would make S06 score 100% for a bulb falling out of an inverted socket under gravity.

So: `all_of([...])` over plain predicates, plus `sustained(pred, seconds)`, with the gate expressed
as reviewable data rather than prose. Then it is **enforceable**, which is the point:

`tests/test_subtask_contract.py` (no GPU, no `isaacsim_ci` marker) asserts —

1. every subtask with `payload` set has a payload-held conjunct in its success gate;
2. every subtask whose scene has a ladder has a `not ladder_tipped` conjunct;
3. every leaf's `policy` observation group term names equal the base's exactly (the frozen-group
   rule);
4. every leaf's shaping reward term names equal the base's exactly (no fork regrowth);
5. every leaf sets every required hook (construct all fifteen);
6. the canonical names are the only names — no `robot_fall`/`termination_penalty` split returns.

Design plus enforcement, rather than design and hope.

## What NOT to abstract

- **Scene presets stay plain functions.** `InteractiveSceneCfg` treats *every* dataclass field as a
  scene entity, so preset knobs cannot live on the scene cfg at all. Compose small functions, which
  the family already does (`apply_remove_preset` = `apply_tabletop_preset` + `_add_parts_bin`).
- **No payload polymorphism class.** Ladder (7.25 kg, not fragile, 500 N tripwire) and bulb
  (0.035 kg, 50 N glass / 300 N cap) differ by 200× in mass and entirely in failure mode. A common
  interface leaks immediately. A payload is an entity name, a frozen in-root offset, and two force
  bounds.
- **`mdp/` reward functions stay flat module functions.** They already are, `distance_progress`
  already forbids closures, and the thin-wrapper idiom (`removal_bulb_disposal_distance`) is the
  established way to bind an `asset_cfg`. Nothing to gain.
- **No `SuccessGate` class hierarchy** — see the combinator above. Gates are conjunctions of
  independent conditions; inheritance cannot express that as data, composition can.
- **No action-space hierarchy.** All fifteen use identical whole-body joint-position actions. One
  field in the base.
