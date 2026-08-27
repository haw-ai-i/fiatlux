# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Success-gate plumbing for the subtask family.

A subtask's success gate is a conjunction, and several need a debounce -- "the object is at rest,
the hand has let go, and it has stayed that way for a second". Two facts constrain how that can be
built:

* ``ManagerBasedRLEnv.step`` computes terminations BEFORE rewards.
* A ``ManagerTermBase`` referenced by two managers is instantiated once per manager. A stateful gate
  in both therefore keeps two counters, reset independently and evaluated at different points in the
  step, which can disagree.

So the debounce lives in exactly ONE place -- the ``success`` termination -- and the reward side
reads that term's result for the step instead of recomputing it. ``success_term_fired`` is the
reward-side reader; it holds no state, so there is no second counter to drift.

A gate written as an ``all_of`` list can be taken apart from outside: a test can assert a conjunct
is present, and :class:`gate_progress` can count how many an episode satisfied. A conjunction
``&``-ed together inside one function body is a single opaque bool and supports neither.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, TerminationTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class sustained(ManagerTermBase):
    """True once ``predicate_fn`` has held continuously for ``seconds``.

    Use ONLY in the termination manager; pair it with :func:`success_term_fired` on the reward side.
    A momentary dip resets the counter, so a state that is only stable while a hand steadies it does
    not pass.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._held = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._held[slice(None) if env_ids is None else env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        predicate_fn: Callable[..., torch.Tensor],
        seconds: float,
        predicate_params: dict | None = None,
    ) -> torch.Tensor:
        now = predicate_fn(env, **(predicate_params or {}))
        self._held = torch.where(now, self._held + 1, torch.zeros_like(self._held))
        return self._held >= max(1, round(seconds / env.step_dt))


def success_term_fired(env: ManagerBasedRLEnv, term_name: str = "success") -> torch.Tensor:
    """The ``success`` termination's flag for this step, as a float reward.

    Terminations compute before rewards, so this reads the decision the termination manager already
    made rather than evaluating the gate a second time. That keeps one definition of success even
    when the gate is stateful, and pays exactly on the terminating step -- once, since the episode
    ends there.
    """
    tm = env.termination_manager
    getter = getattr(tm, "get_term", None)
    if callable(getter):
        try:
            return getter(term_name).float()
        except KeyError:
            pass
    store = getattr(tm, "_term_dones", None)
    if isinstance(store, dict) and term_name in store:
        return store[term_name].float()
    return torch.zeros(env.num_envs, device=env.device)


def all_of(
    env: ManagerBasedRLEnv,
    predicates: Sequence[tuple[Callable[..., torch.Tensor], dict]],
) -> torch.Tensor:
    """Conjunction of stateless predicates, as ``[(fn, params), ...]``.

    Expressing a gate as data rather than prose is what lets a test assert a conjunct is present --
    an omitted one produces a gate that passes vacuously.
    """
    out: torch.Tensor | None = None
    for fn, params in predicates:
        val = fn(env, **(params or {}))
        out = val if out is None else (out & val)
    return torch.ones(env.num_envs, dtype=torch.bool, device=env.device) if out is None else out


def conjuncts_of(predicate_fn: Callable, params: dict | None) -> list[tuple[Callable, dict]]:
    """The gate's conditions as a list, unwrapping ``sustained`` and ``all_of``.

    A gate that is one opaque predicate returns a single-element list; its partial credit then
    collapses to its success flag.
    """
    params = dict(params or {})
    if predicate_fn is sustained or getattr(predicate_fn, "__name__", "") == "sustained":
        return conjuncts_of(params["predicate_fn"], params.get("predicate_params"))
    if predicate_fn is all_of:
        return [(fn, dict(pp or {})) for fn, pp in params.get("predicates", [])]
    return [(predicate_fn, params)]


class gate_progress(ManagerTermBase):
    """Pays the increment of the episode's best normalized gate-conjunct fraction:

        (best_simultaneously_true - true_at_reset) / (total - true_at_reset)

    The episode sum telescopes to that fraction, in [0, 1]; the gate firing forces it to 1.0.

    Normalized against the start state because conjuncts like ``robot_standing`` are true at t=0
    on every subtask -- the same reason ``distance_progress`` divides by the episode's own ``d0``.
    Counted simultaneously because the gate is a conjunction, and best-so-far because the channel
    measures how far the episode got; unlike ``distance_progress`` it is therefore bankable, which
    is safe for a score channel read once per episode rather than a dense per-step gradient.

    A gate whose conjuncts are all true at reset scores 1.0 and means the episode starts solved --
    a start-state defect, caught by the per-subtask start checks.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._at_reset = torch.zeros(env.num_envs, device=env.device)
        self._best = torch.zeros(env.num_envs, device=env.device)

    def _count(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        predicates: Sequence[tuple[Callable, dict]] = self.cfg.params["predicates"]
        total = torch.zeros(env.num_envs, device=env.device)
        for fn, pp in predicates:
            total += fn(env, **(pp or {})).float()
        return total

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._at_reset[ids] = self._count(self._env)[ids]
        self._best[ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv, predicates: Sequence[tuple[Callable, dict]]) -> torch.Tensor:
        n = len(predicates)
        gained = (self._count(env) - self._at_reset).clamp_min(0.0)
        headroom = (n - self._at_reset).clamp_min(1e-6)
        frac = torch.where(self._at_reset >= n, torch.ones_like(gained), gained / headroom).clamp(0.0, 1.0)
        increment = (frac - self._best).clamp_min(0.0)
        self._best = torch.maximum(self._best, frac)
        return increment
