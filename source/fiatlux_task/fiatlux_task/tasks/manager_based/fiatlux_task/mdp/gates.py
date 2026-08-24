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
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, TerminationTermCfg

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
