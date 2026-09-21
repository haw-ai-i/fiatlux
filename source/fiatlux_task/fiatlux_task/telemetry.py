# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark-side telemetry: the one place evaluation metrics are defined (issue #16).

This module is the *benchmark's* logging surface, not a policy's, and it is built to be
extended here and nowhere else:

- **Channels** (*what*): :meth:`ScoreLogger.step` turns raw step artifacts (the env, its
  ``extras``, done counts) into named metrics -- the manager score channels
  (``Episode_Reward/<term>``, ``Episode_Termination/<term>``), success, episode length,
  control effort, contact force. Add a metric here and it shows up everywhere at once:
  the live sinks, the final summary, and ``eval.py``'s JSON (which is
  :meth:`ScoreLogger.close`'s return value -- scripts do not aggregate on their own).
- **Sinks** (*where*): the :class:`Sink` protocol (``log`` / ``video`` / ``close``).
  ``WandbSink`` is the shipped backend; a TensorBoard or CSV sink is a new class in
  this file, nothing else changes.
- **Orchestration** (*when*): scripts make three calls -- :meth:`ScoreLogger.from_args`,
  :meth:`ScoreLogger.step` once per env step (a no-op unless episodes finished), and
  :meth:`ScoreLogger.close`.

The channels are defined by the task's own reward/termination managers, so every
submission logs the same names regardless of how the policy was produced. Policy-side
*training* telemetry stays framework-side by design (``scripts/rsl_rl/train.py
--logger wandb``). The ``--wandb*`` CLI flags are declared inline by the scripts (this
package cannot be imported before the AppLauncher starts); keep them in sync with
:meth:`ScoreLogger.from_args`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from .recording import term_flag
from .tasks.manager_based.fiatlux_task.mdp import observations as _obs

if TYPE_CHECKING:
    import torch

# Manager-log channels that constitute the benchmark's score breakdown.
SCORE_CHANNEL_PREFIXES = ("Episode_Reward/", "Episode_Termination/")


class Sink(Protocol):
    """A telemetry backend. Implementations live in this module."""

    def log(self, payload: dict[str, float], step: int) -> None:
        """Emit one batch of named metrics at the given completed-episode count."""
        ...

    def video(self, path: str | Path, caption: str | None) -> None:
        """Attach a rollout video to the run."""
        ...

    def close(self, summary: dict) -> None:
        """Stamp the final aggregate results and release the backend."""
        ...


class WandbSink:
    """Weights & Biases backend (lazy import; ``WANDB_MODE=offline`` works unauthed)."""

    def __init__(
        self,
        *,
        project: str,
        entity: str | None = None,
        run_name: str | None = None,
        config: dict | None = None,
    ):
        try:
            import wandb
        except ModuleNotFoundError as err:  # pragma: no cover - env dependent
            raise ModuleNotFoundError(
                "wandb is required for benchmark telemetry: `uv sync` (it is a project "
                "dependency) or `pip install wandb`. Offline use: WANDB_MODE=offline."
            ) from err
        self._wandb = wandb
        # wandb >= 0.26 lazy-loads its API surface; pyright cannot see the attributes.
        self._run = wandb.init(  # pyright: ignore[reportAttributeAccessIssue]
            project=project, entity=entity, name=run_name, config=config or {}
        )

    def log(self, payload: dict[str, float], step: int) -> None:
        self._run.log(payload, step=step)

    def video(self, path: str | Path, caption: str | None) -> None:
        clip = self._wandb.Video(str(path), caption=caption)  # pyright: ignore[reportAttributeAccessIssue]
        self._run.log({"rollout": clip})

    def close(self, summary: dict) -> None:
        for key, value in summary.items():
            self._run.summary[key] = value
        self._run.finish()


class ScoreLogger:
    """Aggregate the benchmark's evaluation metrics and stream them to sinks.

    One instance per evaluation run. :meth:`step` consumes each env step's raw
    artifacts; :meth:`close` returns the final aggregate results (the dict scripts
    print / write as JSON) after stamping it into every sink. Constructing it with no
    sinks is valid and cheap -- aggregation still runs, nothing is streamed.
    """

    def __init__(self, sinks: list[Sink] | None = None, config: dict | None = None):
        self._sinks = sinks or []
        self._config = config or {}
        self.episodes_done = 0
        self.successes = 0
        self._ep_lengths: list[int] = []
        self._control_efforts: list[float] = []
        self.peak_contact_force = 0.0
        # Read on the first step; undoes the manager's scaling in `gate_progress`.
        self._max_episode_length_s: float | None = None
        self._step_dt: float | None = None
        # Weighted sums of the managers' per-episode term stats (each `log` entry is
        # already averaged over the envs that reset that step, so weight by count).
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        # Optional policy-side diagnostics (see `step`'s policy_info): running means.
        self._policy_sums: dict[str, float] = {}
        self._policy_counts: dict[str, int] = {}

    @classmethod
    def from_args(cls, args_cli, extra_config: dict | None = None) -> ScoreLogger:
        """Build from parsed CLI args (sink-less unless ``--wandb`` was passed)."""
        config = {
            "task": args_cli.task,
            "policy": args_cli.policy,
            "seed": args_cli.seed,
            **(extra_config or {}),
        }
        sinks: list[Sink] = []
        if getattr(args_cli, "wandb", False):
            # getattr throughout: a script may declare --wandb without the companion flags.
            run_name = getattr(args_cli, "wandb_run_name", None) or (
                f"{args_cli.task}-{Path(str(args_cli.policy)).stem}-seed{args_cli.seed}"
            )
            sinks.append(
                WandbSink(
                    project=getattr(args_cli, "wandb_project", "fiatlux"),
                    entity=getattr(args_cli, "wandb_entity", None),
                    run_name=run_name,
                    config=config,
                )
            )
        return cls(sinks, config)

    # ------------------------------------------------------------------ per step
    def step(
        self,
        env,
        extras: dict,
        done: torch.Tensor,
        episode_lengths: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
        max_episodes: int | None = None,
        policy_info: dict[str, float] | None = None,
    ) -> int:
        """Consume one env step; returns how many episodes were newly counted.

        Args:
            env: The (unwrapped) env; used for the success term flag and sensors.
            extras: ``env.step``'s info dict (the manager ``log`` lives here).
            done: Per-env done mask for this step (``terminated | truncated``).
            episode_lengths: Per-env step-in-episode counts *at* this step (optional;
                feeds ``mean_episode_length``).
            actions: This step's actions (optional; feeds ``mean_control_effort``).
            max_episodes: Cap on episodes to count (evaluation protocol budget).
            policy_info: Optional policy-side diagnostics for this step (e.g. a critic
                value estimate). The benchmark treats policies as opaque callables, so
                these are accepted, never required: a policy may refresh an ``info``
                dict attribute per call and scripts forward it. Streamed as running
                means under the ``policy/`` namespace, kept apart from score channels.
        """
        import torch

        if actions is not None:
            self._control_efforts.append(float(torch.mean(torch.sum(actions**2, dim=-1))))
        if policy_info:
            for key, value in policy_info.items():
                self._policy_sums[key] = self._policy_sums.get(key, 0.0) + float(value)
                self._policy_counts[key] = self._policy_counts.get(key, 0) + 1
        if self._max_episode_length_s is None:
            self._max_episode_length_s = float(getattr(env, "max_episode_length_s", 0.0)) or None
            self._step_dt = float(getattr(env, "step_dt", 0.0)) or None
        # BOTH hands, matching the recorded channels and the scorer (issue #89). Watching the
        # right hand alone reported a left-handed crush as clean.
        for sensor_name in ("hand_contact", "left_hand_contact"):
            if sensor_name not in env.scene.sensors:
                continue
            # Force on the OBJECT, matching the recorded contact_force channels and the
            # scorer. The sensor's net force also carries scenery and self-contact.
            forces = _obs.object_contact_forces(env.scene.sensors[sensor_name])
            self.peak_contact_force = max(self.peak_contact_force, float(torch.norm(forces, dim=-1).max()))

        done_ids = torch.nonzero(done, as_tuple=False).flatten()
        if len(done_ids) == 0:
            return 0
        # The env's own `success` termination term is the task-agnostic verdict
        # (valid for the terminating step; survives the in-step auto-reset).
        success = term_flag(env, "success", env.num_envs, env.device)
        counted = 0
        for i in done_ids.tolist():
            if max_episodes is not None and self.episodes_done >= max_episodes:
                break
            self.successes += int(bool(success[i].item()))
            if episode_lengths is not None:
                self._ep_lengths.append(int(episode_lengths[i].item()))
            self.episodes_done += 1
            counted += 1
        if counted == 0:
            return 0

        log = extras.get("log") or {}
        payload: dict[str, float] = {}
        for key, value in log.items():
            if key.startswith(SCORE_CHANNEL_PREFIXES):
                payload[key] = float(value)
                self._sums[key] = self._sums.get(key, 0.0) + float(value) * counted
                self._counts[key] = self._counts.get(key, 0) + counted
        payload["episodes"] = self.episodes_done
        payload["success_rate"] = self.success_rate
        for key in self._policy_sums:
            payload[f"policy/{key}"] = self._policy_sums[key] / self._policy_counts[key]
        for sink in self._sinks:
            sink.log(payload, step=self.episodes_done)
        return counted

    # ------------------------------------------------------------------ results
    @property
    def success_rate(self) -> float:
        return self.successes / max(self.episodes_done, 1)

    @property
    def gate_progress(self) -> float:
        """Mean partial credit per episode, in [0, 1] (``mdp.gates.gate_progress``).

        ``RewardManager`` scales every term by ``weight * dt`` and logs the episode sum divided
        by ``max_episode_length_s``, so the telescoped value the channel is built to carry is
        ``logged * max_episode_length_s / step_dt``. The logged value on its own is not
        comparable across subtasks: their horizons run 20 s to 90 s.
        """
        key = "Episode_Reward/gate_progress"
        if key not in self._sums or not self._max_episode_length_s or not self._step_dt:
            return 0.0
        mean_logged = self._sums[key] / self._counts[key]
        return mean_logged * self._max_episode_length_s / self._step_dt

    def results(self) -> dict:
        """The final aggregate results (also what :meth:`close` stamps into sinks)."""

        def _mean(xs):
            return float(sum(xs) / len(xs)) if xs else 0.0

        results = {
            **self._config,
            "episodes": self.episodes_done,
            "success_rate": self.success_rate,
            "gate_progress": self.gate_progress,
            "mean_episode_length": _mean(self._ep_lengths),
            "mean_control_effort": _mean(self._control_efforts),
            "peak_contact_force": self.peak_contact_force,
            "score_breakdown": {key: self._sums[key] / self._counts[key] for key in sorted(self._sums)},
        }
        if self._policy_sums:
            results["policy_diagnostics"] = {
                key: self._policy_sums[key] / self._policy_counts[key] for key in sorted(self._policy_sums)
            }
        return results

    def video(self, path: str | Path, caption: str | None = None) -> None:
        """Attach a rollout MP4 (e.g. ``record_run.py``'s video) to every sink."""
        for sink in self._sinks:
            sink.video(path, caption)

    def close(self) -> dict:
        """Finalize: stamp the results into every sink and return them."""
        results = self.results()
        for sink in self._sinks:
            sink.close(results)
        return results
