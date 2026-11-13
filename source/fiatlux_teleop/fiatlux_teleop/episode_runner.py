"""Operator-paced teleop as an EPISODE MACHINE: arm on a button, end on a gate.

The session loop this drives is::

    IDLE (sim live, robot standing, operator input NOT forwarded)
      -- button --> ARMED (input forwarded, bag open, clock running)
                      |-- success gate latches ------> success/
                      |-- a failure predicate fires -> failure/
                      |-- the clock runs out --------> failure/ (reason "timeout")
                                                          |
                      reset to the scene start <----------+
                      back to IDLE

Why the predicates are evaluated HERE rather than by the env
------------------------------------------------------------
A termination in the cfg resets the scene *inside* ``env.step``. That is why
``subtask_teleop`` clears them all and why ``sonic_teleop`` clears ``success`` too: an
auto-reset mid-take yanks the scene away from the operator, and the last recorded bag row
would carry post-reset state. So the terms stay cleared and this module evaluates the same
functions out-of-band -- exactly the trick ``teleop_recording`` already uses for the success
gate. The episode ends because the DRIVER stops it, not because the env reset underneath it.

That also buys the idle window for free: during IDLE the predicates are evaluated and
*reported* but never end anything, so an operator whose bulb was crushed by the staging grip
(issue #198) is told before committing to a take instead of banking an unusable one.

Timeout is counted from the ARM, not from the scene reset, and defaults to the task's own RL
budget so a recorded demonstration is one the evaluation environment would accept. Measured
across 121 VR takes, that costs nothing on seven of the eight recorded subtasks (their
successes run 5.6-18.7 s mean against a 120 s budget); S01-MoveLadder is the exception at
91.2 s mean / 152.5 s max, where it is deliberately binding.
"""

from __future__ import annotations

import inspect
import json
import os
import time
from dataclasses import dataclass, field

__all__ = [
    "EpisodeRunner",
    "FailureWatch",
    "TakeStore",
    "Verdict",
    "failure_terms_of",
]

# A termination that is NOT a failure signal for this loop.
#   success  -- the win condition; the recorder latches it and the driver passes it in
#   time_out -- the env's own horizon, counted from the scene RESET; this loop counts from
#               the ARM instead, so the env's version would fire early and mislabel a take
_NOT_FAILURE = ("success", "time_out")


@dataclass
class Verdict:
    """How an episode ended."""

    outcome: str  # "success" | "failure"
    reason: str  # "success" | "timeout" | the termination term's name
    steps: int
    seconds: float

    @property
    def ok(self) -> bool:
        return self.outcome == "success"

    def as_dict(self) -> dict:
        return {
            "outcome": self.outcome,
            "reason": self.reason,
            "steps": self.steps,
            "seconds": round(self.seconds, 3),
        }


def failure_terms_of(env_cfg) -> dict:
    """``{name: term_cfg}`` for every FAILURE termination the task declares.

    The whole ``DoneTerm`` is kept, not just its ``func``: a stateful term is a
    ``ManagerTermBase`` *class* that Isaac Lab instantiates as ``cls(cfg, env)``, and it needs
    its own cfg to do that. ``payload_lost`` -- S06's only failure exit -- is one of those.

    The teleop cfg has already nulled them by the time anyone can look (``subtask_teleop``
    clears terminations in ``__post_init__``), so the terms are read off a throwaway instance
    of the RL cfg the teleop cfg derives from. That is safe to build: a layout draw constructs
    its own ``random.Random`` from the module-level layout seed rather than consuming a shared
    stream, so a second instantiation cannot perturb the real env's layout.

    Returns an empty mapping -- never raises -- if the cfg is not shaped like a subtask twin;
    the caller then runs with success and timeout only, and says so.
    """
    for base in type(env_cfg).__mro__[1:]:
        try:
            terms = getattr(base(), "terminations", None)
        except Exception:  # noqa: BLE001 - abstract/partial bases in the MRO are expected
            continue
        if terms is None:
            continue
        found = {}
        for name, term in vars(terms).items():
            if name.startswith("_") or name in _NOT_FAILURE or term is None:
                continue
            if getattr(term, "func", None) is None:
                continue
            found[name] = term
        if found:
            return found
    return {}


class FailureWatch:
    """Evaluates failure predicates against the live env, without terminating anything.

    Terms are probed once on the first evaluation and any that raise are dropped loudly: a
    predicate that cannot be read must not masquerade as a silent False, which offline would
    read as "this failure never happened" rather than "we never checked".
    """

    def __init__(self, env, terms: dict, verbose: bool = True):
        self.env = env
        self._terms = dict(terms)
        self._resolved = False
        self._verbose = verbose
        self._stateful: list = []
        self.dropped: list[str] = []

    def _resolve(self) -> None:
        """Prepare each term the way ``TerminationManager`` would, then drop whatever fails.

        Two things the manager does that nothing does for a term the cfg has nulled:
        ``SceneEntityCfg`` params are bound to the scene (without it a sensor-reading predicate
        raises on every step), and a term whose ``func`` is a ``ManagerTermBase`` *class* is
        instantiated as ``cls(cfg, env)`` -- stateful terms like ``payload_lost`` and
        ``payload_struck`` are classes, and calling the class itself just raises.
        """
        live = {}
        for name, term in self._terms.items():
            params = dict(getattr(term, "params", None) or {})
            for value in params.values():
                resolve = getattr(value, "resolve", None)
                if callable(resolve):
                    try:
                        resolve(self.env.scene)
                    except Exception as e:  # noqa: BLE001
                        if self._verbose:
                            print(f"[episode] could not resolve {value!r} for {name}: {e}", flush=True)
            fn = term.func
            try:
                if inspect.isclass(fn):
                    fn = fn(cfg=term, env=self.env)
                    self._stateful.append(fn)
                fn(self.env, **params)
            except Exception as e:  # noqa: BLE001
                self.dropped.append(name)
                if self._verbose:
                    print(f"[episode] WARNING: failure term {name!r} is not evaluable ({e}); NOT watched", flush=True)
                continue
            live[name] = (fn, params)
        self._terms = live
        self._resolved = True
        if self._verbose:
            watched = ", ".join(sorted(live)) or "(none)"
            print(f"[episode] watching failure terms: {watched}", flush=True)

    def reset(self) -> None:
        """Clear per-episode state in the stateful terms, as the manager does on an env reset.

        These instances live outside the TerminationManager, so nothing else clears them --
        without this, ``payload_lost``'s sustain counter and ``payload_struck``'s velocity
        history carry over from the previous take and can fire on the first steps of the next.
        """
        if not self._resolved:
            self._resolve()
        for fn in self._stateful:
            reset = getattr(fn, "reset", None)
            if callable(reset):
                try:
                    reset(env_ids=None)
                except Exception as e:  # noqa: BLE001
                    if self._verbose:
                        print(f"[episode] could not reset {type(fn).__name__}: {e}", flush=True)

    @property
    def names(self) -> list[str]:
        """The terms actually being watched -- resolves first, so this never over-reports."""
        if not self._resolved:
            self._resolve()
        return sorted(self._terms)

    def fired(self) -> list[str]:
        """Names of every failure predicate true on this step (empty when all clear)."""
        if not self._resolved:
            self._resolve()
        out = []
        for name, (fn, params) in self._terms.items():
            try:
                value = fn(self.env, **params)
            except Exception as e:  # noqa: BLE001
                # Hold nothing: a transient read failure is not evidence of failure.
                if self._verbose:
                    print(f"[episode] failure term {name!r} raised this step ({e}); skipped", flush=True)
                continue
            try:
                hit = bool(value.reshape(-1)[0])
            except AttributeError:
                hit = bool(value)
            if hit:
                out.append(name)
        return out


class EpisodeRunner:
    """The IDLE/ARMED state machine. One ``step`` per env step; one ``arm`` per button press."""

    IDLE = "idle"
    ARMED = "armed"

    def __init__(
        self,
        watch: FailureWatch,
        timeout_steps: int,
        step_dt: float = 0.02,
        warn_every_steps: int = 250,
    ):
        self.watch = watch
        self.timeout_steps = int(timeout_steps)
        self.step_dt = float(step_dt)
        self.warn_every = int(warn_every_steps)
        self.state = self.IDLE
        self.steps = 0
        self._idle_steps = 0
        self._warned: dict[str, int] = {}

    # -- state ---------------------------------------------------------------
    @property
    def armed(self) -> bool:
        return self.state == self.ARMED

    @property
    def seconds(self) -> float:
        return self.steps * self.step_dt

    @property
    def remaining_s(self) -> float:
        return max(0.0, (self.timeout_steps - self.steps) * self.step_dt)

    def arm(self) -> None:
        # Stateful terms keep per-episode history; clear it so the previous take's sustain
        # counters cannot fire on the first steps of this one.
        self.watch.reset()
        self.state = self.ARMED
        self.steps = 0
        self._warned.clear()
        print(
            f"[episode] ARMED -- commands live, clock running ({self.timeout_steps} steps / "
            f"{self.timeout_steps * self.step_dt:.0f} s)",
            flush=True,
        )

    def disarm(self) -> None:
        self.state = self.IDLE
        self.steps = 0
        self._idle_steps = 0
        self._warned.clear()

    # -- per-step ------------------------------------------------------------
    def step(self, success_fired: bool) -> Verdict | None:
        """Advance one env step. Returns a :class:`Verdict` on the step the episode ends.

        While IDLE this only *reports* predicates that are already true -- the point of the
        idle window is that the operator finds out before committing, not that the harness
        starts banking zero-length failures while nobody is driving.
        """
        if not self.armed:
            self._idle_steps += 1
            if self.warn_every and self._idle_steps % self.warn_every == 0:
                for name in self.watch.fired():
                    seen = self._warned.get(name, 0)
                    self._warned[name] = seen + 1
                    if seen == 0:
                        print(
                            f"[episode] !! {name} is ALREADY TRUE before the take starts -- "
                            "reset before arming, this take would fail immediately",
                            flush=True,
                        )
            return None

        self.steps += 1
        if success_fired:
            return Verdict("success", "success", self.steps, self.seconds)
        hits = self.watch.fired()
        if hits:
            return Verdict("failure", hits[0], self.steps, self.seconds)
        if self.timeout_steps and self.steps >= self.timeout_steps:
            return Verdict("failure", "timeout", self.steps, self.seconds)
        return None


@dataclass
class TakeStore:
    """Where finished takes land, and the session index that makes them findable.

    Layout -- task above session, so one task's demonstrations are one listing no matter how
    many sittings produced them; the failure *reason* is metadata rather than a directory
    level, because a level per reason gives a sparse tree of mostly-empty folders::

        <root>/<task-id>/<session>/success/take_003_seed1763706024_184233/
        <root>/<task-id>/<session>/failure/take_004_seed257239630_184901/
        <root>/<task-id>/<session>/manifest.jsonl

    The caller owns where a take is written WHILE it records (``sonic_teleop`` keeps its
    ``epNN/`` folders at the session root); this moves the finished folder into ``success/``
    or ``failure/`` on seal. So those two directories only ever hold complete takes, and a
    session killed mid-take leaves its debris at the root where no loader scans. The manifest
    is append-only JSONL for the same reason: a killed session keeps every line it wrote.
    """

    root: str
    task_id: str
    session: str = field(default_factory=lambda: time.strftime("%Y-%m-%d_%H%M%S"))
    operator: str | None = None
    # Set when the caller already owns a session directory -- ``sonic_teleop``'s ``--out`` tree
    # is <captures>/<task>/<hand>/<kind>/<input>/<date>/<time>, which ALREADY puts the task
    # above the session. Re-deriving the path here would nest a second copy of both.
    session_dir_override: str | None = None

    @property
    def session_dir(self) -> str:
        if self.session_dir_override:
            return self.session_dir_override
        return os.path.join(self.root, self.task_id, self.session)

    @property
    def manifest(self) -> str:
        return os.path.join(self.session_dir, "manifest.jsonl")

    def seal(self, take_dir: str, verdict: Verdict, info: dict | None = None) -> str:
        """Move a finished take under ``success/``/``failure/`` and index it. Returns its path."""
        dest_dir = os.path.join(self.session_dir, "success" if verdict.ok else "failure")
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, os.path.basename(take_dir))
        try:
            os.replace(take_dir, dest)
        except OSError as e:
            print(f"[episode] could not move {take_dir} -> {dest} ({e}); left in place", flush=True)
            dest = take_dir
        record = {
            "take": os.path.basename(dest),
            "task": self.task_id,
            "session": self.session,
            "operator": self.operator,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "path": os.path.relpath(dest, self.root),
            **verdict.as_dict(),
        }
        score = (info or {}).get("score") or {}
        if score.get("mean_score") is not None:
            record["score"] = score["mean_score"]
        for key in ("seed", "commit", "benchmark_version"):
            if (info or {}).get(key) is not None:
                record[key] = info[key]
        try:
            with open(os.path.join(dest, "verdict.json"), "w") as fh:
                json.dump(record, fh, indent=2)
        except OSError as e:
            print(f"[episode] could not write verdict.json ({e})", flush=True)
        try:
            os.makedirs(self.session_dir, exist_ok=True)
            with open(self.manifest, "a") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError as e:
            print(f"[episode] could not append to {self.manifest} ({e})", flush=True)
        return dest
