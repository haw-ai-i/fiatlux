# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trajectory recorder: the experiment "bag" for the Fiatlux benchmark.

There is no ROS in this project, so a "bag" here is a self-describing trajectory
file plus a JSON metadata sidecar. The recorder taps the *raw* environment every
step and stores everything needed to re-derive any score offline -- so scoring is
a separate, replayable step (see ``scripts/score.py``) and never has to re-run the
simulator.

Design:
- ``TrajectoryRecorder.record_step(...)`` is called once per ``env.step``; it
  buffers per-env arrays on the host.
- Given ``out_dir`` at construction, each episode is written to the bag as it ends and
  its steps leave the buffer, so RAM holds only in-flight episodes. Without it the whole
  run stays buffered until ``write``.
- ``write(out_dir, fmt)`` closes the bag (robomimic-style ``data/demo_<i>`` groups, split
  on the ``done`` flag) as HDF5 (default; ``h5py`` ships with Isaac Lab) or a flat ``.npz``
  fallback, plus a human-readable ``meta.json`` header. ``.npz`` has no append mode, so it
  is always fully buffered.

Terminal-step caveat: Isaac Lab's ``ManagerBasedRLEnv.step`` auto-resets episodes
that finished, so object poses read *after* a terminating step reflect the *next*
episode. We therefore capture success/drop from the **termination manager term
flags** (valid for the terminating step) rather than recomputing them from the
(already reset) poses. Continuous pose errors are still logged every step; the
scorer uses the per-episode minimum, which is robust to the reset jump.
"""

from __future__ import annotations

import datetime
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from .grasp_poses import GLASS_CONTACT_LIMIT_N
from .tasks.manager_based.fiatlux_task.mdp import attach as _attach
from .tasks.manager_based.fiatlux_task.mdp import observations as _obs
from .tasks.manager_based.fiatlux_task.mdp import rewards as _rewards

RECORDER_VERSION = "1.0"

# Fallback drop floor for a bulb no task rule covers (issue #202).
DEFAULT_BULB_DROP_FLOOR_M = 0.4


def _benchmark_version() -> str:
    """Read the benchmark version from the extension manifest (not a git SHA)."""
    manifest = Path(__file__).parents[1] / "config" / "extension.toml"
    try:
        try:
            import tomllib  # py311+
        except ModuleNotFoundError:  # pragma: no cover - older interpreters
            import tomli as tomllib  # type: ignore
        with open(manifest, "rb") as fh:
            return tomllib.load(fh)["package"]["version"]
    except Exception:
        return "unknown"


def _git_commit() -> dict[str, object]:
    """The checkout this bag was recorded from (issue #181).

    ``dirty`` matters as much as the SHA: a bag recorded from a modified tree cannot be
    reproduced by checking that SHA out.
    """
    repo = Path(__file__).resolve().parents[3]

    def run(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=5, check=False
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    unknown = {"commit": "unknown", "commit_dirty": None, "commit_branch": None}
    # One call for toplevel/HEAD/branch (rev-parse prints one line per positional arg,
    # in order) instead of three, so a slow git (network fs, lock contention) only
    # costs one 5s timeout instead of three.
    rev_info = run("rev-parse", "--show-toplevel", "HEAD", "--abbrev-ref", "HEAD")
    lines = rev_info.splitlines() if rev_info is not None else []
    if len(lines) != 3 or lines[0] != str(repo):
        return unknown
    _, commit, branch = lines
    status = run("status", "--porcelain")
    return {
        "commit": commit,
        "commit_dirty": None if status is None else bool(status),
        "commit_branch": None if branch == "HEAD" else branch,
    }


def _np(t: torch.Tensor | np.ndarray) -> np.ndarray:
    if hasattr(t, "detach"):
        arr = t.detach().to("cpu").numpy()
    else:
        arr = np.asarray(t)
    return arr.copy()


def _disambiguate_names(names: list[str]) -> list[str]:
    """Ensure conjunct names are unique across a gate's predicates."""
    total: dict[str, int] = {}
    for n in names:
        total[n] = total.get(n, 0) + 1
    counts: dict[str, int] = {}
    unique: list[str] = []
    for n in names:
        if total[n] == 1:
            unique.append(n)
        else:
            idx = counts.get(n, 0)
            counts[n] = idx + 1
            unique.append(f"{n}_{idx}")
    return unique


def _tracked_object_names(env) -> list[str]:
    """Every rigid object worth recording, or the cfg's curated ``record_objects`` list.

    A curated name is kept when the entity it resolves to has a root pose, which is what the
    columns here are: ``root_pos_w``, ``root_quat_w`` and the two velocities. Articulations serve
    those as well as rigid objects do. Contact sensors, ``AssetBaseCfg`` props (an
    ``XformPrimView`` under ``scene.extras``, with no ``.data``) and object collections do not,
    and are skipped with a warning -- a silently missing column reads offline as an object that
    never moved.

    The default stays every RIGID object. Sweeping in the articulations would record the robot
    here as well, whose root state and joints already have their own columns.
    """
    declared = getattr(getattr(env, "cfg", None), "record_objects", None)
    if declared is None:
        return list(getattr(env.scene, "rigid_objects", {}).keys())

    tracked, skipped = [], []
    for name in declared:
        try:
            entity = env.scene[name]
        except KeyError:
            skipped.append(f"{name} (not in the scene)")
            continue
        if hasattr(getattr(entity, "data", None), "root_pos_w"):
            tracked.append(name)
        else:
            skipped.append(f"{name} ({type(entity).__name__} has no root pose)")
    if skipped:
        print(f"[recording] WARNING: record_objects entries not recorded: {', '.join(skipped)}", flush=True)
    return tracked


# Each of these has a named column of its own: hand_contact as ``contact_force``,
# left_hand_contact as ``contact_force_left`` (#89), grip_contact as ``grip_force`` (#106).
_NAMED_CONTACT_SENSORS = ("hand_contact", "left_hand_contact", "grip_contact")


def _extra_contact_names(env) -> list[str]:
    """Every contact sensor that has no named column of its own, discovered rather than listed.

    ``release_contact`` and ``grasp_contact`` are read by the disposal and grasp success
    conditions; their columns are what says which condition stayed unmet.
    """
    sensors = getattr(env.scene, "sensors", {})
    return [s for s in sensors if s.endswith("_contact") and s not in _NAMED_CONTACT_SENSORS]


def _gym_task_id(env) -> str | None:
    """The registered env id (``FIATLUX-S06-DisposeBulb-v0``), which keys the subtask weights."""
    spec = getattr(env, "spec", None)
    return getattr(spec, "id", None)


def term_flag(env, name: str, n: int, device) -> torch.Tensor:
    """Read a termination term flag for the current step (False vector if absent).

    The term flags reflect the step that triggered the done and survive the
    in-``step`` auto-reset, so this is the correct source for "did this episode end in
    success / a drop" *when the task names its drop term* ``bulb_dropped``. It does not
    for ``FIATLUX-Replace-v0``, which names two, ``fresh_bulb_dropped`` / ``old_bulb_dropped``
    (issue #134): the recorded ``dropped_term`` column is an all-False vector for every
    Replace-v0 bag, and ``score.py``'s separate per-bulb-position height check is what
    actually catches drops there, not this column. We try the public accessor first, then
    the manager's internal store for cross-version robustness. Shared by
    :class:`TrajectoryRecorder` (and its ``fiatlux_teleop`` subclass), which write the
    ``success``/``dropped``/``timeout`` term columns into the bag, and by
    ``telemetry.ScoreLogger.step``, which reads the ``success`` term through it for the
    live ``success_rate``.
    """
    tm = env.termination_manager
    getter = getattr(tm, "get_term", None)
    if callable(getter):
        try:
            return getter(name)
        except KeyError:
            pass
    store = getattr(tm, "_term_dones", None)
    if isinstance(store, dict) and name in store:
        return store[name]
    return torch.zeros(n, dtype=torch.bool, device=device)


class _EpisodeSink:
    """Append finished episodes to an open HDF5 bag, one group per episode."""

    def __init__(self, out_dir: Path):
        import h5py

        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = out_dir
        self.path = out_dir / "run.h5"
        self._f = h5py.File(self.path, "w")
        self._data = self._f.create_group("data")
        self.lengths: list[int] = []

    @property
    def count(self) -> int:
        return len(self.lengths)

    def add(self, episode: dict[str, np.ndarray]) -> None:
        g = self._data.create_group(f"demo_{self.count}")
        for key, arr in episode.items():
            g.create_dataset(key, data=arr, compression="gzip")
        self.lengths.append(int(episode["done"].shape[0]))

    def close(self, meta: dict) -> None:
        self._f.attrs["meta"] = json.dumps(meta)
        self._f.close()


class TrajectoryRecorder:
    """Buffers a rollout and writes a self-describing trajectory bag."""

    # Per-step fields captured from the environment. Each maps to a callable that
    # returns a ``(num_envs, ...)`` tensor for the current state.
    def __init__(
        self,
        env,
        *,
        policy_spec: str,
        seed: int,
        checkpoint: str | None = None,
        out_dir: str | Path | None = None,
        fmt: str = "hdf5",
    ):
        self.env = env
        self.device = env.device
        self.n = env.num_envs

        robot = env.scene["robot"]
        self._ee_ids, ee_names = robot.find_bodies(_ee_body_name(env))
        self._ee_id = self._ee_ids[0]
        # Both arms (issue #89). Recording the right one alone hid every left-handed event: a
        # bulb carried in the left hand read as a bulb nobody was holding.
        left_ids, left_names = robot.find_bodies(_left_ee_body_name(env))
        self._left_ee_id = left_ids[0] if left_ids else None
        self._left_ee_name = left_names[0] if left_names else None
        self._left_contact = env.scene.sensors.get("left_hand_contact")

        # Resolve the retention manager once, not per step: whether a task wires
        # mdp.bulb_attachment is fixed for the whole run, and a key that appeared midway
        # through would give the buffers ragged lengths. Tasks without the term (Remove,
        # Install, Carry today) simply record no lock columns.
        self._attachment = _attach.attachment_manager(env)

        # Which bulb this task manipulates, resolved once (issue #76 Step 1). Bulbs are named by
        # placement: a bulb seated in the socket is ``old_bulb``, one anywhere else is
        # ``fresh_bulb``. Remove and Carry build only the seated one, so a hardcoded lookup
        # raised KeyError for them. The serialized keys stay ``bulb_*`` so old bags still parse.
        #
        # KNOWN LIMIT, carried over from the `scene["bulb"]` lookup this replaces: presence
        # cannot disambiguate a two-bulb scene. S03-S06 run the Replace scene and manipulate the
        # OLD bulb, and this picks the fresh one for them, so their bags record a parked bulb.
        # The behaviour is unchanged by the rename -- the old lookup resolved to the same fresh
        # bulb -- but fixing it needs the task to DECLARE its manipuland, which is #76 Step 3.
        self._bulb_entity = "fresh_bulb" if "fresh_bulb" in env.scene.rigid_objects else "old_bulb"
        self._old_bulb_binned = _old_bulb_starts_binned(env)

        # The grip channel (issue #106), optional: only the legs that hold something wire
        # ``grip_contact``. Resolved once for the same reason the bulb is -- a key that appeared
        # midway through the run would give the buffers ragged lengths.
        self._grip_sensor = "grip_contact" if "grip_contact" in env.scene.sensors else None
        self._extra_contacts = _extra_contact_names(env)

        # EVERY rigid object, not one guessed bulb. The `_bulb_entity` heuristic above cannot
        # disambiguate a two-bulb scene, so S03-S06's bags recorded a parked bulb while the task
        # was about the other one; and the crate every disposal gate reads was absent entirely.
        # An env cfg may curate the list with a `record_objects` tuple.
        self._tracked_objects = _tracked_object_names(env)
        # Resolved once, not per step: `env.scene[name]` is a dict/category lookup, and
        # `object_state_fields` runs on every step of every episode.
        self._tracked_entities = [(name, env.scene[name]) for name in self._tracked_objects]
        # Issue #202: every bulb the scene carries, not the one this leg happens to manipulate.
        # The old guess pointed S03-S06's penalty channels at a fresh bulb parked on the bench,
        # so a dropped or crushed old bulb scored `penalties: none`. Intersected with the tracked
        # objects on purpose: these names are published as ``penalised_bulbs`` and an offline
        # scorer reads a ``<name>_pos`` column for each, so naming a bulb the bag does not carry
        # would be a penalty that silently never fires -- this issue over again.
        self._bulb_entities = [n for n in ("fresh_bulb", "old_bulb") if n in self._tracked_objects]
        # The success gate's conjuncts, so `gate_progress` -- half of a subtask's score -- can be
        # recomputed offline instead of only existing inside a live reward manager.
        self._gate_conjuncts, self._gate_seconds = self._resolve_gate(env)
        self._gate_resolved = False
        self._gate_col_names: list[str] = []

        self._buf: dict[str, list[np.ndarray]] = {}
        self._base = 0
        self._t = 0
        self._ep_start = np.zeros(self.n, dtype=np.int64)
        self._sink = _EpisodeSink(Path(out_dir)) if out_dir is not None and fmt == "hdf5" else None
        self._meta = self._build_meta(policy_spec=policy_spec, seed=seed, checkpoint=checkpoint, ee_name=ee_names[0])

    def _resolve_gate(self, env) -> tuple[list, float]:
        """The ``success`` term's conjunct list and sustain window, or ``([], 0.0)``.

        Gates come in two shapes -- ``sustained`` (predicates nested under ``predicate_params``,
        plus a hold window) and a bare ``all_of`` -- so this goes through the benchmark's own
        unwrapper rather than reaching into either shape by hand. Overridable: the teleop driver
        clears the termination so a success cannot reset the scene mid-take, and stashes the gate
        elsewhere.
        """
        try:
            from .tasks.manager_based.fiatlux_task.mdp.gates import conjuncts_of

            term = getattr(getattr(env.cfg, "terminations", None), "success", None)
            if term is None:
                return [], 0.0
            params = term.params or {}
            return list(conjuncts_of(term.func, params)), float(params.get("seconds") or 0.0)
        except Exception:  # noqa: BLE001 - a task without a data-shaped gate records no columns
            return [], 0.0

    # -- capture ---------------------------------------------------------------
    def object_state_fields(self) -> dict:
        """Pose and both velocities for every tracked object.

        Angular velocity as well as linear: ``place_terms.object_at_rest`` gates on both, so
        without it an object that is still rocking reads as settled offline while the live gate
        stays shut.
        """
        fields: dict = {}
        for name, obj in self._tracked_entities:
            fields[f"{name}_pos"] = obj.data.root_pos_w
            fields[f"{name}_quat"] = obj.data.root_quat_w
            fields[f"{name}_lin_vel"] = obj.data.root_lin_vel_w
            fields[f"{name}_ang_vel"] = obj.data.root_ang_vel_w
        return fields

    def contact_fields(self) -> dict:
        """One ``<sensor>_force`` column per contact sensor that has no named column of its own."""
        return {
            f"{name}_force": _obs.object_contact_forces(self.env.scene.sensors[name]) for name in self._extra_contacts
        }

    def gate_fields(self) -> dict:
        """One column per success-gate conjunct, evaluated live.

        A bag that carries only ``success_term`` says whether the episode finished and nothing
        about how far it got. These columns are what ``scripts/score.py`` recomputes
        ``gate_progress`` from, and what tells an operator which condition broke a take.

        The evaluable set is settled on the first recorded step and fixed from then on. The
        recorder is built before ``env.reset``, so a conjunct reading a sensor cannot be probed
        in ``__init__``; and a column that came and went would give the per-episode buffers
        ragged lengths. A conjunct that cannot be evaluated is dropped, loudly, rather than
        recorded as a silent False that would read offline as a condition genuinely unmet.

        A task that declared a gate and can evaluate none of it raises rather than recording a run
        that turns out to be unscoreable only once it is over.

        A conjunct that evaluated fine on the probing step but raises on some later one (a
        transient sensor read, a NaN) does not get to take the whole run down with it: its column
        holds its last known value for that step, and the failure is still printed so it is not
        silently wrong offline.
        """
        if self._gate_resolved:
            row = {}
            for (fn, params), col_name in zip(self._gate_conjuncts, self._gate_col_names):
                try:
                    row[col_name] = self._gate_last_values[col_name] = fn(self.env, **(params or {}))
                except Exception as e:  # noqa: BLE001
                    msg = f"gate conjunct {col_name} failed this step ({e}); holding its last value"
                    print(f"[recording] WARNING: {msg}", flush=True)
                    row[col_name] = self._gate_last_values[col_name]
            return row

        self._gate_resolved = True
        had_conjuncts = bool(self._gate_conjuncts)
        usable, dropped = [], []
        for fn, params in self._gate_conjuncts:
            try:
                usable.append((fn, params, fn(self.env, **(params or {}))))
            except Exception as e:  # noqa: BLE001
                dropped.append(f"{getattr(fn, '__name__', 'conjunct')} ({e})")
        if dropped:
            print(f"[recording] WARNING: gate conjuncts not recordable: {', '.join(dropped)}", flush=True)
        if had_conjuncts and not usable:
            raise RuntimeError(
                "every success-gate conjunct failed to evaluate, so this run can record no gate "
                f"columns and cannot be scored: {'; '.join(dropped)}"
            )
        self._gate_conjuncts = [(fn, params) for fn, params, _ in usable]
        base_names = [getattr(fn, "__name__", "conjunct") for fn, _, _ in usable]
        self._gate_col_names = [f"gate_{n}" for n in _disambiguate_names(base_names)]
        self._meta["gate_conjuncts"] = [n.replace("gate_", "", 1) for n in self._gate_col_names]
        self._gate_last_values = {col_name: value for col_name, (_, _, value) in zip(self._gate_col_names, usable)}
        return dict(self._gate_last_values)

    def world_state_fields(self) -> dict:
        """Root pose/velocity channels every task shares (issue #107).

        The robot's root pose was missing from the bag, which left an S01 evaluation short of the
        quantities its success conditions are written in. The joint vector alone does not give it:
        a floating-base robot's root pose is not derivable from ``joint_pos``. The ladder is not
        here -- it is an ordinary tracked object, and ``object_state_fields`` writes it under the
        same four names.

        ``grip_force`` is the ladder-or-payload grip channel (issue #106). The shared
        ``hand_contact`` sensor filters the bulbs the preset built, so on a ladder leg it reads a
        flat 0.0 N -- indistinguishable from a dead sensor. ``grip_contact`` is filtered to ONE
        target, so its column is attributable; recording it is what makes "was the ladder actually
        gripped" answerable offline.
        """
        env = self.env
        robot = env.scene["robot"]
        fields = {
            "robot_root_pos": robot.data.root_pos_w,
            "robot_root_quat": robot.data.root_quat_w,
            "robot_root_lin_vel": robot.data.root_lin_vel_w,
            "robot_root_ang_vel": robot.data.root_ang_vel_w,
        }
        if self._grip_sensor is not None:
            fields["grip_force"] = _obs.object_contact_forces(env.scene.sensors[self._grip_sensor])
        return fields

    def record_step(self, obs, actions, reward, terminated, truncated) -> None:
        env = self.env
        robot = env.scene["robot"]
        bulb = env.scene[self._bulb_entity]
        socket = env.scene["socket"]
        contact = env.scene.sensors["hand_contact"]

        done = terminated | truncated
        body_state = robot.data.body_state_w[:, self._ee_id, :7]  # (N, 7) pos+quat

        step = {
            "actions": actions,
            "joint_pos": robot.data.joint_pos,
            "joint_vel": robot.data.joint_vel,
            "joint_acc": robot.data.joint_acc,
            "eef_pose": body_state,
            **(
                {"eef_pose_left": robot.data.body_state_w[:, self._left_ee_id, :7]}
                if self._left_ee_id is not None
                else {}
            ),
            "bulb_pos": bulb.data.root_pos_w,
            "bulb_quat": bulb.data.root_quat_w,
            "bulb_lin_vel": bulb.data.root_lin_vel_w,
            "socket_pos": socket.data.root_pos_w,
            "socket_quat": socket.data.root_quat_w,
            "contact_force": _obs.object_contact_forces(contact),  # (N, B, 3), objects only
            **(
                {"contact_force_left": _obs.object_contact_forces(self._left_contact)}
                if self._left_contact is not None
                else {}
            ),
            "policy_obs": obs["policy"] if isinstance(obs, dict) else obs,
            "reward": reward,
            "pos_error": _rewards._bulb_socket_pos_error(env, self._bulb_entity),
            "ori_error": _rewards._bulb_socket_ori_error(env, self._bulb_entity),
            "step_in_episode": env.episode_length_buf.clone(),
            "terminated": terminated,
            "truncated": truncated,
            "done": done,
            "success_term": term_flag(env, "success", self.n, self.device),
            "dropped_term": term_flag(env, "bulb_dropped", self.n, self.device),
            "timeout_term": term_flag(env, "time_out", self.n, self.device),
        }
        step.update(self.world_state_fields())
        step.update(self.object_state_fields())
        step.update(self.contact_fields())
        step.update(self.gate_fields())
        # Seated state (issue #167). Only tasks that wire mdp.bulb_attachment have it.
        if self._attachment is not None:
            step.update(_attach.bulb_lock_telemetry(env))
        for key, value in step.items():
            self._buf.setdefault(key, []).append(_np(value))
        self._t += 1
        self._flush_completed()

    # -- streaming -------------------------------------------------------------
    def _flush_completed(self) -> None:
        """Write out every episode that ended on the step just recorded, then drop the
        steps no in-flight episode still needs."""
        if self._sink is None or "done" not in self._buf:
            return
        done = np.asarray(self._buf["done"][-1]).reshape(-1)
        end = self._t - 1
        for e in np.nonzero(done)[0]:
            lo = int(self._ep_start[e]) - self._base
            hi = end - self._base
            self._sink.add({k: np.stack([row[e] for row in v[lo : hi + 1]], axis=0) for k, v in self._buf.items()})
            self._ep_start[e] = end + 1
        keep = int(self._ep_start.min())
        if keep > self._base:
            cut = keep - self._base
            for v in self._buf.values():
                del v[:cut]
            self._base = keep

    # -- serialization ---------------------------------------------------------
    def episodes(self) -> list[dict[str, np.ndarray]]:
        """Segment the flat buffer into completed episodes (split on ``done``)."""
        if not self._buf:
            return []
        stacked = {k: np.stack(v, axis=0) for k, v in self._buf.items()}  # (T, N, ...)
        done = stacked["done"]  # (T, N) bool
        t_steps = done.shape[0]

        episodes: list[dict[str, np.ndarray]] = []
        for e in range(self.n):
            start = 0
            for t in range(t_steps):
                if done[t, e]:
                    episodes.append({k: v[start : t + 1, e] for k, v in stacked.items()})
                    start = t + 1
        return episodes

    def write(self, out_dir: str | Path, *, fmt: str = "hdf5") -> dict:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self._meta["fields"] = sorted(self._buf.keys())

        if self._sink is not None:
            if fmt != "hdf5":
                # The sink was already opened as hdf5 at construction (out_dir + the default
                # fmt="hdf5" is what creates it); honoring a different fmt here would mean
                # transcoding the episodes already streamed out, not just picking a format.
                raise ValueError(
                    f"recorder was constructed with out_dir set, which streams to hdf5 as it "
                    f"records; write(fmt={fmt!r}) cannot retroactively change that. Pass "
                    f"fmt='hdf5' (or omit it), or construct the recorder without out_dir to "
                    f"buffer in RAM and pick a format at write() time."
                )
            self._meta["num_episodes"] = self._sink.count
            self._meta["episode_lengths"] = list(self._sink.lengths)
            self._meta["format"] = "hdf5"
            bag_path = self._sink.path
            self._meta["bag_file"] = bag_path.name
            self._sink.close(self._meta)
            n_episodes = self._sink.count
        else:
            episodes = self.episodes()
            self._meta["num_episodes"] = len(episodes)
            self._meta["episode_lengths"] = [int(ep["done"].shape[0]) for ep in episodes]
            self._meta["format"] = fmt

            if fmt == "hdf5":
                bag_path = out / "run.h5"
                self._write_hdf5(bag_path, episodes)
            elif fmt == "npz":
                bag_path = out / "run.npz"
                self._write_npz(bag_path, episodes)
            else:
                raise ValueError(f"unknown bag format: {fmt!r} (use 'hdf5' or 'npz')")

            self._meta["bag_file"] = bag_path.name
            n_episodes = len(episodes)

        meta_path = out / "meta.json"
        with open(meta_path, "w") as fh:
            json.dump(self._meta, fh, indent=2)
        return {"bag": str(bag_path), "meta": str(meta_path), "episodes": n_episodes}

    def _write_hdf5(self, path: Path, episodes: list[dict[str, np.ndarray]]) -> None:
        import h5py

        with h5py.File(path, "w") as f:
            f.attrs["meta"] = json.dumps(self._meta)
            data = f.create_group("data")
            for i, ep in enumerate(episodes):
                g = data.create_group(f"demo_{i}")
                for key, arr in ep.items():
                    g.create_dataset(key, data=arr, compression="gzip")

    def _write_npz(self, path: Path, episodes: list[dict[str, np.ndarray]]) -> None:
        # Flat layout: concatenate episodes and store boundary indices.
        flat: dict[str, list[np.ndarray]] = {}
        ends: list[int] = []
        offset = 0
        for ep in episodes:
            offset += int(ep["done"].shape[0])
            ends.append(offset)
            for key, arr in ep.items():
                flat.setdefault(key, []).append(arr)
        payload = {k: np.concatenate(v, axis=0) for k, v in flat.items()}
        payload["episode_ends"] = np.asarray(ends, dtype=np.int64)
        np.savez_compressed(path, **payload)

    # -- metadata --------------------------------------------------------------
    def _build_meta(self, *, policy_spec, seed, checkpoint, ee_name) -> dict:
        from .robots.g1 import G1_ARM_JOINTS, G1_HAND_JOINTS

        env = self.env
        cfg = env.cfg
        robot = env.scene["robot"]
        contact = env.scene.sensors["hand_contact"]

        success_params = _term_params(cfg, "success")
        drop_params = _term_params(cfg, "bulb_dropped")

        return {
            "task": getattr(cfg, "task_name", None) or type(cfg).__name__,
            # The gym id, which is what fiatlux_task.subtask_score keys its difficulty weights
            # on. `task` above is a cfg class name, so a bag could not be scored as a subtask.
            "task_id": _gym_task_id(env),
            "gate_conjuncts": [getattr(fn, "__name__", "conjunct") for fn, _ in self._gate_conjuncts],
            "gate_sustain_seconds": self._gate_seconds,
            "benchmark_version": _benchmark_version(),
            **_git_commit(),
            "recorder_version": RECORDER_VERSION,
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "policy": policy_spec,
            "checkpoint": checkpoint,
            "seed": seed,
            "num_envs": self.n,
            "sim_dt": float(cfg.sim.dt),
            "decimation": int(cfg.decimation),
            "step_dt": float(env.step_dt),
            "episode_length_s": float(cfg.episode_length_s),
            "action_dim": int(env.action_space.shape[-1]),
            # action columns (what the policy commands) vs joint_pos/vel columns (all joints).
            "action_joint_order": list(G1_ARM_JOINTS) + list(G1_HAND_JOINTS),
            "joint_names": list(robot.joint_names),
            "ee_body": ee_name,
            "contact_bodies": list(getattr(contact, "body_names", []) or []),
            "contact_bodies_left": list(getattr(self._left_contact, "body_names", []) or []),
            "ee_body_left": self._left_ee_name,
            "policy_obs_dim": _policy_obs_dim(env),
            "success_pos_threshold": float(success_params.get("pos_threshold", 0.015)),
            "success_ori_threshold": float(success_params.get("ori_threshold", 0.2)),
            "drop_min_height": float(drop_params.get("min_height", 0.4)),
            # Issue #202: breaking either bulb is a penalty in every subtask, so both are
            # scored. Dropping is per bulb, against a floor the TASK sets -- see
            # ``_drop_floors_by_bulb``.
            "penalised_bulbs": list(self._bulb_entities),
            "drop_min_height_by_bulb": self._drop_floors_by_bulb(cfg),
            "glass_contact_limit_n": GLASS_CONTACT_LIMIT_N,
            # Issue #167: says whether the *_phase columns are present, so an offline reader
            # does not have to probe the arrays to find out.
            "has_bulb_attachment": self._attachment is not None,
        }

    def _drop_floors_by_bulb(self, cfg) -> dict:
        """The height each bulb must stay above, per bulb. ``None`` means exempt.

        A single floor cannot serve both bulbs, because the height that means "dropped" depends
        on where the TASK wants the bulb to end up, and the two ends differ: S05 carries the old
        bulb and a metre of altitude lost on the way is a fault, while S06 *disposes* of the same
        bulb and a successful take leaves it on the crate floor at z ~= 0. Nothing about the
        start state separates those, so the task has to say, in this order:

        1. ``cfg.bulb_drop_floors`` -- the task states it outright (``None`` == exempt).
        2. the task's own ``bulb_dropped`` termination, for the bulb that term names: a task
           that already declares what counts as dropping should not have it restated here.
        3. a bulb that BEGINS in the crate is already disposed of (S07 on), so exempt.
        4. otherwise the family default.
        """
        declared = dict(getattr(cfg, "bulb_drop_floors", None) or {})
        drop_params = _term_params(cfg, "bulb_dropped")
        # A term that names no asset means the bulb the task is about (S08 does this).
        term_asset = getattr(drop_params.get("asset_cfg", None), "name", None) or (
            self._bulb_entity if drop_params else None
        )
        term_height = drop_params.get("min_height", None)
        floors: dict = {}
        for name in self._bulb_entities:
            if name in declared:
                floors[name] = None if declared[name] is None else float(declared[name])
            elif term_asset == name and term_height is not None:
                floors[name] = float(term_height)
            elif name == "old_bulb" and self._old_bulb_binned:
                floors[name] = None
            else:
                floors[name] = DEFAULT_BULB_DROP_FLOOR_M
        return floors


def _old_bulb_starts_binned(env) -> bool:
    """Whether the old bulb begins the episode already in the disposal crate (issue #202)."""
    if "old_bulb" not in env.scene.rigid_objects or "bin" not in env.scene.rigid_objects:
        return False
    bulb = env.scene["old_bulb"].data.root_pos_w[0]
    crate = env.scene["bin"].data.root_pos_w[0]
    return bool((bulb[:2] - crate[:2]).norm() < 0.35 and bulb[2] < 0.4)


def _ee_body_name(env) -> str:
    """Resolve the end-effector body name from the package constant."""
    from .robots.g1 import G1_EE_BODY

    return G1_EE_BODY


def _left_ee_body_name(env) -> str:
    """The left arm's end-effector body (issue #89)."""
    from .robots.g1 import G1_LEFT_EE_BODY

    return G1_LEFT_EE_BODY


def _policy_obs_dim(env) -> int | None:
    """Width of the policy observation group, if discoverable."""
    try:
        return int(env.observation_space["policy"].shape[-1])
    except (TypeError, KeyError, AttributeError, IndexError):
        return None


def _term_params(cfg, name: str) -> dict:
    term = getattr(cfg.terminations, name, None)
    params = getattr(term, "params", None)
    return params if isinstance(params, dict) else {}
