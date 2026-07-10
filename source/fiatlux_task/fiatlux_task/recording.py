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
- ``write(out_dir, fmt)`` segments the buffered stream into per-episode groups
  (split on the ``done`` flag, robomimic-style ``data/demo_<i>``) and writes either
  HDF5 (default; ``h5py`` ships with Isaac Lab) or a flat ``.npz`` fallback, plus a
  human-readable ``meta.json`` header.

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
from pathlib import Path

import numpy as np
import torch

from .tasks.manager_based.fiatlux_task.mdp import rewards as _rewards

RECORDER_VERSION = "1.0"


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


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().to("cpu").numpy()


def term_flag(env, name: str, n: int, device) -> torch.Tensor:
    """Read a termination term flag for the current step (False vector if absent).

    The term flags reflect the step that triggered the done and survive the
    in-``step`` auto-reset, so they are the correct source for "did this episode
    end in success / a drop". We try the public accessor first, then the manager's
    internal store for cross-version robustness. Shared with ``scripts/eval.py``,
    which reads the ``success`` term through it.
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


class TrajectoryRecorder:
    """Buffers a full rollout and writes a self-describing trajectory bag."""

    # Per-step fields captured from the environment. Each maps to a callable that
    # returns a ``(num_envs, ...)`` tensor for the current state.
    def __init__(self, env, *, policy_spec: str, seed: int, checkpoint: str | None = None):
        self.env = env
        self.device = env.device
        self.n = env.num_envs

        robot = env.scene["robot"]
        self._ee_ids, ee_names = robot.find_bodies(_ee_body_name(env))
        self._ee_id = self._ee_ids[0]

        self._buf: dict[str, list[np.ndarray]] = {}
        self._meta = self._build_meta(
            policy_spec=policy_spec, seed=seed, checkpoint=checkpoint, ee_name=ee_names[0]
        )

    # -- capture ---------------------------------------------------------------
    def record_step(self, obs, actions, reward, terminated, truncated) -> None:
        env = self.env
        robot = env.scene["robot"]
        bulb = env.scene["bulb"]
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
            "bulb_pos": bulb.data.root_pos_w,
            "bulb_quat": bulb.data.root_quat_w,
            "bulb_lin_vel": bulb.data.root_lin_vel_w,
            "socket_pos": socket.data.root_pos_w,
            "socket_quat": socket.data.root_quat_w,
            "contact_force": contact.data.net_forces_w,  # (N, B, 3)
            "policy_obs": obs["policy"] if isinstance(obs, dict) else obs,
            "reward": reward,
            "pos_error": _rewards._bulb_socket_pos_error(env),
            "ori_error": _rewards._bulb_socket_ori_error(env),
            "step_in_episode": env.episode_length_buf.clone(),
            "terminated": terminated,
            "truncated": truncated,
            "done": done,
            "success_term": term_flag(env, "success", self.n, self.device),
            "dropped_term": term_flag(env, "bulb_dropped", self.n, self.device),
            "timeout_term": term_flag(env, "time_out", self.n, self.device),
        }
        for key, value in step.items():
            self._buf.setdefault(key, []).append(_np(value))

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
                    episodes.append(
                        {k: v[start : t + 1, e] for k, v in stacked.items()}
                    )
                    start = t + 1
        return episodes

    def write(self, out_dir: str | Path, *, fmt: str = "hdf5") -> dict:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        episodes = self.episodes()

        self._meta["num_episodes"] = len(episodes)
        self._meta["episode_lengths"] = [int(ep["done"].shape[0]) for ep in episodes]
        self._meta["fields"] = sorted(self._buf.keys())
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
        meta_path = out / "meta.json"
        with open(meta_path, "w") as fh:
            json.dump(self._meta, fh, indent=2)
        return {"bag": str(bag_path), "meta": str(meta_path), "episodes": len(episodes)}

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
            "benchmark_version": _benchmark_version(),
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
            "policy_obs_dim": _policy_obs_dim(env),
            "success_pos_threshold": float(success_params.get("pos_threshold", 0.015)),
            "success_ori_threshold": float(success_params.get("ori_threshold", 0.2)),
            "drop_min_height": float(drop_params.get("min_height", 0.4)),
        }


def _ee_body_name(env) -> str:
    """Resolve the end-effector body name from the package constant."""
    from .robots.g1 import G1_EE_BODY

    return G1_EE_BODY


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
