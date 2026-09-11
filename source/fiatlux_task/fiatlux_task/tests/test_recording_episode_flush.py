# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Guardrails for the recorder's per-episode flush (issue #168).

``recording`` imports the task's mdp package, which pulls in Isaac Lab, so the three modules it
needs are stubbed here, as in ``test_recording_gate_columns``. The recorder is then driven
through its buffer bookkeeping directly: a live env is not needed to segment a stream.
"""

import sys
import types

import numpy as np
import pytest

# env e finishes an episode on each of these steps; env 2 never finishes a second one.
DONE_STEPS = {0: (4, 9, 14), 1: (6, 13), 2: (11,)}
N_ENVS = 3
N_STEPS = 20


@pytest.fixture(scope="module")
def recording():
    for name in (
        "fiatlux_task.tasks",
        "fiatlux_task.tasks.manager_based",
        "fiatlux_task.tasks.manager_based.fiatlux_task",
        "fiatlux_task.tasks.manager_based.fiatlux_task.mdp",
    ):
        sys.modules.setdefault(name, types.ModuleType(name))
    mdp = sys.modules["fiatlux_task.tasks.manager_based.fiatlux_task.mdp"]
    for sub in ("attach", "observations", "rewards"):
        mod = types.ModuleType(f"{mdp.__name__}.{sub}")
        sys.modules[mod.__name__] = mod
        setattr(mdp, sub, mod)

    import fiatlux_task.recording as rec

    return rec


def _make(rec, out_dir):
    r = object.__new__(rec.TrajectoryRecorder)
    r.n = N_ENVS
    r._buf = {}
    r._base = 0
    r._t = 0
    r._ep_start = np.zeros(N_ENVS, dtype=np.int64)
    r._sink = rec._EpisodeSink(out_dir) if out_dir is not None else None
    r._meta = {}
    return r


def _step_rows(t):
    done = np.array([t in DONE_STEPS[e] for e in range(N_ENVS)], dtype=bool)
    value = np.array([[t * 100 + e, e] for e in range(N_ENVS)], dtype=np.float32)
    return {"done": done, "value": value}


def _feed(r, steps=N_STEPS):
    widths = []
    for t in range(steps):
        for key, arr in _step_rows(t).items():
            r._buf.setdefault(key, []).append(arr)
        r._t += 1
        r._flush_completed()
        widths.append(len(r._buf["done"]))
    return widths


def _reference_episodes():
    """The split the pre-#168 buffered path produced, as (env, tuple-of-values)."""
    out = []
    for e in range(N_ENVS):
        start = 0
        for t in range(N_STEPS):
            if t in DONE_STEPS[e]:
                out.append(tuple(float(s * 100 + e) for s in range(start, t + 1)))
                start = t + 1
    return sorted(out)


def _streamed_episodes(path):
    import h5py

    out = []
    with h5py.File(path, "r") as f:
        data = f["data"]
        for name in sorted(data, key=lambda n: int(n.split("_")[-1])):
            out.append(tuple(float(v) for v in data[name]["value"][:, 0]))
    return sorted(out)


def test_streamed_episodes_match_the_buffered_split(recording, tmp_path):
    r = _make(recording, tmp_path)
    _feed(r)
    info = r.write(tmp_path)

    assert _streamed_episodes(info["bag"]) == _reference_episodes()
    assert info["episodes"] == sum(len(v) for v in DONE_STEPS.values())
    assert r._meta["episode_lengths"] == [5, 7, 5, 12, 7, 5]


def test_trailing_partial_episode_is_dropped(recording, tmp_path):
    r = _make(recording, tmp_path)
    _feed(r)
    info = r.write(tmp_path)

    written = _streamed_episodes(info["bag"])
    assert info["episodes"] == 6
    unfinished = sum(N_STEPS - (DONE_STEPS[e][-1] + 1) for e in range(N_ENVS))
    assert sum(len(ep) for ep in written) == N_STEPS * N_ENVS - unfinished
    assert not any(v % 100 == 2 and v >= 1200 for ep in written for v in ep)


def test_buffer_does_not_grow_with_the_run(recording, tmp_path):
    r = _make(recording, tmp_path)
    widths = _feed(r)

    assert max(widths) < N_STEPS
    assert widths[-1] == N_STEPS - min(DONE_STEPS[e][-1] + 1 for e in DONE_STEPS)


def test_without_out_dir_the_recorder_still_buffers_the_whole_run(recording, tmp_path):
    r = _make(recording, None)
    _feed(r)

    assert len(r._buf["done"]) == N_STEPS
    assert sorted(tuple(float(v) for v in ep["value"][:, 0]) for ep in r.episodes()) == _reference_episodes()
