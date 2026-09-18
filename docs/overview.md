# Overview

The Fiatlux Benchmark evaluates a robot's ability to perform **light-bulb
replacement** with a Unitree G1 humanoid in NVIDIA Isaac Lab. It is an internal
research benchmark: a small, reproducible Isaac Lab extension you can train and
compare policies against.

## Goal

Replace a light bulb in a ceiling or wall fixture: carry a step ladder to it, climb, remove the
old bulb, carry it down to a disposal crate, collect a fresh one from the bench, climb back and
seat it — aligning position and orientation without excessive contact force.

Two framings of the same work, both standard Isaac Lab manager-based RL environments:

- **`FIATLUX-Replace-v0`** — the whole job as **one flat episode**. The benchmark task.
- **Twelve subtasks** (`FIATLUX-S01-MoveLadder-v0` … `FIATLUX-S12-ClimbDown-v0`) — the same
  chain cut into legs, each with its own success gate, so a policy can be trained and scored on
  one capability at a time. Each has a `-Teleop-v0` twin for human demonstration.

## Why this design

- **Pure Python / Isaac Lab.** No ROS, no C++ engine, no Zenoh. Tasks are
  `gym.register`-ed env ids, like Meta-World / ManiSkill / NVIDIA's
  IsaacLabEvalTasks.
- **Sensor-realizable by default.** The default observation group only uses
  signals available on the real robot (proprioception, head-mounted RGB camera and lidar,
  contact forces). Ground-truth poses live in a separate `privileged` group — the benchmark
  calls the two modes **standard** and **cheatcode**.
- **Hardware-realizable actions.** Joint-position targets map directly to the
  Unitree SDK for later sim-to-real.

## Documents

- [getting_started.md](getting_started.md) — install, run, train, and the physics verifiers.
- [task_spec.md](task_spec.md) — observation / action / reward / success spec, for the flat
  task and for each of the twelve subtasks.
- [subtask_teleop.md](subtask_teleop.md) — driving the subtasks by hand, in VR or from the
  keyboard, and recording demonstrations.
- [../source/fiatlux_teleop/README.md](../source/fiatlux_teleop/README.md) — the teleop
  package itself: one-time setup, the headset flow, and how to make a new task teleop-able.
- [scoring.md](scoring.md) — the evaluation protocol, the metrics, and the baseline policies.
- [roadmap.md](roadmap.md) — climbing, combined task, and sim-to-real.
- [asset_collection_todo.md](asset_collection_todo.md) — where each USD asset came from.

Asset provenance — how the ladder colliders were authored, the Omniverse pack scan, the
material fixes — lives in [collision_authoring_explained.md](collision_authoring_explained.md)
and the `omniverse_*` docs. You do not need them to run the benchmark.
