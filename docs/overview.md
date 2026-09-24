# Overview

The Fiatlux Benchmark evaluates a robot's ability to perform **light-bulb
replacement** with a Unitree G1 humanoid in NVIDIA Isaac Lab. It is an internal
research benchmark: a small, reproducible Isaac Lab extension you can train and
compare policies against.

## Goal

Drive a grasped light bulb into a socket — aligning position and orientation and
seating it without excessive contact force. The task is framed as a standard
Isaac Lab manager-based RL environment.

## Why this design

- **Pure Python / Isaac Lab.** No ROS, no C++ engine, no Zenoh. Tasks are
  `gym.register`-ed env ids, like Meta-World / ManiSkill / NVIDIA's
  IsaacLabEvalTasks.
- **Sensor-realizable by default.** The default observation group only uses
  signals available on the real robot (proprioception, wrist camera, contact
  forces). Ground-truth poses live in a separate `privileged` group.
- **Hardware-realizable actions.** Joint-position targets map directly to the
  Unitree SDK for later sim-to-real.

## Documents

- [getting_started.md](getting_started.md) — install and run.
- [task_spec.md](task_spec.md) — observation / action / reward / success spec.
- [scoring.md](scoring.md) — the evaluation protocol and metrics.
- [roadmap.md](roadmap.md) — climbing, combined task, and sim-to-real.
