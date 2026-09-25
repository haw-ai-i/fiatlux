---
pretty_name: Fiatlux teleoperation demonstrations
license: cc-by-4.0
tags:
  - robotics
  - teleoperation
  - humanoid
  - isaac-lab
  - imitation-learning
size_categories:
  - n<1K
---

# Fiatlux teleoperation demonstrations

VR teleoperation takes on the twelve [Fiatlux](https://github.com/haw-ai-i/fiatlux) subtasks
(`FIATLUX-S01-MoveLadder-Teleop-v0` … `S12`), recorded with the benchmark's own recorder and scored
offline by the benchmark's scorer. Recorded in simulation (Isaac Sim 5.1 / Isaac Lab) with the
Unitree G1 + Dex3 hands; the operator drove the arms and hands from a Pico VR headset while the
SONIC whole-body controller handled locomotion.

## Layout

```
FIATLUX-S<nn>-<Subtask>-Teleop-v0/dex3/hdf5/vr/<YYYY-MM-DD>/<HHMMSS>/
  ep<nn>_score<x.xx>/
    run.h5              # observations, actions, contacts, gate signals per step
    meta.json           # task id, benchmark version, commit, seed, joint order, thresholds
    ego.mp4             # head-camera video of the episode
    ego_poster.png      # first frame
    score_report.txt    # the scorer's verdict and gate timeline
```

125 episodes; 3.7 GB. The `run.h5` field list and the action joint order are in each `meta.json`.
The `commit` field names the benchmark revision that produced the take; the benchmark repository's
history was flattened on 2026-09-21, so those SHAs no longer resolve there. Files are read back with
the benchmark's `scripts/score.py` / `scripts/score_subtasks.py`.

## What is not in here

No operator identity, no real-robot data, no camera footage of people or places: every frame is
rendered from the simulator.

## License

CC-BY-4.0. Cite the Fiatlux paper when you use these demonstrations (see the
[repository README](https://github.com/haw-ai-i/fiatlux#readme)).
