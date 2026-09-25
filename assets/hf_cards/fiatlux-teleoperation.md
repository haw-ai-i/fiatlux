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

Everything sits under one top-level `2026-09-13-dex3-teleop-takes/` folder, in two trees.

The eight non-climb subtasks (S01, S03, S05–S09, S11) are split by the scorer's verdict:

```
FIATLUX-S<nn>-<Subtask>-Teleop-v0/dex3/hdf5/vr/{success,fail}/
  <YYYY-MM-DD>_<HHMMSS>_ep<nn>_score<x.xx>/
    run.h5              # observations, actions, contacts, gate signals per step
    meta.json           # task id, benchmark version, commit, seed, joint order, thresholds
    score_report.txt    # the scorer's verdict and gate timeline
    ego.mp4             # head-camera video of the episode
    ego_poster.png      # its first frame
    video.mp4           # third-person video of the episode
    video_poster.png    # its first frame
```

`success/` holds 80 takes, all scored 1.00; `fail/` holds 27, all 0.00.

The four climb subtasks (S02, S04, S10, S12) are grouped by ladder tread instead, with no
success/fail split -- the score in the folder name is the verdict:

```
climb-task/S<nn>-<Subtask>/tread<N>/
  run.log                         # the driver's console log for the take
  ep00_score<x.xx>/               # same seven files as above
  failed-attempt<k>/              # an earlier, abandoned try at the same tread, when kept
    run.log
    ep00_score<x.xx>/
climb-task/_grids/*.mp4           # side-by-side montages of the tread takes
```

That is 18 climb takes. The `<task>/dex3/hdf5/vr/` prefix matches the recorder's own capture tree
(`docs/subtask_teleop.md` in the benchmark repository); the date/time folders the recorder writes
were folded into each take's folder name when the takes were published.

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
