---
pretty_name: Fiatlux policy baselines
license: cc-by-4.0
tags:
  - robotics
  - humanoid
  - isaac-lab
  - benchmark
size_categories:
  - n<1K
---

# Fiatlux policy baselines

Recorded and scored runs of the reference policies on the twelve [Fiatlux](https://github.com/haw-ai-i/fiatlux)
subtasks, as reported in the paper. Each run is a benchmark recorder bag plus its score, so the
numbers can be re-scored offline without a simulator.

## Layout

```
2026-09-14-zero-baseline/            # the zero-action policy
2026-09-14-random-baseline/          # the random-action policy
  FIATLUX-S<nn>-<Subtask>-v0/
    seed{0,1,2,3}/
      run.h5                         # observations, actions, contacts, gate signals per step
      meta.json                      # task id, benchmark version, commit, seed, policy, thresholds

2026-09-14-n17-zeroshot-dex3/        # GR00T N1.7 zero-shot, Dex3 hands
  S<nn>-<Subtask>/
    run.h5, meta.json                # one recorded run per subtask
    video/run.mp4, run_poster.png    # head-camera video of that run
  scores/seed{0,1,2,3}/S<nn>-<Subtask>.json   # scorer output per seed
  run.sh                             # the command line that produced the runs
  t4.md, t4.tex                      # the per-subtask table (success, gate, score, terminations)
```

292 files; 0.9 GB. Twelve subtasks; the zero and random baselines have four seeds each, while
n17-zeroshot-dex3 has one recorded run per subtask (re-scored across all four seeds under
`scores/`). Re-score any `run.h5` with the
benchmark's `scripts/score_subtasks.py`. The `commit` field in `meta.json` names the benchmark
revision that produced the run; the repository's history was flattened on 2026-09-21, so those
SHAs no longer resolve there.

## License

CC-BY-4.0. Cite the Fiatlux paper when you use these runs (see the
[repository README](https://github.com/haw-ai-i/fiatlux#readme)).
