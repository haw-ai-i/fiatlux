# Scoring & Evaluation Protocol

Evaluation is a single command, `scripts/eval.py`, run from a fixed seed for a
fixed number of episodes. Same `--task`, `--seed`, `--policy` (and checkpoint)
→ same numbers.

```bash
python scripts/eval.py --task FIATLUX-Insert-v0 --policy random --episodes 20 --seed 0
python scripts/eval.py --task FIATLUX-Insert-v0 --policy rsl_rl --checkpoint <model.pt> \
    --episodes 50 --seed 0 --output results.json
```

## Metrics

| Metric | Meaning | Primary? |
| --- | --- | --- |
| `success_rate` | fraction of episodes ending with the bulb seated | ✅ headline |
| `mean_episode_length` | avg steps per episode | |
| `mean_final_pos_error` | avg bulb→socket distance at episode end (m) | |
| `mean_control_effort` | avg Σ(action²) per step | |
| `peak_contact_force` | max net hand contact force (N) | safety |

The benchmark is **multi-dimensional on purpose**: report success rate alongside
control effort and peak contact force — a policy that succeeds by slamming the
bulb in is not a good policy.

## Reporting convention

- Default protocol: `--episodes 50 --seed 0`.
- Report all five metrics, the policy type, and the checkpoint.
- For learned policies, also report seeds `0,1,2` and their mean ± std.

## Baselines

- `zero` — no action (sanity floor).
- `random` — uniform random actions (sanity floor).
- `rsl_rl` — a trained PPO checkpoint (`scripts/rsl_rl/train.py`).
