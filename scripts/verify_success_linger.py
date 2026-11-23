"""Does a subtask still build and step with the after-success linger, and is its gate wired as intended?

Every subtask's `success` termination should be `mdp.sustained` carrying `linger_seconds`
(subtask_env_cfg.SUCCESS_LINGER_SECONDS), with the same conjunct list its leaf declares, and it
must not fire on a fresh episode.

    python scripts/verify_success_linger.py --task FIATLUX-S07-ApproachNewBulb-v0 --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True)
ap.add_argument("--steps", type=int, default=20)
AppLauncher.add_app_launcher_args(ap)
a = ap.parse_args()
a.enable_cameras = True
app = AppLauncher(a).app

import fiatlux_task.tasks  # noqa: E402,F401
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from fiatlux_task.tasks.manager_based.fiatlux_task import mdp  # noqa: E402
from fiatlux_task.tasks.manager_based.fiatlux_task.subtask_env_cfg import SUCCESS_LINGER_SECONDS  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

cfg = parse_env_cfg(a.task, num_envs=1)
leaf = mdp.conjuncts_of(cfg.success_predicate, cfg.success_params)
term = cfg.terminations.success
wired = mdp.conjuncts_of(term.func, term.params)
ok = True


def check(label, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {a.task} {label} {detail}")


check("term is sustained", term.func is mdp.sustained, f"({getattr(term.func, '__name__', term.func)})")
check(
    "linger",
    abs(float(term.params.get("linger_seconds", -1)) - SUCCESS_LINGER_SECONDS) < 1e-9,
    f"({term.params.get('linger_seconds')} s)",
)
check(
    "hold unchanged",
    float(term.params.get("seconds")) == float((cfg.success_params or {}).get("seconds", 0.0)),
    f"({term.params.get('seconds')} s)",
)
check(
    "conjuncts unchanged",
    [f for f, _ in wired] == [f for f, _ in leaf] and [p for _, p in wired] == [p for _, p in leaf],
    "(" + ", ".join(getattr(f, "__name__", "?") for f, _ in wired) + ")",
)
check(
    "gate_progress reads the same conjuncts",
    [f for f, _ in cfg.rewards.gate_progress.params["predicates"]] == [f for f, _ in leaf],
)
env = gym.make(a.task, cfg=cfg).unwrapped
env.reset()
fired = False
for _ in range(a.steps):
    _, _, term_, trunc_, _ = env.step(torch.zeros((1, env.action_space.shape[-1]), device=env.device))
    fired |= bool(env.termination_manager.get_term("success")[0])
check(f"steps {a.steps} times, success does not fire on a fresh episode", not fired)
print(f"[linger] {a.task}: {'ALL PASS' if ok else 'FAILED'}")
sys.stdout.flush()
env.close()
app.close()
