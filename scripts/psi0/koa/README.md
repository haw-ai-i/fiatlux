# Running the S06 fine-tune on Koa

Koa is UH Manoa's HPC cluster. Nothing here changes the stock pipeline in
`docs/psi0_reproduce.md` section 9 — these scripts call `scripts/psi0/*` unmodified and add
only what a Slurm cluster with no Isaac Lab needs. Keep it that way: if a stage needs a change,
change it here, not in the script it wraps.

## Cluster facts worth knowing before you start

| | |
|---|---|
| login nodes | `$HOME`, `/tmp` and Lustre are **noexec**, and there is no `git` — nothing runs here |
| compute nodes | exec on all three, `/usr/bin/git`, full internet (PyPI, HF, GitHub) |
| `module load` | does **not** change `python3`; uv fetches its own standalone 3.11 |
| `$HOME` | 50 GB quota — keep venvs, caches and checkpoints out of it |
| scratch | `/mnt/lustre/koa/scratch/$USER` (PB-scale) — everything lives here |
| GPUs | `gpu` partition: A30 24 GB (non-preemptible). `kill-shared` has H100/H200/L40 but **preempts** |

## Once

```bash
export PSI0_KOA_REPO=$PWD                       # this fiatlux checkout
export PSI0_KOA_ROOT=/mnt/lustre/koa/scratch/$USER/psi0
mkdir -p "$PSI0_KOA_ROOT" && cd "$PSI0_KOA_ROOT"

printf '%s' 'hf_...'     > .hf_token   && chmod 600 .hf_token    # private teleop dataset
printf '%s' 'wandb_...'  > .wandb_key  && chmod 600 .wandb_key   # logging
```

Neither key is in the repo, and neither is read by anything but the script that needs it.

## Then

```bash
cd "$PSI0_KOA_ROOT"
HF_TOKEN=$(cat .hf_token) sbatch "$PSI0_KOA_REPO/scripts/psi0/koa/setup.sbatch"   # ~20 min
sbatch "$PSI0_KOA_REPO/scripts/psi0/koa/prep.sbatch"                              # ~12 min
sbatch "$PSI0_KOA_REPO/scripts/psi0/koa/finetune.sbatch"                          # 2000 steps
```

Each stage is idempotent — re-running skips whatever already exists — and logs to
`psi0-koa.<name>.<jobid>.out` in the directory you submit from. Run `prep` only after `setup`
reports `done`: it asserts on the artifacts `setup` fetches.

### When the `gpu` partition is backed up

`finetune.sbatch` asks for an A30 on `gpu` because that is the non-preemptible card the recipe
is tuned for. That queue is often deep — a single user's job array can put the estimated start
weeks out (`squeue --start -j <id>` tells you). `kill-shared` holds the bigger cards (L40 48 GB,
H100 80 GB, H200 141 GB) and usually has one free; it is preemptible, but `PreemptMode=REQUEUE`,
so a preempted job is resubmitted automatically. Command-line flags override the `#SBATCH`
directives, so no second script is needed:

```bash
export TS=$(date +%y%m%d%H%M)    # fixed, so a requeue resumes the same run dir
export CKPT_EVERY=500            # bounds what one preemption can cost
sbatch -p kill-shared --gres=gpu:NV-L40:1 --requeue \
    "$PSI0_KOA_REPO/scripts/psi0/koa/finetune.sbatch"
```

Both halves matter. The recipe already passes `--train.resume_from_checkpoint=latest`, but
`TS` defaults to the wall clock, so without pinning it a requeued job invents a new run
directory and trains from scratch instead of resuming. `GraceTime=0` on that partition means
there is no signal before the kill — the last checkpoint on disk is all you keep.

## What each script adds, and why

**`setup_koa.sh`** splits `setup.sh` either side of one pin. `pyproject.toml` pins
`numpy<2.0.0` but leaves `pyarrow` unbounded, and upstream's `uv.lock` at `4f3720d` does not
parse — so `setup.sh` re-resolves from `pyproject.toml` and lands on pyarrow ≥ 21, which
refuses to import against numpy 1.x:

```
ImportError: pyarrow requires NumPy 2.0 or newer, found 1.26.4
```

`datasets` needs pyarrow ≥ 15, so the venv is held in the 15–20 window after step 1, before
anything imports pyarrow. **This one is not Koa-specific** — any machine resolving today hits
it, and it takes down step 4, `build_psi0_pack.py` and the trainer alike. It lives here so the
stock path stays as the runbook describes it; move it into `setup.sh` if you would rather fix
it for everyone.

**`prep_koa.sh`** runs stages a, b, d and e. Two wrinkles:

- `encode_sonic_tokens.py` needs no simulator but imports `fiatlux_task.groot`, and that
  package's `__init__.py` does `from .tasks import *` → Isaac Lab. `run_encode.py` pre-seeds
  `sys.modules` with a bare `fiatlux_task` pointing at the real source directory, so the leaf
  modules import and `__init__.py` never runs.
- `build_psi0_pack.py` shells out to `ffmpeg` and `ffprobe`, which are not on Koa's default
  PATH. The FFmpeg module is loaded in a subshell so its GCCcore paths never reach the trainer.

It also passes `--encoder` explicitly, since the stock default is `~/tools/sonic_models/...`
and on Koa everything is in scratch.

**`finetune_koa.sh`** adds wandb. The recipe's header documents a deliberate "no wandb" and it
is not edited; instead the script regenerates a wandb-enabled copy from the pristine recipe on
every run, injecting `--log.report_to=wandb --wandb.project=...` and overriding `HERE` so the
copy still finds `train_sdpa.py` in the repo. Generation keys on one line and aborts if that
line moves, so the copy cannot silently drift. Logging itself is the trainer's own path
(Psi0's own `scripts/train.py` → `accelerator.init_trackers`), not a reimplementation. If the node cannot
reach `api.wandb.ai` the run goes offline rather than dying; the script prints the `wandb sync`
command.

Set `WANDB_PROJECT_NAME` to change the project (default `fiatlux-psi0-s06`). The run name is
the trainer's own: `$EXP` + task + `$TS`. All the recipe's knobs still work —
`BATCH`, `ACCUM`, `STEPS`, `WARMUP`, `LR`, `CKPT_EVERY`, `EXP`, `TS`.

## Evaluating

`eval_finetuned.sh` needs the simulator, so it does not run here. Pull the checkpoint from
`$PSI0_KOA_ROOT/ft/runs/finetune/<run>/` to a machine with Isaac Lab and follow stage g of the
runbook.
