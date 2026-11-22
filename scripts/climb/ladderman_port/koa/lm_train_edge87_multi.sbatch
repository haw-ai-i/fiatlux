#!/bin/bash
# Several rebuild-87 teachers in ONE job, one per GPU. ece605 allows one job per user, and the trainer is
# single-GPU, so the variants run as separate steps of this job, each with its own GPU, cores and memory.
# Each step is lm_train_edge87.sbatch run as a plain script with its variant in the environment; a requeue
# resumes every variant from its own newest checkpoint.
#
#   RUNS   space-separated variants "family:clip:ladder:seed:profile". Default, the two the node has room for:
#            edge87free : rebuild 87, free-standing benchmark ladder, continues lm-edge72 iteration 4,800
#            bulb17free : bulb rebuild 17 (rebuild 87's legs, bulb arm), free ladder, continues lm-bulb15
#                         iteration 11,400 (profile "bulb": the bulb arm's contact point and seed checkpoint)
#          Diagnostics that do not fit beside them (run them in a later job):
#            edge87fixed:edge87:fixed:0:std    what the free ladder costs
#            edge87nfree:edge87n:free:0:std    what the toe raise costs or buys
#   NEW    iterations to train on top of each run's seed checkpoint (default 7500).
#   SUFFIX appended to every family, so a test never shares a lineage with the real runs:
#       sbatch --export=ALL,SUFFIX=test,NEW=30 -t 1:00:00 lm_train_edge87_multi.sbatch
#
# TWO runs. A third exhausted the node's 251 GB (see MEMORY in lm_train_edge87.sbatch). Change --gres / -c /
# --mem together with the number of RUNS: 1 GPU, 4 cores, MEM_PER_RUN each.
#SBATCH -J lm-edge87
#SBATCH -A ece605
#SBATCH -p ece605
#SBATCH --requeue
#SBATCH --gres=gpu:2
#SBATCH -c 8
#SBATCH --mem=180G
#SBATCH -t 24:00:00
#SBATCH -o lm-edge87-multi.%j.out
set -eo pipefail
ROOT=/mnt/lustre/koa/scratch/$USER/ladderman
RUNS="${RUNS:-edge87free:edge87:free:0:std bulb17free:bulb17:free:0:bulb}"
NEW="${NEW:-7500}"
MEM_PER_RUN="${MEM_PER_RUN:-90G}"
cd "$ROOT/jobs"
echo "[multi] job $SLURM_JOB_ID on $(hostname), $(date); runs: $RUNS"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
pids=()
for r in $RUNS; do
  IFS=: read -r fam clip ladder seed prof <<< "$r"
  if [ "${prof:-std}" = bulb ]; then eef="[0.005,-0.03,0.0]"; ck=bulb15_checkpoint_11400.pt; else eef="[0.17,0.017,0.0]"; ck=edge72_checkpoint_4800.pt; fi
  total=$(( $(echo "$ck" | sed -E "s/.*checkpoint_([0-9]+)\.pt/\1/") + NEW ))
  fam="$fam${SUFFIX:-}"
  echo "[multi] starting $fam (clip $clip, ladder $ladder, seed $seed, from $ck to iteration $total) -> lm-edge87.$fam.$SLURM_JOB_ID.out"
  FAMILY="$fam" CLIPNAME="$clip" LADDER="$ladder" SEED="$seed" ITERS="$total" EEF_R="$eef" SEEDCKPT="$ck" \
    srun --exact -N1 -n1 -c4 --gres=gpu:1 --mem="$MEM_PER_RUN" \
         --output="lm-edge87.$fam.$SLURM_JOB_ID.out" --error="lm-edge87.$fam.$SLURM_JOB_ID.out" --open-mode=append \
         bash "$ROOT/jobs/lm_train_edge87.sbatch" &
  pids+=($!)
  # Start-ups one at a time: the next run starts once this one has printed its first training iteration,
  # or has died. (Kept from when the start-up hang was blamed on the shared simulator caches; the cause was
  # the node running out of memory. It still keeps two 4,096-environment scene builds from overlapping.)
  for _ in $(seq 1 180); do
    grep -aq "\[iter " "lm-edge87.$fam.$SLURM_JOB_ID.out" 2>/dev/null && break
    kill -0 "${pids[-1]}" 2>/dev/null || break
    sleep 10
  done
  grep -aq "\[iter " "lm-edge87.$fam.$SLURM_JOB_ID.out" 2>/dev/null && echo "[multi] $fam is training, $(date +%T)" || echo "[multi] WARNING: $fam did not reach its first iteration in 30 min, $(date +%T)"
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=$?; done
free -g | awk '/^Mem:/{print "[multi] node memory at the end: used "$3" G, available "$7" G of "$2" G"}'
echo "[multi] all runs ended, worst exit $rc, $(date)"
sacct -j "$SLURM_JOB_ID" --format=JobID%20,JobName%12,State,Elapsed,MaxRSS,ReqMem -n || true
exit $rc
