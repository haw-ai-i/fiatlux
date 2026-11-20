#!/bin/bash
# A racing COPY of one of the lm_train_*.sbatch jobs on another GPU type:
#
#     sbatch -J <same job name as the original> lm_copy.sbatch <inner script>
#     scontrol update JobId=<id> TresPerNode=gres/gpu:<type>:1
#
# The inner script's own guard makes the copies exclusive: whichever starts while another job of
# the same name is RUNNING stands down. They share one checkpoint family, so any of them resumes
# the others' work.
#
# Why a wrapper. This job has hung at simulator startup on H100 and RTX 5000 nodes (one thread at
# 100% CPU, GPU idle, for up to 2 h 12 min); it has only ever been seen training on A4000 cards.
# So a copy is given 25 minutes to print its first training iteration. If it has not, it is
# killed -- no idle GPU -- and, if that leaves no job of this name in the queue, the A4000
# original is put back.
#SBATCH -p kill-shared
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -t 3-00:00:00
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH -o lm-copy.%j.out
set -o pipefail
INNER=${1:?inner sbatch script}
JOBS=/mnt/lustre/koa/scratch/$USER/ladderman/jobs
LOG=$JOBS/lm-copy.$SLURM_JOB_ID.out
GRACE=${GRACE:-1500}
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
echo "[copy] $SLURM_JOB_NAME as a copy of $INNER on $(hostname), $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), ${VRAM} MiB, $(date -u)"
if [ "${VRAM:-0}" -lt 15000 ]; then echo "[copy] under 15000 MiB -- standing down"; exit 0; fi

setsid bash "$JOBS/$INNER" &
PID=$!
T0=$(date +%s)
while kill -0 $PID 2>/dev/null; do
  sleep 30
  if grep -aqE '[0-9]+/[0-9]+ \[[0-9:]+<' "$LOG"; then
    echo "[copy] training is iterating after $(( $(date +%s) - T0 )) s; watchdog done"
    wait $PID; exit $?
  fi
  if [ $(( $(date +%s) - T0 )) -gt "$GRACE" ]; then
    echo "[copy] no training iteration after ${GRACE} s on $(hostname) -- killing the hung start"
    kill -TERM -- -$PID 2>/dev/null; sleep 10; kill -KILL -- -$PID 2>/dev/null
    OTHERS=$(squeue -h -u "$USER" -n "$SLURM_JOB_NAME" -o '%i' | grep -v "^$SLURM_JOB_ID$" | wc -l)
    if [ "$OTHERS" -eq 0 ]; then
      NEW=$(sbatch --parsable --gres=gpu:1 --constraint= -J "$SLURM_JOB_NAME" "$JOBS/$INNER") && scontrol update JobId=$NEW TresPerNode=gres/gpu:nv-rtx-a4000:1
      echo "[copy] nothing else of this name queued; put the A4000 original back as $NEW"
    fi
    exit 0
  fi
done
wait $PID; exit $?
