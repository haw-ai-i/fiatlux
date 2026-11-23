#!/bin/bash
# Teachers on the benchmark-bodied robot, one per GPU, in ONE job. A sibling of lm_train_edge87_multi.sbatch
# (which stays as it is); each step is lm_train_fxbody.sbatch with its variant in the environment.
#
#   RUNS   space-separated variants "family:clip:ladder:seed:profile". Default:
#            edge87fx : rebuild 87, free ladder, benchmark body, both arms held to the clip (profile std);
#                       continues CK_STD (default edge87free_checkpoint_7500.pt)
#            bulb20fx : bulb rebuild 20 (slow level opening, slow reach at the mount), the same plus the
#                       carry-hand terms (profile bulb); continues CK_BULB (default bulb17free_checkpoint_14100.pt)
#   NEW    iterations to train on top of each run's seed checkpoint (default 7500).
#   SUFFIX appended to every family, so a test never shares a lineage with the real runs:
#       sbatch --export=ALL,SUFFIX=test,NEW=30 -t 1:00:00 lm_train_fxbody_multi.sbatch
#   CARRY_FORCE_LIMIT / CARRY_JUMP_LIMIT are passed through to the bulb profile (see lm_train_fxbody.sbatch).
#
# TWO runs: see MEMORY in lm_train_edge87.sbatch. The benchmark body adds collision shapes; the scale test
# with SUFFIX=test is what says whether two still fit.
#   Profiles: std | bulb | desc | descbulb, each optionally with the suffix "feet" (e.g. stdfeet).
#   A descent at 4,096 environments needs PhysX pair buffers 3x the stock size (DESC_CAP), and those do not
#   fit a 16 GB card (CUDA out of memory at start-up on an A4000). On a 16 GB card: DESC_ENVS=2048 DESC_CAP=1.5.
#SBATCH -J lm-fxbody
#SBATCH -A ece605
#SBATCH -p ece605
#SBATCH --requeue
#SBATCH --gres=gpu:2
#SBATCH -c 8
#SBATCH --mem=180G
#SBATCH -t 24:00:00
#SBATCH -o lm-fxbody-multi.%j.out
set -eo pipefail
ROOT=/mnt/lustre/koa/scratch/$USER/ladderman
RUNS="${RUNS:-edge87fx:edge87:free:0:std bulb20fx:bulb20:free:0:bulb}"
CK_STD="${CK_STD:-edge87free_checkpoint_7500.pt}"; CK_BULB="${CK_BULB:-bulb17free_checkpoint_14100.pt}"
NEW="${NEW:-7500}"
MEM_PER_RUN="${MEM_PER_RUN:-90G}"
cd "$ROOT/jobs"
echo "[multi] job $SLURM_JOB_ID on $(hostname), $(date); runs: $RUNS"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
pids=()
for r in $RUNS; do
  IFS=: read -r fam clip ladder seed prof <<< "$r"
  if [ "${prof:-std}" = bulb ]; then eef="[0.005,-0.03,0.0]"; ck="$CK_BULB"; else eef="[0.17,0.017,0.0]"; ck="$CK_STD"; fi
  # profile suffixes: "feet" adds the world-frame ankle terms; a "desc" prefix is a descent (platform start,
  # larger PhysX pair buffers), seeded from CK_DESC / CK_DESCBULB
  feet=""; floor=265; cap=1.5; envs=4096; base="${prof:-std}"
  case "$base" in *feet) feet=1; base="${base%feet}";; esac
  case "$base" in
    desc)     base=std;  floor=0; cap="${DESC_CAP:-3.0}"; envs="${DESC_ENVS:-4096}"; eef="[0.17,0.017,0.0]";  ck="${CK_DESC:-edge87fx_checkpoint_8100.pt}";;
    descbulb) base=bulb; floor=0; cap="${DESC_CAP:-3.0}"; envs="${DESC_ENVS:-4096}"; eef="[0.005,-0.03,0.0]"; ck="${CK_DESCBULB:-bulb20fx_checkpoint_14700.pt}";;
  esac
  prof="$base"
  total=$(( $(echo "$ck" | sed -E "s/.*checkpoint_([0-9]+)\.pt/\1/") + NEW ))
  fam="$fam${SUFFIX:-}"
  echo "[multi] starting $fam (clip $clip, ladder $ladder, seed $seed, from $ck to iteration $total) -> lm-fxbody.$fam.$SLURM_JOB_ID.out"
  FAMILY="$fam" PROFILE="${prof:-std}" FEET="$feet" FLOOR_STEPS="$floor" CAPACITY_SCALE="$cap" ENVS="$envs" CLIPNAME="$clip" LADDER="$ladder" SEED="$seed" ITERS="$total" EEF_R="$eef" SEEDCKPT="$ck" \
    srun --exact -N1 -n1 -c4 --gres=gpu:1 --mem="$MEM_PER_RUN" \
         --output="lm-fxbody.$fam.$SLURM_JOB_ID.out" --error="lm-fxbody.$fam.$SLURM_JOB_ID.out" --open-mode=append \
         bash "$ROOT/jobs/lm_train_fxbody.sbatch" &
  pids+=($!)
  # Start-ups one at a time: the next run starts once this one has printed its first training iteration,
  # or has died. (Kept from when the start-up hang was blamed on the shared simulator caches; the cause was
  # the node running out of memory. It still keeps two 4,096-environment scene builds from overlapping.)
  for _ in $(seq 1 180); do
    grep -aq "\[iter " "lm-fxbody.$fam.$SLURM_JOB_ID.out" 2>/dev/null && break
    kill -0 "${pids[-1]}" 2>/dev/null || break
    sleep 10
  done
  grep -aq "\[iter " "lm-fxbody.$fam.$SLURM_JOB_ID.out" 2>/dev/null && echo "[multi] $fam is training, $(date +%T)" || echo "[multi] WARNING: $fam did not reach its first iteration in 30 min, $(date +%T)"
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=$?; done
free -g | awk '/^Mem:/{print "[multi] node memory at the end: used "$3" G, available "$7" G of "$2" G"}'
echo "[multi] all runs ended, worst exit $rc, $(date)"
sacct -j "$SLURM_JOB_ID" --format=JobID%20,JobName%12,State,Elapsed,MaxRSS,ReqMem -n || true
exit $rc
