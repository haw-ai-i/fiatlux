#!/bin/bash
# Teacher on the fiatlux STEP LADDER with the open Dex3 hand, reference clip "edge67".
#
# What is new against the topout runs
# -----------------------------------
#   ladder   : the fiatlux step ladder as collision boxes (ladder_config.kind=fiatlux_step),
#              hold anchors from the clip's own anchors.json; geometry randomisation OFF.
#   robot    : g1_29dof_dex3open-complexfeet-complex (rigid open Dex3 hand, 34.39 kg) with the
#              fiatlux actuator gains (task.robot.name=g1_fiatlux).
#   wrists   : not policy outputs; their position targets follow the clip (JointPositionRefWrist).
#   clip     : edge67 -- palms on the front edges of tread 4, platform and handrail; feet grounded;
#              as edge60, plus feet level (pitch and roll), weight centred between the feet, soles kept out of the treads.
#              All 7 frozen poses hold 3 s on plain joint control in fiatlux. Known weak point: the left
#              shin grazes the next tread for about 0.5 s as the left foot lands on each step.
#
# This is the SECOND lineage (seed 1) on the H100/H200 nodes of kill-shared, which preempts:
# --requeue puts it back in the queue and it resumes from its own newest checkpoint (saved every
# 300 iterations), training only the iterations that are left. The ece605 job is lineage one.
#
# COLD start: ladder, hand, action term and clip all differ from every earlier teacher.
# A requeue or a resubmission resumes from this run family's newest checkpoint.
#SBATCH -J lm-edge67h
#SBATCH -p kill-shared
#SBATCH --constraint=hopper
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -t 3-00:00:00
#SBATCH -c 4
#SBATCH --mem=96G   # measured: ~62 GiB at 4096 envs; 48G was OOM-killed
#SBATCH -o lm-edge67h.%j.out
set -eo pipefail
ROOT=/mnt/lustre/koa/scratch/$USER/ladderman
export PIP_CACHE_DIR=$ROOT/pipcache XDG_CACHE_HOME=$ROOT/cache XDG_DATA_HOME=$ROOT/share
export TMPDIR=$ROOT/tmp OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
cd "$ROOT/LadderMan"
source thirdparty/miniconda3/etc/profile.d/conda.sh
conda activate ladderman

MINE=$(squeue -h -u "$USER" -t RUNNING -o '%i %j' | grep -E 'lm-edge67h' | grep -v "^$SLURM_JOB_ID " || true)
if [ -n "$MINE" ]; then
  echo "[lm] another lm-edge67h is already RUNNING -- standing down so we do not fork the lineage:"
  echo "$MINE"
  exit 0
fi

VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
ENVS="${ENVS:-4096}"
if [ "${VRAM:-0}" -lt 15000 ]; then
  echo "[lm] landed on a ${VRAM}MiB card; 4096 envs needs >=15000. Standing down."
  exit 0
fi
ITERS="${ITERS:-7500}"
CLIP=data/motion/ladder_data/edge67
OUT="$ROOT/teacher/edge67h-$SLURM_JOB_ID"
mkdir -p "$OUT"

CKPT=$(ls -t "$ROOT"/teacher/edge67h-*/checkpoints/checkpoint_[0-9]*.pt 2>/dev/null | head -1 || true)
RESUME=()
if [ -n "$CKPT" ]; then
  echo "[lm] resume from this run family: $CKPT"
  RESUME=(checkpoint_path="$CKPT")
  DONE=$(basename "$CKPT" .pt | sed 's/checkpoint_//')
  ITERS=$((ITERS - DONE))
  echo "[lm] $DONE iterations done, $ITERS left"
  if [ "$ITERS" -le 0 ]; then echo "[lm] nothing left to train"; exit 0; fi
else
  echo "[lm] cold start"
fi

echo "[lm] node=$(hostname) vram=${VRAM}MiB envs=$ENVS iters=$ITERS clip=$CLIP $(date)"
md5sum $CLIP/motion.npz $CLIP/anchors.json
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
free -g | awk '/^Mem:/{print "[lm] node RAM: "$2" G"}'

exec python scripts/train.py algo=teacher task=teacher "${RESUME[@]}" \
    task.robot.name=g1_fiatlux \
    task.robot.robot_type=g1_29dof_dex3open-complexfeet-complex \
    task.command.data_path=$CLIP \
    +task.command.ladder_config.kind=fiatlux_step \
    +task.command.ladder_config.anchors_file=$CLIP/anchors.json \
    +task.command.ladder_config.contact_idx=13 \
    '+task.command.contact_target_pos_offset=[[0.0,0.115,0.0],[0.0,-0.115,0.0]]' \
    'task.command.contact_eef_pos_offset=[[0.17,-0.017,0.0],[0.17,0.017,0.0]]' \
    task.randomization.ladder_geometry_randomization.enabled=false \
    task.action._target_=active_adaptation.envs.mdp.action_refwrist.JointPositionRefWrist \
    task.num_envs="$ENVS" algo.max_iters="$ITERS" \
    seed=1 \
    hydra.run.dir="$OUT" \
    wandb.mode=disabled
