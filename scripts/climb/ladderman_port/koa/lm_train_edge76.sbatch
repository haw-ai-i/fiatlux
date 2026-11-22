#!/bin/bash
# Teacher on the fiatlux STEP LADDER with the open Dex3 hand, reference clip "edge72".
#
# What is new against the topout runs
# -----------------------------------
#   ladder   : the fiatlux step ladder as collision boxes (ladder_config.kind=fiatlux_step),
#              hold anchors from the clip's own anchors.json; geometry randomisation OFF.
#   robot    : g1_29dof_dex3open-complexfeet-complex (rigid open Dex3 hand, 34.39 kg) with the
#              fiatlux actuator gains (task.robot.name=g1_fiatlux).
#   wrists   : not policy outputs; their position targets follow the clip (JointPositionRefWrist).
#   clip     : edge76 -- rebuild 74 (the two-handed climb, handrail hold counted 1.0 s after both hands are
#              on, torso overlap removed) with a WALK-IN in front: it starts in S02's spawn pose (upright, arms
#              at zero, 0.22 m behind the old first frame), crouches, takes three short steps (left half, right,
#              left) and joins the old clip at its step 20. 265 walk-in steps; every old step number is +245.
#              1,724 steps, plus a 180-step TAIL (make_topout_suffix.py): two short steps forward on the platform and the
#              arms lowered, ending in the next stage's (S03) measured start pose. 1,904 steps; episode limit 2,000.
#   starts   : RobotTrackingSpawn (ladderman_port/command_spawn.py). 30% of episodes start at step 0, as every
#              benchmark episode does, and a start inside the 265 walk-in steps takes the benchmark's reset
#              range for the root, x and y +-0.05 m (yaw +-0.1 rad and joints +-0.1 rad are LadderMan's already,
#              and cover the benchmark's +-0.1 rad and +-0.05 rad). Mid-climb starts keep LadderMan's +-0.02 m.
#
# COLD start: ladder, hand, action term and clip all differ from every earlier teacher.
# A requeue or a resubmission resumes from this run family's newest checkpoint.
#SBATCH -J lm-edge76
#SBATCH -A ece605
#SBATCH -p ece605
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -c 4
#SBATCH --mem=96G   # measured: ~62 GiB at 4096 envs; 48G was OOM-killed
#SBATCH -o lm-edge76.%j.out
set -eo pipefail
ROOT=/mnt/lustre/koa/scratch/$USER/ladderman
export PIP_CACHE_DIR=$ROOT/pipcache XDG_CACHE_HOME=$ROOT/cache XDG_DATA_HOME=$ROOT/share
export TMPDIR=$ROOT/tmp OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
cd "$ROOT/LadderMan"
source thirdparty/miniconda3/etc/profile.d/conda.sh
conda activate ladderman

MINE=$(squeue -h -u "$USER" -t RUNNING -o '%i %j' | grep -E 'lm-edge76' | grep -v "^$SLURM_JOB_ID " || true)
if [ -n "$MINE" ]; then
  echo "[lm] another lm-edge76 is already RUNNING -- standing down so we do not fork the lineage:"
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
CLIP=data/motion/ladder_data/edge76
OUT="$ROOT/teacher/edge76-$SLURM_JOB_ID"
mkdir -p "$OUT"

CKPT=$(ls -t "$ROOT"/teacher/edge76-*/checkpoints/checkpoint_[0-9]*.pt 2>/dev/null | head -1 || true)
# First start: continue the two-handed teacher (lm-edge72, iteration 4,800: 48 of 64 full climbs) from a
# pinned copy, so this job never depends on, or touches, the run that produced it.
if [ -z "$CKPT" ] && [ -f "$ROOT/teacher/keep/edge72_checkpoint_4800.pt" ]; then
  mkdir -p "$ROOT/teacher/edge76-seed"; cp -n "$ROOT/teacher/keep/edge72_checkpoint_4800.pt" "$ROOT/teacher/edge76-seed/checkpoint_4800.pt"
  CKPT="$ROOT/teacher/edge76-seed/checkpoint_4800.pt"; echo "[lm] no edge76 checkpoint yet: continuing the two-handed teacher"
fi
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
    +task.randomization.perturb_body_materials.homogeneous=true \
    +task.sim.physx_capacity_scale=1.5 \
    task.max_episode_length=2000 \
    task.action._target_=active_adaptation.envs.mdp.action_refwrist.JointPositionRefWrist \
    task.command._target_=active_adaptation.envs.mdp.commands.hdmi.command_spawn.RobotTrackingSpawn \
    +task.command.start_zero_prob=0.3 +task.command.floor_steps=265 \
    '+task.command.floor_pose_range={x:[-0.05,0.05],y:[-0.05,0.05]}' \
    task.num_envs="$ENVS" algo.max_iters="$ITERS" \
    hydra.run.dir="$OUT" \
    wandb.mode=disabled
