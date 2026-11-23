#!/bin/bash
# Teacher on the BENCHMARK-BODIED robot (g1_29dof_dex3open-complexfeet-fxbody), with the opt-in arm and
# carry-hand terms. A sibling of lm_train_edge87.sbatch, which stays as it is.
#
# What is new against lm_train_edge87.sbatch
# ------------------------------------------
#   robot   : the benchmark's body collision shapes where the stock trainer robot differed (shoulder pitch and
#             roll colliders, upper arm, hip pitch, pelvis contour, chest plate: make_fxbody_urdf.py). Without
#             them a teacher tucks its arms through space that is solid in the benchmark. ROBOT overrides it.
#   rewards : PROFILE=std  -> both arms held to the clip (tight_arm_* terms, bulb_hand.py)
#             PROFILE=bulb -> the same, plus the carry hand: wrist-speed tracking, contact-force and
#                             acceleration penalties, and the two terminations (wrist force over
#                             CARRY_FORCE_LIMIT newtons, 0 = that termination off; velocity jump over
#                             CARRY_JUMP_LIMIT m/s, 0 = off; both exempt an episode's first CARRY_GRACE
#                             steps, default 25)
#             PROFILE=none -> stock rewards (to measure the body alone)
#   Everything else (ladder, wrists, starts, clip handling, memory notes, ITERS / FAMILY / CLIPNAME / LADDER /
#   SEED / EEF_R / SEEDCKPT) is lm_train_edge87.sbatch's; read its header.
#
#       sbatch --export=ALL,FAMILY=fxtest,ITERS=7530,SEEDCKPT=edge87free_checkpoint_7500.pt -t 1:00:00 lm_train_fxbody.sbatch
#   A second job beside the multi-GPU one (another partition, its OWN family): ALLOW_PARALLEL=1 skips the
#   stand-down below, e.g. on the preemptible partition, away from the ece605 node (two runs fill its memory):
#       sbatch -A uh -p kill-shared -J lm-bulb21 --exclude=gn-15-35-00 -t 2-00:00:00 \\
#           --export=ALL,ALLOW_PARALLEL=1,FAMILY=bulb21fx,PROFILE=bulb,CLIPNAME=bulb21,... lm_train_fxbody.sbatch
#   FLOOR_STEPS (default 265) is the length of the floor walk-in whose starts get the benchmark's +-0.05 m root
#   range; a DESCENT starts on the platform: FLOOR_STEPS=0. START_ZERO (default 0.3) is the share of episodes
#   forced to start at step 0.
#   CAPACITY_SCALE (default 1.5) scales PhysX's GPU pair buffers. The S12 descent asked for up to 171.6M
#   aggregate pairs against the 93M that 1.5 gives (`foundLostAggregatePairsCapacity ... will miss
#   interactions`, 24 times in 68 iterations): CAPACITY_SCALE=3.0 for the descents.
#   FEET=1 adds the two world-frame ankle terms (tight_feet_*): where the robot stands at the top.
#   RESET_LIFT (default 0) raises the root this many metres at every reset. A start on the PLATFORM needs it
#   (0.03): without it the ankles read up to 28 kN in the first step and 4,096 robots are thrown at once.
#   SPEED_LIMIT (m/s, default off) ends an episode on the step any body is faster than that; ENV_SPACING
#   (default the trainer's 3.0 m) spreads the environments. A descent needs both (6.0 and 6.0): a robot thrown
#   by a reset on the platform otherwise travels 4-9 m, into its neighbours, and the pair buffers explode.
#SBATCH -J lm-fxbody
#SBATCH -A ece605
#SBATCH -p ece605
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -c 4
#SBATCH --mem=120G  # fixed ladder measured ~62 GiB at 4096 envs (48G was OOM-killed); the free ladder is unmeasured
#SBATCH -o lm-fxbody.%j.out
set -eo pipefail
ROOT=/mnt/lustre/koa/scratch/$USER/ladderman
export PIP_CACHE_DIR=$ROOT/pipcache XDG_CACHE_HOME=$ROOT/cache XDG_DATA_HOME=$ROOT/share
export TMPDIR=$ROOT/tmp OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
cd "$ROOT/LadderMan"
source thirdparty/miniconda3/etc/profile.d/conda.sh
conda activate ladderman

FAMILY="${FAMILY:-edge87fx}"
ROBOT="${ROBOT:-g1_29dof_dex3open-complexfeet-fxbody}"
PROFILE="${PROFILE:-std}"
MINE=$(squeue -h -u "$USER" -t RUNNING -o '%i %j' | grep -E 'lm-fxbody' | grep -v "^$SLURM_JOB_ID " || true)
if [ -n "$MINE" ] && [ -z "${ALLOW_PARALLEL:-}" ]; then
  echo "[lm] another lm-fxbody is already RUNNING -- standing down so we do not fork the lineage:"
  echo "$MINE"
  exit 0
fi

# The trainer code this run needs. Refuse to start on an older copy rather than train something else.
for need in active_adaptation/envs/mdp/commands/hdmi/command_spawn.py \
            active_adaptation/envs/mdp/commands/hdmi/bulb_hand.py \
            "active_adaptation/assets/g1/$ROBOT.urdf" \
            assets_fiatlux/AlumStep_D/AluminumStepLadder_D01_PR_NVD_01_collision_rigid.usd; do
  [ -f "$need" ] || { echo "[lm] missing $need -- install the staged trainer files first"; exit 1; }
done
grep -q "bulb_hand" active_adaptation/envs/mdp/commands/__init__.py || { echo "[lm] commands/__init__.py does not import bulb_hand -- install the staged trainer files first"; exit 1; }
grep -q "def install_free_ladder" active_adaptation/envs/ladder_fiatlux.py || { echo "[lm] ladder_fiatlux.py has no free ladder -- install the staged trainer files first"; exit 1; }

VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
ENVS="${ENVS:-4096}"
if [ "${VRAM:-0}" -lt 15000 ]; then
  echo "[lm] landed on a ${VRAM}MiB card; 4096 envs needs >=15000. Standing down."
  exit 0
fi
ITERS="${ITERS:-12300}"
CLIP=data/motion/ladder_data/${CLIPNAME:-edge87}
LADDER="${LADDER:-free}"; SEED="${SEED:-0}"
FREE=(); [ "$LADDER" = free ] && FREE=(+task.command.ladder_config.free=asset +task.sim.physx_collision_stack="${COLLISION_STACK:-268435456}")   # 256 MiB: the default 64 MiB dropped contacts
[ "$LADDER" = free ] || [ "$LADDER" = fixed ] || { echo "[lm] LADDER must be free or fixed"; exit 1; }
OUT="$ROOT/teacher/$FAMILY-$SLURM_JOB_ID"
mkdir -p "$OUT"

CKPT=$(ls -t "$ROOT"/teacher/"$FAMILY"-[0-9]*/checkpoints/checkpoint_[0-9]*.pt 2>/dev/null | head -1 || true)
# First start: continue the two-handed teacher (lm-edge72, iteration 4,800: 48 of 64 full climbs) from a
# pinned copy, so this job never depends on, or touches, the run that produced it.
SEEDCKPT="${SEEDCKPT:-edge72_checkpoint_4800.pt}"; SEEDIT=$(echo "$SEEDCKPT" | sed -E "s/.*checkpoint_([0-9]+)\.pt/\1/")
if [ -z "$CKPT" ] && [ -f "$ROOT/teacher/keep/$SEEDCKPT" ]; then
  mkdir -p "$ROOT/teacher/$FAMILY-seed"; cp -n "$ROOT/teacher/keep/$SEEDCKPT" "$ROOT/teacher/$FAMILY-seed/checkpoint_$SEEDIT.pt"
  CKPT="$ROOT/teacher/$FAMILY-seed/checkpoint_$SEEDIT.pt"; echo "[lm] no $FAMILY checkpoint yet: continuing $SEEDCKPT"
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

TERMS=()
TIGHT=('+task.reward.tracking.tight_arm_joint_pos_tracking={weight:0.5,sigma:0.15,tolerance:0.03,joint_names:[".*_shoulder_.*_joint",".*_elbow_joint"]}'
       '+task.reward.tracking.tight_arm_pos_tracking={weight:0.5,sigma:0.05,tolerance:0.02,body_names:[".*_elbow_link",".*_wrist_yaw_link"]}'
       '+task.reward.tracking.tight_arm_ori_tracking={weight:0.5,sigma:0.2,tolerance:0.05,body_names:[".*_elbow_link",".*_wrist_yaw_link"]}')
case "$PROFILE" in
  none) ;;
  std)  TERMS=("${TIGHT[@]}") ;;
  bulb) TERMS=("${TIGHT[@]}"
               '+task.reward.tracking.carry_hand_linvel_tracking={weight:0.5,sigma:0.3,tolerance:0.05,body_names:["right_wrist_yaw_link"]}'
               '+task.reward.loco.carry_hand_contact_force_l2={weight:1.0,free_force:15.0,scale:25.0}'
               '+task.reward.loco.carry_hand_accel_l2={weight:0.5,free_accel:5.0,scale:10.0}')
        [ "${CARRY_FORCE_LIMIT:-0}" != 0 ] && TERMS+=("+task.termination.carry_hand_force_exceeded={limit:${CARRY_FORCE_LIMIT},min_steps:1,grace_steps:${CARRY_GRACE:-25}}")
        [ "${CARRY_JUMP_LIMIT:-2.4}" != 0 ] && TERMS+=("+task.termination.carry_hand_velocity_jump={limit:${CARRY_JUMP_LIMIT:-2.4},grace_steps:${CARRY_GRACE:-25}}") ;;
  *) echo "[lm] PROFILE must be none, std or bulb"; exit 1 ;;
esac
if [ -n "${FEET:-}" ]; then
  TERMS+=('+task.reward.tracking.tight_feet_pos_tracking={weight:0.5,sigma:0.06,tolerance:0.005,body_names:[".*_ankle_roll_link"]}'
          '+task.reward.tracking.tight_feet_ori_tracking={weight:0.5,sigma:0.3,tolerance:0.05,body_names:[".*_ankle_roll_link"]}')
fi
[ "${SPEED_LIMIT:-0}" != 0 ] && TERMS+=("+task.termination.body_speed_exceeded={limit:${SPEED_LIMIT}}")
[ -n "${ENV_SPACING:-}" ] && TERMS+=("task.viewer.env_spacing=${ENV_SPACING}")
echo "[lm] robot=$ROBOT profile=$PROFILE terms: ${#TERMS[@]}"; printf '[lm]   %s\n' "${TERMS[@]}"
md5sum active_adaptation/envs/mdp/commands/hdmi/bulb_hand.py "active_adaptation/assets/g1/$ROBOT.urdf"
md5sum active_adaptation/envs/ladder_fiatlux.py active_adaptation/envs/ladder_env.py active_adaptation/envs/mdp/commands/hdmi/command_spawn.py
echo "[lm] family=$FAMILY ladder=$LADDER seed=$SEED node=$(hostname) vram=${VRAM}MiB envs=$ENVS iters=$ITERS clip=$CLIP $(date)"
md5sum $CLIP/motion.npz $CLIP/anchors.json
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
free -g | awk '/^Mem:/{print "[lm] node RAM: "$2" G"}'

exec python scripts/train.py algo=teacher task=teacher "${RESUME[@]}" \
    task.robot.name=g1_fiatlux \
    task.robot.robot_type="$ROBOT" "${TERMS[@]}" \
    task.command.data_path=$CLIP \
    +task.command.ladder_config.kind=fiatlux_step \
    +task.command.ladder_config.anchors_file=$CLIP/anchors.json \
    +task.command.ladder_config.contact_idx=13 \
    '+task.command.contact_target_pos_offset=[[0.0,0.115,0.0],[0.0,-0.115,0.0]]' \
    "task.command.contact_eef_pos_offset=[[0.17,-0.017,0.0],${EEF_R:-[0.17,0.017,0.0]}]" \
    task.randomization.ladder_geometry_randomization.enabled=false \
    +task.randomization.perturb_body_materials.homogeneous=true \
    +task.sim.physx_capacity_scale="${CAPACITY_SCALE:-1.5}" \
    "${FREE[@]}" seed="$SEED" \
    task.max_episode_length=2100 \
    task.action._target_=active_adaptation.envs.mdp.action_refwrist.JointPositionRefWrist \
    task.command._target_=active_adaptation.envs.mdp.commands.hdmi.command_spawn.RobotTrackingSpawn \
    +task.command.start_zero_prob="${START_ZERO:-0.3}" +task.command.floor_steps="${FLOOR_STEPS:-265}" \
    '+task.command.floor_pose_range={x:[-0.05,0.05],y:[-0.05,0.05]}' \
    +task.command.reset_lift="${RESET_LIFT:-0.0}" \
    task.num_envs="$ENVS" algo.max_iters="$ITERS" \
    hydra.run.dir="$OUT" \
    wandb.mode=disabled
