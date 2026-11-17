#!/usr/bin/env bash
# Poll Koa for new teacher checkpoints and score each one in the real S02 env.
#
# Why this exists: `train/stats` prints every 950 iterations (~1.6 h), but checkpoints land every
# 300 (~31 min), so checkpoints are the faster signal by 3x. And training success is measured in
# LadderMan's OWN env -- it says nothing about whether the policy survives in S02, which is the
# only question that matters for a bag.
#
# One Isaac process at a time, strictly sequential: two overlapping Kit runs OOM this box.
#
#   bash scripts/climb/probe_checkpoints.sh <job_id> <out_dir> [motion_dir] [hours]
set -uo pipefail

JOB="${1:?usage: probe_checkpoints.sh <job_id> <out_dir> [motion_dir] [hours]}"
OUT="${2:?}"
MOTION="${3:-/home/yujin-chen/psi0_ft/climb/topout}"
HOURS="${4:-13}"

REPO=/home/yujin-chen/robotica_project/fiatlux/fiatlux-climb
PY=/home/yujin-chen/robotica_project/fiatlux/fiatlux/.venv/bin/python
REMOTE="/mnt/lustre/koa/scratch/yujin31/ladderman/teacher/topout-${JOB}/checkpoints"
CSV="$OUT/probe.csv"
mkdir -p "$OUT/ckpt" "$OUT/logs"
[ -f "$CSV" ] || echo "iter,match_torque,steps,seconds,pelvis_max_m,reason" > "$CSV"

echo "[probe] job $JOB -> $OUT ; motion $MOTION ; budget ${HOURS}h"
END=$(( $(date +%s) + HOURS*3600 ))

run_one() {           # $1 = local ckpt, $2 = iter, $3 = "" or "--match_torque"
  local tag="mt"; [ -z "$3" ] && tag="raw"
  local log="$OUT/logs/iter${2}_${tag}.log"
  OMNI_KIT_ACCEPT_EULA=YES timeout 900 "$PY" -u "$REPO/scripts/climb/record_climb.py" \
      --headless --checkpoint "$1" --motion "$MOTION" \
      --max_steps 1200 --print_every 400 $3 > "$log" 2>&1
  local steps reason pel
  steps=$(grep -oE '^\[RESULT\] [0-9]+ steps' "$log" | grep -oE '[0-9]+' | head -1)
  pel=$(grep -oE 'highest pelvis [0-9.]+' "$log" | grep -oE '[0-9.]+' | head -1)
  reason=$(grep -oE "reason=\[[^]]*\]" "$log" | head -1 | sed "s/^reason=\[//; s/\]$//; s/'//g; s/, /+/g")
  steps="${steps:-0}"; pel="${pel:-nan}"; reason="${reason:-none}"
  local mt=0; [ -n "$3" ] && mt=1
  printf '%s,%s,%s,%.2f,%s,%s\n' "$2" "$mt" "$steps" \
      "$(awk "BEGIN{print $steps/50}")" "$pel" "$reason" >> "$CSV"
  echo "[probe] iter $2 $tag -> $steps steps ($(awk "BEGIN{printf \"%.2f\", $steps/50}")s), pelvis $pel, $reason"
}

declare -A SEEN
while [ "$(date +%s)" -lt "$END" ]; do
  mapfile -t NEW < <(ssh koa "ls -1 $REMOTE/*.pt 2>/dev/null" 2>/dev/null | grep -oE 'checkpoint_[0-9]+\.pt' | sort -t_ -k2 -n)
  for f in "${NEW[@]:-}"; do
    [ -z "$f" ] && continue
    [ -n "${SEEN[$f]:-}" ] && continue
    SEEN[$f]=1
    it=$(echo "$f" | grep -oE '[0-9]+')
    local_ck="$OUT/ckpt/$f"
    scp -q "koa:$REMOTE/$f" "$local_ck" 2>/dev/null || { echo "[probe] scp failed for $f"; unset "SEEN[$f]"; continue; }
    run_one "$local_ck" "$it" ""
    run_one "$local_ck" "$it" "--match_torque"
    rm -f "$local_ck"          # ~10 MB each; the CSV is the artifact
  done
  # a checkpoint lands every ~31 min; poll well inside that
  sleep 420
done
echo "[probe] budget reached; $(($(wc -l < "$CSV") - 1)) rows in $CSV"
