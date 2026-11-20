#!/usr/bin/env bash
# Poll Koa for new g1_fiatlux-actuator teacher checkpoints and score each one in the real S02 env.
#
# A sibling of probe_checkpoints.sh, not a flag on it, for two reasons. Bash re-reads a running
# script by byte offset, so editing that one while a prober is live executes garbage from the
# middle of a line. And this asks a different question of a different teacher:
#
#   * NO --match_torque. That conversion exists to make OUR actuators reproduce LADDERMAN's
#     torque, and the whole point of the g1_fiatlux teacher is that it was trained on ours --
#     so converting now would push the targets the wrong way. Raw is the correct mode here and
#     the only one this script runs.
#   * SEVERAL LAYOUT SEEDS PER CHECKPOINT. probe_checkpoints.sh scores n=1 per checkpoint, and
#     layout dominated survival in every sweep so far (seed 0 gave 251/349/165 steps where seed 1
#     gave 611/138/163 on the same checkpoints). A one-layout curve is a sample of the layout
#     draw, not a trend in the policy.
#
# What the columns mean. `gain_mm` is pelvis_max minus the pelvis height the robot SPAWNED at,
# which is the number that has never moved: one tread rise is 234 mm and the best of 34 overnight
# episodes managed 36 mm. `steps` is clip progress, and the clip is authored at 120 fps and
# resampled to 50 Hz, so the 2253-frame reference is 939 steps long in these units, not 2253:
#
#     step 119  reference pelvis +234 mm   one tread rise
#     step 281                   +468 mm   two
#     step 459                   +702 mm   three
#     step 939                  +1179 mm   end of clip, on the platform
#
# So an episode only has to last ~2.4 s before the reference has already asked for a full tread.
# Both columns are reported because neither alone is the answer: steps say how far into the clip
# the robot stayed upright, gain_mm says whether any of that was upward.
#
# One Isaac process at a time, strictly sequential: two overlapping Kit runs OOM this 30 GB box
# and the kernel takes VS Code with them.
#
#   bash scripts/climb/probe_fx_checkpoints.sh <remote_ckpt_dir> <out_dir> [motion] [seeds] [stride] [hours]
#
# <remote_ckpt_dir>  absolute path ON KOA, e.g. /mnt/lustre/.../teacher/topout-fx-15477480/checkpoints
# <seeds>            layout seeds; every probed checkpoint is run against every one of them.
# <stride>           probe every Nth checkpoint (checkpoints land every 300 iters). 1 = all.
set -uo pipefail

REMOTE="${1:?usage: probe_fx_checkpoints.sh <remote_ckpt_dir> <out_dir> [motion] [seeds] [stride] [hours]}"
OUT="${2:?}"
MOTION="${3:-/home/yujin-chen/psi0_ft/climb/topout}"
SEEDS="${4:-0,1,2}"
STRIDE="${5:-2}"
HOURS="${6:-20}"

REPO=/home/yujin-chen/robotica_project/fiatlux/fiatlux-climb
PY=/home/yujin-chen/robotica_project/fiatlux/fiatlux/.venv/bin/python
CSV="$OUT/probe_fx.csv"

# Check by explicit PID listing, never pkill -f (that pattern matches this script's own command
# line). SIGTERM does not stop a Kit process; a stuck one needs kill -9.
busy=$(ps -eo pid,args | grep -E 'record_climb\.py|probe_checkpoints\.sh|probe_standoff\.sh' \
         | grep -v grep | grep -v "^ *$$ " || true)
if [ -n "$busy" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "[fxprobe] another climb process is live; a second Kit run would OOM the box:"
  echo "$busy" | sed 's/^/          /'
  echo "[fxprobe] stop it by its explicit PID (kill -9 <pid>), then rerun. FORCE=1 overrides."
  exit 1
fi

mkdir -p "$OUT/ckpt" "$OUT/logs"
[ -f "$CSV" ] || echo "iter,layout_seed,steps,seconds,pelvis0_m,pelvis_max_m,gain_mm,align_fwd_m,reason" > "$CSV"

echo "[fxprobe] $REMOTE -> $OUT"
echo "[fxprobe] motion $MOTION ; seeds $SEEDS ; stride $STRIDE ; budget ${HOURS}h ; raw mode only"
END=$(( $(date +%s) + HOURS*3600 ))

run_one() {           # $1 = local ckpt, $2 = iter, $3 = layout seed
  local log="$OUT/logs/iter$2_s$3.log"
  OMNI_KIT_ACCEPT_EULA=YES timeout 900 "$PY" -u "$REPO/scripts/climb/record_climb.py" \
      --headless --checkpoint "$1" --motion "$MOTION" \
      --layout_seed "$3" --max_steps 1200 --print_every 400 ${EXTRA:-} > "$log" 2>&1
  local steps reason pel pel0 fwd gain
  steps=$(grep -oE '^\[RESULT\] [0-9]+ steps' "$log" | grep -oE '[0-9]+' | head -1)
  pel=$(grep -oE 'highest pelvis [0-9.]+' "$log" | grep -oE '[0-9.]+' | head -1)
  # the spawn pelvis height, from the [ALIGN ] robot-root triple: "robot root [x, y, z]"
  pel0=$(grep -oE 'robot root \[[^]]*\]' "$log" | head -1 | tr -d '[]' | awk -F', ' '{print $3}')
  fwd=$(grep -oE 'at t=0 is [+-][0-9.]+ m forward' "$log" | grep -oE '[+-][0-9.]+' | head -1)
  reason=$(grep -oE "reason=\[[^]]*\]" "$log" | head -1 | sed "s/^reason=\[//; s/\]$//; s/'//g; s/, /+/g")
  steps="${steps:-0}"; pel="${pel:-nan}"; pel0="${pel0:-nan}"; fwd="${fwd:-nan}"; reason="${reason:-none}"
  if [ "$pel" = "nan" ] || [ "$pel0" = "nan" ]; then gain="nan"; else
    gain=$(awk "BEGIN{printf \"%.1f\", ($pel-$pel0)*1000}"); fi
  printf '%s,%s,%s,%.2f,%s,%s,%s,%s,%s\n' "$2" "$3" "$steps" \
      "$(awk "BEGIN{print $steps/50}")" "$pel0" "$pel" "$gain" "$fwd" "$reason" >> "$CSV"
  echo "[fxprobe] iter $2 seed $3 -> $steps steps ($(awk "BEGIN{printf \"%.1f\", $steps/50}")s), gain ${gain} mm, $reason"
}

IFS=',' read -ra SS <<< "$SEEDS"
declare -A SEEN
while [ "$(date +%s)" -lt "$END" ]; do
  mapfile -t NEW < <(ssh koa "ls -1 $REMOTE/*.pt 2>/dev/null" 2>/dev/null \
                       | grep -oE 'checkpoint_[0-9]+\.pt' | sort -t_ -k2 -n)
  for f in "${NEW[@]:-}"; do
    [ -z "$f" ] && continue
    [ -n "${SEEN[$f]:-}" ] && continue
    SEEN[$f]=1
    it=$(echo "$f" | grep -oE '[0-9]+')
    # stride is counted in checkpoints, which land every 300 iterations
    if [ "$STRIDE" -gt 1 ] && [ $(( (it / 300) % STRIDE )) -ne 0 ]; then
      echo "[fxprobe] skip iter $it (stride $STRIDE)"; continue
    fi
    local_ck="$OUT/ckpt/$f"
    scp -q "koa:$REMOTE/$f" "$local_ck" 2>/dev/null || { echo "[fxprobe] scp failed for $f"; unset "SEEN[$f]"; continue; }
    for s in "${SS[@]}"; do run_one "$local_ck" "$it" "$s"; done
    rm -f "$local_ck"          # ~10 MB each; the CSV is the artifact
  done
  sleep 420
done
echo "[fxprobe] budget reached; $(($(wc -l < "$CSV") - 1)) rows in $CSV"
