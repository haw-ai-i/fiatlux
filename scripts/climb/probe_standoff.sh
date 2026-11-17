#!/usr/bin/env bash
# Sweep the spawn standoff on ONE checkpoint, across SEVERAL pinned layouts, in the real S02 env.
#
# Why this is a sibling of probe_checkpoints.sh rather than a flag on it: that script polls Koa
# for hours and bash re-reads a running script by byte offset, so editing it in place while a
# prober is live executes garbage from the middle of a line. This also asks a different question
# -- one checkpoint, many geometries, instead of many checkpoints, one geometry.
#
# The question: across 34 probe episodes at iters 3000-7800 the pelvis never rose more than 36 mm
# above its 0.790 m spawn height (one tread rise is 234 mm), so the policy never leaves the floor.
# The reference is anchored to the LADDER while the robot is staged ~285 mm further out, so the
# clip's first move reaches for a rung that is not where the robot's hands are. record_climb.py
# measures that offset directly ([ALIGN ], the align_fwd_m column) and --standoff_delta closes it.
#
# LAYOUT SEEDS ARE THE POINT. The Replace layout (ladder pose and yaw, fixture mount) is drawn in
# __post_init__ from OS entropy unless set_layout_seed() is called first, so an unseeded sweep
# compares scenes, not standoffs -- the first version of this script did exactly that and its
# ladder moved 0.6 m and 100 degrees between consecutive runs. Every delta is now run against the
# same set of pinned layouts, so a difference between deltas is a difference in standoff.
#
#   bash scripts/climb/probe_standoff.sh <checkpoint.pt> <out_dir> [motion] [deltas] [seeds] [modes]
#
# <deltas> metres to move the spawn TOWARD the ladder; keep 0 in the list as the control.
# <seeds>  layout seeds; each delta is run against every one of them.
# <modes>  "mt", "raw", or "raw,mt".
set -uo pipefail

CKPT="${1:?usage: probe_standoff.sh <checkpoint.pt> <out_dir> [motion] [deltas] [seeds] [modes]}"
OUT="${2:?}"
MOTION="${3:-/home/yujin-chen/psi0_ft/climb/topout}"
DELTAS="${4:-0,0.20,0.35}"
SEEDS="${5:-0,1,2}"
MODES="${6:-mt}"

REPO=/home/yujin-chen/robotica_project/fiatlux/fiatlux-climb
PY=/home/yujin-chen/robotica_project/fiatlux/fiatlux/.venv/bin/python
CSV="$OUT/standoff.csv"

[ -f "$CKPT" ] || { echo "no such checkpoint: $CKPT"; exit 1; }

# One Kit process at a time -- two overlapping runs OOM this 30 GB box and the kernel takes
# VS Code with them. Check by explicit PID listing, never pkill -f (that pattern matches this
# script's own command line). Note SIGTERM does not stop a Kit process; these need kill -9.
busy=$(ps -eo pid,args | grep -E 'record_climb\.py|probe_checkpoints\.sh' | grep -v grep || true)
if [ -n "$busy" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "[standoff] another climb process is live; a second Kit run would OOM the box:"
  echo "$busy" | sed 's/^/           /'
  echo "[standoff] stop it by its explicit PID (kill -9 <pid>), then rerun. FORCE=1 overrides."
  exit 1
fi

mkdir -p "$OUT/logs"
[ -f "$CSV" ] || echo "layout_seed,delta_m,standoff_m,match_torque,steps,seconds,pelvis_max_m,align_fwd_m,reason" > "$CSV"
echo "[standoff] $(basename "$CKPT") ; motion $MOTION"
echo "[standoff] deltas $DELTAS x seeds $SEEDS x modes $MODES"

run_one() {           # $1 = seed, $2 = delta, $3 = "" or "--match_torque"
  local tag="mt"; [ -z "$3" ] && tag="raw"
  local log="$OUT/logs/s$1_d$2_${tag}.log"
  OMNI_KIT_ACCEPT_EULA=YES timeout 900 "$PY" -u "$REPO/scripts/climb/record_climb.py" \
      --headless --checkpoint "$CKPT" --motion "$MOTION" \
      --layout_seed "$1" --standoff_delta "$2" \
      --max_steps 1200 --print_every 400 $3 ${EXTRA:-} > "$log" 2>&1
  local steps reason pel stand fwd
  steps=$(grep -oE '^\[RESULT\] [0-9]+ steps' "$log" | grep -oE '[0-9]+' | head -1)
  pel=$(grep -oE 'highest pelvis [0-9.]+' "$log" | grep -oE '[0-9.]+' | head -1)
  stand=$(grep -oE 'standoff [0-9.]+ m' "$log" | grep -oE '[0-9.]+' | head -1)
  fwd=$(grep -oE 'at t=0 is [+-][0-9.]+ m forward' "$log" | grep -oE '[+-][0-9.]+' | head -1)
  reason=$(grep -oE "reason=\[[^]]*\]" "$log" | head -1 | sed "s/^reason=\[//; s/\]$//; s/'//g; s/, /+/g")
  steps="${steps:-0}"; pel="${pel:-nan}"; stand="${stand:-nan}"; fwd="${fwd:-nan}"; reason="${reason:-none}"
  local mt=0; [ -n "$3" ] && mt=1
  printf '%s,%s,%s,%s,%s,%.2f,%s,%s,%s\n' "$1" "$2" "$stand" "$mt" "$steps" \
      "$(awk "BEGIN{print $steps/50}")" "$pel" "$fwd" "$reason" >> "$CSV"
  echo "[standoff] seed $1 delta $2 $tag -> $steps steps, pelvis $pel m, ref $fwd m fwd, $reason"
}

IFS=',' read -ra DS <<< "$DELTAS"
IFS=',' read -ra SS <<< "$SEEDS"
IFS=',' read -ra MS <<< "$MODES"
for s in "${SS[@]}"; do
  for d in "${DS[@]}"; do
    for m in "${MS[@]}"; do
      if [ "$m" = "mt" ]; then run_one "$s" "$d" "--match_torque"; else run_one "$s" "$d" ""; fi
    done
  done
done

echo "[standoff] done; $(($(wc -l < "$CSV") - 1)) rows in $CSV"
