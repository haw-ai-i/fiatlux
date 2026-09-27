#!/usr/bin/env bash
# Psi-0 subtask sweep: each subtask x seed, one episode, recorded as a bag and scored by
# scripts/score.py. Seed 0 also keeps the robot's ego-camera video (the frames Psi-0 sees).
# Resumable -- a run that already has score.json is skipped. Needs the Psi-0 server running
# (scripts/psi0/serve.sh) and the Dex3 robot (psi0 refuses anything else).
#
#   scripts/psi0/sweep.sh                         # all 12 subtasks, seeds 0-3
#   SEEDS="0" scripts/psi0/sweep.sh S06 S08       # a subset
#   PSI0_OUT=logs/runs/psi0_ft PSI0_SPEC=psi0:localhost:8015 scripts/psi0/sweep.sh S06
#
# Roll up afterwards, one score_subtasks.py call per seed:
#   python scripts/score_subtasks.py "$PSI0_OUT"/*/seed0/score.json
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$(pwd)/source/fiatlux_task:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES

PY="${PY:-.venv/bin/python}"
OUT="${PSI0_OUT:-logs/runs/psi0_zeroshot}"
SPEC="${PSI0_SPEC:-psi0}"
SEEDS="${SEEDS:-0 1 2 3}"
# Hard wall-clock cap per run: a 120 s episode is 6000 env steps plus ~240 server round trips.
RUN_TIMEOUT_S="${RUN_TIMEOUT_S:-2400}"

ALL=(S01-MoveLadder S02-ClimbLadder S03-RemoveOldBulb S04-DescendWithBulb S05-CarryBulbToDisposal
     S06-DisposeBulb S07-ApproachNewBulb S08-GrabNewBulb S09-CarryBulbToLadder S10-ClimbWithBulb
     S11-ScrewInBulb S12-ClimbDown)
if [ $# -gt 0 ]; then
    SUBTASKS=()
    for want in "$@"; do
        for s in "${ALL[@]}"; do [[ "$s" == "$want"* ]] && SUBTASKS+=("$s"); done
    done
else
    SUBTASKS=("${ALL[@]}")
fi

mkdir -p "$OUT"
# Seed-major: every subtask for one seed before the next seed, so a sweep cut short still leaves
# complete per-seed roll-ups (score_subtasks.py scores one seed's twelve results at a time).
for seed in $SEEDS; do
    for sub in "${SUBTASKS[@]}"; do
        task="FIATLUX-${sub}-v0"
        dir="$OUT/$task/seed$seed"
        if [ -f "$dir/score.json" ]; then
            echo "[skip] $task seed$seed"
            continue
        fi
        rm -rf "$dir" && mkdir -p "$dir"
        record=(--record bag)
        [ "$seed" = "0" ] && record=(--record both --cam ego --video_length 6000)
        echo "[run] $task seed$seed $(date -u +%H:%M:%S)"
        t0=$(date +%s)
        timeout -k 60 "$RUN_TIMEOUT_S" "$PY" -u scripts/record_run.py --task "$task" --policy "$SPEC" \
            --robot dex3 --episodes 1 --seed "$seed" "${record[@]}" --enable_cameras --headless \
            --out "$dir" > "$dir/record.log" 2>&1
        rc=$?
        if [ $rc -ne 0 ] || [ ! -f "$dir/meta.json" ]; then
            echo "[FAIL record] $task seed$seed rc=$rc ($(( $(date +%s) - t0 ))s; see $dir/record.log)" | tee -a "$OUT/failures.log"
            continue
        fi
        "$PY" -u scripts/score.py "$dir" --output "$dir/score.json" > "$dir/score.log" 2>&1 \
            || { echo "[FAIL score] $task seed$seed (see $dir/score.log)" | tee -a "$OUT/failures.log"; continue; }
        echo "[ok] $task seed$seed $(( $(date +%s) - t0 ))s"
    done
done
echo "[done] psi0 sweep $(date -u +%H:%M:%S)"
