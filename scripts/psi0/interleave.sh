#!/usr/bin/env bash
# The reported zero-shot protocol: Psi-0 and a same-robot `zero` floor (Dex3, current code),
# alternated one seed at a time so a cut-off at any point leaves matched seeds of both
# (docs/psi0_baseline.md #16, #18). One Isaac instance at a time. Needs the :8014 server up
# (scripts/psi0/serve.sh); the zero runs do not use it.
#
#   scripts/psi0/interleave.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PSI0_ROOT="${PSI0_ROOT:-logs/runs/psi0_zeroshot}"
ZERO_ROOT="${ZERO_ROOT:-logs/runs/zero_dex3}"
mkdir -p "$PSI0_ROOT" "$ZERO_ROOT"
run() {
    echo "=== $1 seed $2 start $(date)"
    PSI0_SPEC=$1 PSI0_OUT=$3 SEEDS="$2" scripts/psi0/sweep.sh >> "$3/sweep.log" 2>&1
    echo "=== $1 seed $2 end $(date)"
}
for seed in ${SEEDS:-0 1 2 3}; do
    run psi0 "$seed" "$PSI0_ROOT"
    run zero "$seed" "$ZERO_ROOT"
done
"${PY:-.venv/bin/python}" scripts/psi0/summarize.py "$PSI0_ROOT"
"${PY:-.venv/bin/python}" scripts/psi0/summarize.py "$ZERO_ROOT"
