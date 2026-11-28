#!/bin/bash
# Stage 1: tokens, pack, warm start, CPU checks. CPU only; run after setup.sbatch.
# export_warmstart.sh loads the 11 GB checkpoint, hence the memory request.
#SBATCH -J psi0-prep
#SBATCH -p shared
#SBATCH -t 04:00:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -o psi0-koa.%x.%j.out
set -eo pipefail
: "${PSI0_KOA_REPO:?set PSI0_KOA_REPO to the fiatlux checkout before sbatch}"
exec bash "$PSI0_KOA_REPO/scripts/psi0/koa/prep_koa.sh"
