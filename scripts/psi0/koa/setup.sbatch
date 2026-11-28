#!/bin/bash
# Stage 0: everything scripts/psi0/setup.sh fetches, plus the pyarrow pin. CPU only.
# Submit with PSI0_KOA_REPO and HF_TOKEN in the environment; see README.md.
#SBATCH -J psi0-setup
#SBATCH -p shared
#SBATCH -t 06:00:00
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -o psi0-koa.%x.%j.out
set -eo pipefail
: "${PSI0_KOA_REPO:?set PSI0_KOA_REPO to the fiatlux checkout before sbatch}"
exec bash "$PSI0_KOA_REPO/scripts/psi0/koa/setup_koa.sh"
