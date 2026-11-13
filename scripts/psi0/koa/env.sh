# Shared environment for the Psi-0 S06 fine-tune on Koa (UH Manoa's HPC cluster).
#
# RUN EVERYTHING IN A BATCH JOB. Koa's login nodes mount $HOME, /tmp and Lustre noexec and have
# no git; compute nodes have exec on all three plus git and full internet. `module load` does
# not alter python3 on either node type, so uv fetches its own standalone 3.11.
#
# Every artifact lives in Lustre scratch, never $HOME (50 GB quota vs 4.3 PB scratch).
#
#   PSI0_KOA_ROOT   workspace for venvs, checkpoints, data, runs   (default: scratch/$USER/psi0)
#   PSI0_KOA_REPO   this fiatlux checkout       (default: derived from this file's location)

export REPO="${PSI0_KOA_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
export ROOT="${PSI0_KOA_ROOT:-/mnt/lustre/koa/scratch/$USER/psi0}"

export TOOLS=$ROOT/tools
export PSI0_REPO=$TOOLS/Psi0
export CKPTS=$TOOLS/psi0_checkpoints
export SONIC_DIR=$TOOLS/sonic_models
export FT=$ROOT/ft
export UV=$ROOT/boot/bin/uv
export HF_HOME=$ROOT/hf
export HF_HUB_CACHE=$ROOT/hf/hub
export PATH=$ROOT/boot/bin:$PATH

# Keep uv's package cache off $HOME and on the same filesystem as the venvs it populates, which
# also silences uv's "failed to hardlink, falling back to full copy" warning.
export UV_CACHE_DIR=$ROOT/uvcache
export UV_LINK_MODE=hardlink
