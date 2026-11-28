#!/usr/bin/env bash
# Split the released Psi-0 SONIC run checkpoint into the warm start the S06 fine-tune reads
# (docs/psi0_finetune.md #9): <out>/model.safetensors (VLM, bf16) + HF config/processor files, and
# <out>/action_header.safetensors (fp32, as stored). Uses Psi0's own scripts/export_psi0_ckpt.py.
# The HF files come from the post-trained VLM the checkpoint was fine-tuned from
# (psi0/postpre.sonic1.0.unifolm.2609092156.40k, weights excluded; scripts/psi0/setup.sh step 2).
#
#   scripts/psi0/export_warmstart.sh
set -euo pipefail
PSI0_REPO="${PSI0_REPO:-$HOME/tools/Psi0}"
RUN="${RUN:-$HOME/tools/psi0_checkpoints/psi0/sonic-checkpoints/multi-task.psi-dream.2609092156}"
HF_FROM="${HF_FROM:-$HOME/psi0_ft/hf_from}"
OUT="${OUT:-$HOME/psi0_ft/init/psi_dream_40k}"
cd "$PSI0_REPO"
exec .venv-psi/bin/python scripts/export_psi0_ckpt.py "$RUN/checkpoints/ckpt_40000" "$OUT" \
    --hf-from "$HF_FROM" --dtype bf16
