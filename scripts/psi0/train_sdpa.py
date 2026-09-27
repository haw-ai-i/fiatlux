"""Run Psi0's ``scripts/train.py`` with PyTorch SDPA attention instead of flash-attn.

Runs in the Psi0 venv (launched by ``finetune_s06.sh`` through torchrun). Psi0's trainers load
the VLM with ``attn_implementation="flash_attention_2"`` hard-coded, which fails without the
``flash_attn`` package. Its inference path already falls back to SDPA (``Psi0Model.from_pretrained``:
"flash_attention_2 on CUDA (if installed), sdpa on XPU/CPU"), and the trainers never use a
flash-attn-only feature (``data_flatten`` / ``data_packing`` are declared in the model config but
read nowhere), so SDPA computes the same attention. Installing flash-attn into the shared venv
would instead switch the zero-shot server to a different attention kernel on its next restart.
"""

import os
import runpy
import sys

from transformers import Qwen3VLForConditionalGeneration

_original = Qwen3VLForConditionalGeneration.from_pretrained


def _from_pretrained_sdpa(*args, **kwargs):
    if kwargs.get("attn_implementation") == "flash_attention_2":
        kwargs["attn_implementation"] = "sdpa"
    return _original(*args, **kwargs)


Qwen3VLForConditionalGeneration.from_pretrained = _from_pretrained_sdpa

train_py = os.path.join(os.environ["PSI0_REPO"], "scripts", "train.py")
sys.argv = [train_py] + sys.argv[1:]
runpy.run_path(train_py, run_name="__main__")
