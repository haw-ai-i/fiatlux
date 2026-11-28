"""CPU check before any GPU time: does the warm-start action header load into the model the trainer
builds, with no missing, unexpected, or mis-shaped keys? (docs/psi0_finetune.md #13: the trainer
loads it with ``strict=False``, which would silently leave a mismatched block random.)

Runs in the Psi0 venv with the exact training arguments: ``CHECK_INIT=1 scripts/psi0/finetune_s06.sh``.
"""

import importlib
import sys

import torch
import tyro
from dotenv import load_dotenv
from safetensors.torch import load_file
from transformers import AutoConfig, Qwen3VLForConditionalGeneration

load_dotenv()
module = importlib.import_module(f"psi.config.train.{sys.argv[1]}")
cfg = tyro.cli(module.DynamicLaunchConfig, config=(tyro.conf.ConsolidateSubcommandArgs,), args=sys.argv[2:])
from psi.models.psi0 import Psi0Model  # noqa: E402

vc = AutoConfig.from_pretrained(cfg.model.model_name_or_path)
vc._attn_implementation = "sdpa"
with torch.device("meta"):
    vlm = Qwen3VLForConditionalGeneration(vc)
model = Psi0Model(model_cfg=cfg.model, vlm_model=vlm)
sd = load_file(f"{cfg.model.pretrained_action_header_path}/action_header.safetensors")
own = model.action_header.state_dict()
missing = sorted(set(own) - set(sd))
unexpected = sorted(set(sd) - set(own))
shape_bad = [k for k in set(own) & set(sd) if tuple(own[k].shape) != tuple(sd[k].shape)]
print(f"header params in model: {len(own)}, in ckpt: {len(sd)}")
print(f"missing ({len(missing)}): {missing[:10]}")
print(f"unexpected ({len(unexpected)}): {unexpected[:10]}")
print(f"shape mismatches ({len(shape_bad)}): {shape_bad[:10]}")
print("trainable header params:", sum(v.numel() for v in own.values()) / 1e6, "M")
