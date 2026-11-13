"""CPU dry run of the Psi-0 fine-tune data path: parse the training args, build the dataset, check samples.

Runs in the Psi0 venv via ``DRY_RUN=1 scripts/psi0/finetune_s06.sh``. No GPU, no model: it parses the
exact argument list the trainer gets (``psi.config.train.finetune_sonic_psi0_config``), loads the
pack through Psi0's own LeRobot wrapper and transforms, and prints what one training sample looks
like -- the shapes, the normalized ranges, and which action dims the loss mask keeps.
"""

import importlib
import sys

import numpy as np
import tyro
from dotenv import load_dotenv
from transformers import AutoProcessor

load_dotenv()
module = importlib.import_module(f"psi.config.train.{sys.argv[1]}")
cfg = tyro.cli(module.DynamicLaunchConfig, config=(tyro.conf.ConsolidateSubcommandArgs,), args=sys.argv[2:])
processor = AutoProcessor.from_pretrained(cfg.model.model_name_or_path)
data = cfg.data(split="train", transform_kwargs=dict(vlm_processor=processor))
print(f"train samples: {len(data)}")
field = cfg.data.transform.field
print(f"action bounds: {len(field.action_min)} dims; state bounds: {len(field.state_min)} dims")
for i in (0, len(data) // 2, len(data) - 1):
    s = data[i]
    keys = sorted(s.keys())
    acts, states, mask = (np.asarray(s[k]) for k in ("actions", "states", "actions_mask"))
    kept = {"token": mask[:, :64].mean(), "hand": mask[:, 64:78].mean(), "neck": mask[:, 78:].mean()}
    print(
        f"[{i}] keys={keys}\n"
        f"    instruction={s.get('instruction')!r}\n"
        f"    states {states.shape} range [{states.min():+.2f},{states.max():+.2f}] neck={states[..., 43:45].ravel()}\n"
        f"    actions {acts.shape} range [{acts.min():+.2f},{acts.max():+.2f}]\n"
        f"    mask {mask.shape} kept: {', '.join(f'{k} {v:.2f}' for k, v in kept.items())}"
    )
