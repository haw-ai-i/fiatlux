"""Validate encode_sonic_tokens.py's encoder layout against Psi-0's own retargeted tokens.

Re-encodes the joint trajectories of Psi-0's UnifoLM SONIC pack with the same g1-mode encoder input
``encode_sonic_tokens.py`` builds, and reports the fraction of token dims equal to the
``action.body_token`` Psi0 stored, against a shuffled-frame control (docs/psi0_finetune.md #3:
66-67 % vs 28 %). Runs in the fiatlux venv from the repo root, on ``unifolm_to_npz.py``'s output:

    python scripts/psi0/unifolm_reencode.py ~/psi0_ft/unifolm_npz
"""

import glob
import sys

import numpy as np

sys.path.insert(0, "scripts/psi0")
sys.path.insert(0, "source/fiatlux_task")
from encode_sonic_tokens import SonicEncoder, encoder_inputs  # noqa: E402
from fiatlux_task.groot import SONIC_JOINT_NAMES  # noqa: E402

enc = SonicEncoder()
rng = np.random.default_rng(0)


def run(variant, state, names, fps_src=30.0):
    col = {n: i for i, n in enumerate(names)}
    q30 = state[:, [col[n] for n in SONIC_JOINT_NAMES]]
    t30 = np.arange(len(q30)) / fps_src
    if variant["rate"] == 50:
        t50 = np.arange(0, t30[-1] + 1e-9, 1 / 50)
        q = np.stack([np.interp(t50, t30, q30[:, j]) for j in range(29)], 1)
        dt = 1 / 50
    else:
        q, dt = q30, 1 / fps_src
    v = np.gradient(q, dt, axis=0) if variant["vel"] == "grad" else np.zeros_like(q)
    quat = np.tile([1.0, 0, 0, 0], (len(q), 1))
    idx = np.arange(0, len(q30), 6)
    if variant["rate"] == 50:
        src = np.minimum(np.round(idx * 50 / fps_src).astype(int), len(q) - 1)
    else:
        src = idx
    tok = enc(encoder_inputs(q, v, quat)[src])
    return idx, tok


variants = [
    {"rate": 50, "vel": "grad"},
    {"rate": 50, "vel": "zero"},
    {"rate": 30, "vel": "grad"},
]
files = sorted(glob.glob(sys.argv[1] + "/*.npz"))
for var in variants:
    eq, mad, eq_ctrl = [], [], []
    for f in files:
        d = np.load(f)
        idx, tok = run(var, d["state"], list(d["names"]))
        ref = d["token"][idx]
        eq.append((np.abs(tok - ref) < 1e-4).mean())
        mad.append(np.abs(tok - ref).mean())
        perm = rng.permutation(len(ref))
        eq_ctrl.append((np.abs(tok - ref[perm]) < 1e-4).mean())
    print(
        var,
        f"dims equal {np.mean(eq):.3f} (shuffled-frame control {np.mean(eq_ctrl):.3f}), mean |diff| {np.mean(mad):.4f}",
    )
