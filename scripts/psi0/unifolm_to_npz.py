"""Dump Psi-0's UnifoLM SONIC LeRobot pack to one ``.npz`` per episode (runs in the Psi0 venv: pandas).

Input: an unzipped ``USC-PSI-Lab/psi-data:sonic/unifolm_sonic_lerobot_val`` (``scripts/psi0/setup.sh``
step 4). Output keys, read by ``verify_sonic_tracking.py`` and ``unifolm_reencode.py``:
``tokens``/``token`` (N, 64) ``action.body_token``, ``token_v1_1`` (N, 64), ``action`` (N, 36;
``[:14]`` = Dex3 targets), ``state`` (N, 43; legs, waist, arms, hands), ``names`` (the 43 state
column names from ``meta/info.json``), ``timestamp`` (N,).

    .venv-psi/bin/python scripts/psi0/unifolm_to_npz.py ~/psi0_ft/unifolm_sonic_lerobot_val ~/psi0_ft/unifolm_npz
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd


def main() -> None:
    root, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(root, "meta/info.json")) as f:
        info = json.load(f)
    names = info["features"]["observation.state"]["names"]
    for pq in sorted(glob.glob(os.path.join(root, "data/chunk-*/*.parquet"))):
        df = pd.read_parquet(pq)
        token = np.stack(df["action.body_token"].to_numpy()).astype(np.float32)
        np.savez(
            os.path.join(out, os.path.basename(pq).replace(".parquet", ".npz")),
            tokens=token,
            token=token,
            token_v1_1=np.stack(df["action.body_token_v1_1"].to_numpy()).astype(np.float32),
            action=np.stack(df["action"].to_numpy()).astype(np.float32),
            state=np.stack(df["observation.state"].to_numpy()).astype(np.float32),
            names=np.array(names),
            timestamp=df["timestamp"].to_numpy().astype(np.float64),
        )
        on_grid = np.allclose(token * 16, np.round(token * 16), atol=1e-4)
        print(f"{os.path.basename(pq)}: {len(df)} rows, token on the 1/16 FSQ grid: {on_grid}")


if __name__ == "__main__":
    main()
