# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Explain one episode's ``gate_progress``: which success conjuncts held, when, and together.

Reads a bag (``run.h5``) and prints the conjuncts true at reset (the normalization baseline), the
row where the most were true at once (what ``scripts/score.py`` credits), when each was first
true, and the tracked bulb's height/speed. This is how the S06 "released into the crate but
scored like a miss" case was diagnosed: ``old_bulb_in_bin`` held for one step while the bulb was
still moving at ~3.9 m/s, then ``old_bulb_struck`` (limit ``BULB_IMPACT_SPEED_LIMIT`` = 2.4 m/s)
ended the episode before ``object_at_rest`` could join it. The last bag row is recorded after the
env's auto-reset, so it is excluded.

    python scripts/psi0/explain_gate.py logs/runs/psi0_ft_s06/FIATLUX-S06-DisposeBulb-v0/seed2/run.h5 \
        [--bulb old_bulb]
"""

import argparse

import h5py
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("bags", nargs="+")
    p.add_argument("--bulb", default="old_bulb", help="payload whose height/speed to report")
    args = p.parse_args()
    for path in args.bags:
        ep = h5py.File(path, "r")["data/demo_0"]
        cols = sorted(k for k in ep if k.startswith("gate_"))
        g = np.stack([ep[c][:].astype(bool).reshape(-1) for c in cols], 1)[:-1]  # drop post-reset row
        names = [c[5:] for c in cols]
        counts = g.sum(1)
        at_reset, best = int(counts[0]), int(counts.max())
        n = len(cols)
        prog = 1.0 if at_reset >= n else max(0.0, (best - at_reset) / (n - at_reset))
        k = int(np.argmax(counts))
        print(f"== {path}")
        print(
            f"   {len(counts)} steps; at reset {at_reset}/{n}; best {best}/{n} at step {k} -> gate_progress {prog:.3f}"
        )
        print("   at reset:", [nm for nm, v in zip(names, g[0]) if v])
        print(f"   best row {k}:", [nm for nm, v in zip(names, g[k]) if v])
        print(
            "   first true:", {nm: (int(np.argmax(g[:, i])) if g[:, i].any() else None) for i, nm in enumerate(names)}
        )
        if f"{args.bulb}_pos" in ep:
            z = ep[f"{args.bulb}_pos"][:-1, 2]
            v = np.linalg.norm(ep[f"{args.bulb}_lin_vel"][:-1], axis=1)
            print(
                f"   {args.bulb}: z {z[0]:.2f} -> min {z.min():.2f}; peak speed {v.max():.2f} m/s"
                f" at step {int(np.argmax(v))}; final speed {v[-1]:.2f} m/s"
            )
        term = {t: bool(ep[t][-1]) for t in ("success_term", "dropped_term", "timeout_term", "terminated") if t in ep}
        print(
            "   last row flags:", term, "(terminated without success/drop/timeout = another termination, e.g. *_struck)"
        )


if __name__ == "__main__":
    main()
