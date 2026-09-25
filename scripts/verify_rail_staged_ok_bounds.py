#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Does _rail_staged_ok() in sonic_teleop.py bound dz on BOTH sides? (code-review finding, fix
verification -- sonic_teleop.py ~L1374)

``_rail_staged_ok`` is meant to catch a badly converged rail-hand approach before the base is
released and the (fake) brace is trusted for support. The original bug: ``ok = f < 20.0 and dz <
0.05`` had no lower bound on ``dz``, so a wrist that never reached the ladder at all (zero contact
force, large negative dz -- dangling in free air) still read as "staged". The fix bounds dz on
both sides (``abs(dz) < 0.05``), so a deep undershoot is now rejected same as a deep overshoot,
while the documented-normal small undershoot (a few cm, the usual DLS stall) still passes.

No Isaac Sim needed -- pure arithmetic on the two scalars the real function reads (``f`` from a
contact sensor, ``dz`` from the wrist pose). To avoid testing a hand-copied guess of the
expression that could silently drift from the real source, this script extracts the literal
``ok = ...`` line out of scripts/teleop/sonic_teleop.py by regex and evaluates THAT string, so a
future edit to the real check is picked up automatically (and a rename/refactor that breaks the
regex fails loudly rather than silently testing stale logic).

Example
-------
    uv run python scripts/verify_rail_staged_ok_bounds.py
    (no Isaac Sim / AppLauncher required; plain python3 also works)
"""

import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "teleop" / "sonic_teleop.py"


def extract_ok_expr() -> str:
    text = SRC.read_text()
    m = re.search(r"def _rail_staged_ok\(\):.*?\n( {16}ok = [^\n]+)\n", text, re.S)
    if not m:
        raise RuntimeError(f"could not locate '_rail_staged_ok's ok = ...' line in {SRC} -- function moved/renamed?")
    line = m.group(1).strip()
    assert line.startswith("ok = "), line
    return line[len("ok = ") :]


def evaluate(expr: str, f: float, dz: float) -> bool:
    return bool(eval(expr, {"__builtins__": {"abs": abs}}, {"f": f, "dz": dz}))


def main() -> int:
    expr = extract_ok_expr()
    print(f"[verify] extracted expression from {SRC.name}: ok = {expr}", flush=True)

    cases = [
        # (label, f newtons, dz meters, expected "should be rejected")
        ("normal stage (small undershoot, light/no contact) -- must still pass", 2.0, -0.02, False),
        ("normal stage (landed almost exactly on target) -- must still pass", 0.5, 0.0, False),
        ("pressing on the ladder (force upper bound)", 45.0, 0.0, True),
        ("parked well above target (dz upper bound, pre-existing)", 0.0, 0.20, True),
        ("deep undershoot, hand dangling in free air, no contact (THE BUG)", 0.0, -0.30, True),
        ("extreme undershoot (IK totally stalled)", 0.5, -1.00, True),
        ("right at the new lower boundary (-0.049 m) -- must still pass", 0.0, -0.049, False),
        ("just past the new lower boundary (-0.051 m) -- must now be rejected", 0.0, -0.051, True),
    ]

    all_ok = True
    for label, f, dz, should_reject in cases:
        ok = evaluate(expr, f, dz)
        rejected = not ok
        verdict = "OK" if rejected == should_reject else "MISMATCH"
        if verdict == "MISMATCH":
            all_ok = False
        print(
            f"[verify] f={f:>6.1f} N  dz={dz:+.3f} m  -> ok={ok!s:5}  "
            f"(rejected={rejected}, expected rejected={should_reject})  [{verdict}]  {label}",
            flush=True,
        )

    undershoot_rejected = not evaluate(expr, 0.0, -0.30)
    if all_ok and undershoot_rejected:
        print(
            "[verify] CONFIRMED FIXED: the deep, non-contacting undershoot (f=0 N, dz=-0.30 m) is "
            "now rejected, and every previously-passing normal-stage case still passes. dz is now "
            "bounded on both sides.",
            flush=True,
        )
    else:
        print("[verify] FIX NOT CONFIRMED -- see MISMATCH rows above.", flush=True)

    return 0 if (all_ok and undershoot_rejected) else 1


if __name__ == "__main__":
    sys.exit(main())
