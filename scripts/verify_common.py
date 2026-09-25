# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared harness pieces for the verify_*.py / diagnose_*.py physics scripts (issue #171).

Every script under this pattern launches Isaac Sim itself (``AppLauncher``) before importing
anything that touches ``pxr``/``omni``, so this module deliberately imports nothing Isaac-Sim-
specific at module scope -- only ``sys``/``os``/``traceback``, which are always safe. Callers
import ``fiatlux_task``/``isaaclab`` symbols themselves, after their own ``AppLauncher`` has run,
and pass the resulting config objects in.

Not every script shares every piece here (some parameterize the task id, some strip terminations
by a different rule, ``verify_s11_insert_replay.py`` returns more than a cfg) -- only the pieces
that were genuinely byte-identical across all of them are factored out; the rest stays in each
script rather than forcing a false shared shape onto real differences.
"""

from __future__ import annotations

import sys


def assert_right_checkout(cfg, required_attr: str) -> None:
    """Fail loudly if ``fiatlux_task`` resolved to the wrong checkout.

    On a machine where ``.venv`` is shared with a different checkout, a bare invocation can
    silently resolve ``fiatlux_task`` to the WRONG checkout's source. Run scripts with
    ``uv run python`` from this repo's root to avoid that; this is the fallback check for when
    it still goes wrong.
    """
    assert hasattr(cfg.scene, required_attr), (
        f"cfg.scene ({type(cfg.scene)} from {sys.modules[type(cfg.scene).__module__].__file__}) has no "
        f"{required_attr} -- fiatlux_task likely resolved to the wrong checkout again; check sys.path "
        "(run with `uv run python` from this repo's root)"
    )


def strip_visual_obs(cfg) -> None:
    """Drop every camera and its image observation term.

    Cameras only cost time in these scripts (zero/scripted action, nothing to see that the
    printed numbers do not already say) and draw no randoms, so dropping them does not perturb
    the layout a seed produced.
    """
    for camera in ("ego_camera", "torso_camera", "wrist_camera"):
        if getattr(cfg.scene, camera, None) is not None:
            setattr(cfg.scene, camera, None)
    for group_name in ("policy", "privileged"):
        group = getattr(cfg.observations, group_name, None)
        for term in ("ego_rgb", "torso_rgb", "wrist_rgb"):
            if group is not None and getattr(group, term, None) is not None:
                setattr(group, term, None)


def strip_drop_terminations(cfg) -> None:
    """Drop the success/dropped-bulb terminations that would end the episode early.

    These scripts step a forced scenario for a fixed duration to observe one specific mechanism;
    letting the benchmark's own success or drop terminations fire would cut that observation
    short before the diagnostic has run its course.
    """
    for term in ("success", "old_bulb_dropped", "fresh_bulb_dropped"):
        if getattr(cfg.terminations, term, None) is not None:
            setattr(cfg.terminations, term, None)


def strip_all_but_timeout(cfg) -> None:
    """Drop every termination except ``time_out``.

    These scripts watch a scripted or replayed episode for its whole configured length; any
    other termination (success, a drop, a fall) firing partway through would cut the window
    short and hide whatever happens after it.
    """
    for term in [t for t in vars(cfg.terminations) if not t.startswith("_")]:
        if term != "time_out" and getattr(cfg.terminations, term, None) is not None:
            setattr(cfg.terminations, term, None)


def run_verify_main(main, simulation_app) -> None:
    """Standard entrypoint for a verify_*/diagnose_*.py ``if __name__ == "__main__":`` block.

    A bare unhandled exception left to unwind through Isaac Sim's app object can preempt Python's
    normal traceback with a silent ``os._exit(1)`` from ``AppLauncher``'s own teardown machinery
    -- swallowing the actual error with zero diagnostic output (found while running these for
    issue #171/#167). Printing the traceback and flushing stdio BEFORE exiting, and calling
    ``os._exit(exit_code)`` directly on failure (skipping ``simulation_app.close()``, which is
    what triggers the swallow), avoids it. On success, ``simulation_app.close()`` tears down
    normally and the process exits 0 the ordinary way.
    """
    import os
    import traceback

    exit_code = 1
    try:
        exit_code = main()
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        if exit_code:
            os._exit(exit_code)
        simulation_app.close()
