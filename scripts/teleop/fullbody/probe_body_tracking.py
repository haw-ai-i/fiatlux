"""Phase-0 probe: prove Pico ankle-tracker data reaches this machine end-to-end.

Runs in the ``vr_teleop`` conda env (NOT the sim env) and opens its OWN headless
OpenXR session against the CloudXR runtime with ``XR_BD_body_tracking`` enabled,
then polls the 24-joint Pico body skeleton and prints the two ankles live.

No Isaac Sim involved: the chain under test is
    Pico Motion Trackers -> Pico body tracking -> web client (bodyTracking ON)
    -> CloudXR runtime -> OpenXR XR_BD_body_tracking -> DeviceIO -> here.

Prerequisites (see run_probe.sh next to this file):
  * CloudXR runtime up (restart_sonic_teleop.sh's [3/5] block, or the standalone
    runtime launch) -- the SIM DOES NOT need to run for this probe.
  * The web client must request body tracking: bundle.js ``bodyTracking:!0``
    (the July patch set it to ``!1`` = off; flip it back for body sessions).
  * Operator in the headset, trackers on both ankles, Pico body-tracking
    calibration done (Motion Tracker app -> body tracking mode), client Connected.

Prints one line per second: per-ankle position (m) + tracked flag. Ctrl-C to quit.
"""

import os
import sys
import time

from isaacteleop.deviceio import DeviceIOSession
from isaacteleop.deviceio_trackers import FullBodyTrackerPico
from isaacteleop.oxr import OpenXRSession
from isaacteleop.retargeting_engine.tensor_types.indices import BodyJointPicoIndex

L, R, H = int(BodyJointPicoIndex.LEFT_ANKLE), int(BodyJointPicoIndex.RIGHT_ANKLE), int(BodyJointPicoIndex.HEAD)


def main() -> int:
    if not os.environ.get("XR_RUNTIME_JSON"):
        os.environ["XR_RUNTIME_JSON"] = os.path.expanduser("~/.cloudxr/openxr_cloudxr.json")
    tracker = FullBodyTrackerPico()
    exts = DeviceIOSession.get_required_extensions([tracker])
    print(f"[probe] creating OpenXR session (exts={exts}) against "
          f"{os.environ['XR_RUNTIME_JSON']}", flush=True)
    oxr = None
    while oxr is None:
        try:
            oxr = OpenXRSession("fiatlux-body-probe", exts)
        except RuntimeError as e:
            if "-35" in str(e):  # XR_ERROR_FORM_FACTOR_UNAVAILABLE: no headset yet
                print("[probe] waiting for the headset to connect...", flush=True)
                time.sleep(3.0)
            else:
                raise
    with DeviceIOSession.run([tracker], oxr.get_handles()) as session:
        print("[probe] DeviceIO session up; waiting for body data "
              "(connect the headset client, body tracking calibrated)...", flush=True)
        last = 0.0
        while True:
            session.update()
            pose = tracker.get_body_pose(session)
            now = time.monotonic()
            if now - last >= 1.0:
                last = now
                if pose is None or pose.data is None:
                    print("[probe] body tracking INACTIVE (no data)", flush=True)
                else:
                    la, ra, hd = pose.data.joints.joints(L), pose.data.joints.joints(R), pose.data.joints.joints(H)
                    def _fmt(j, tag):
                        p = j.pose.position
                        return (f"{tag}[{'OK ' if j.is_valid else 'INV'}] "
                                f"({p.x:+.3f},{p.y:+.3f},{p.z:+.3f})")
                    print(f"[probe] {_fmt(la,'L_ankle')}  {_fmt(ra,'R_ankle')}  {_fmt(hd,'head')}", flush=True)
            time.sleep(0.02)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[probe] bye")
        sys.exit(0)
