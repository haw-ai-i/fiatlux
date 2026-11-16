"""Ankle-tracker bridge: DeviceIO body skeleton -> UDP for the fullbody teleop driver.

Runs in the ``vr_teleop`` conda env (isaacteleop lives there; the sim env has no
DeviceIO). Opens a headless OpenXR session against the CloudXR runtime with
``XR_BD_body_tracking`` -- this COEXISTS with Isaac Sim's own immersive session
(verified live 2026-08-22) -- and streams one small UDP datagram per poll tick to
the driver on localhost:

    "t l_y r_y l_qx l_qy l_qz l_qw r_qx r_qy r_qz r_qw l_valid r_valid"

Body data only flows while the operator is CONNECTED IN AN IMMERSIVE session
(the sim rendering); before that the runtime serves a placeholder skeleton, which
we detect (all joints identical) and report as inactive rather than forwarding.

Launched by restart_fullbody_teleop.sh; standalone:
    conda activate vr_teleop && python body_bridge.py [--port 47901]
"""

import argparse
import os
import socket
import time

from isaacteleop.deviceio import DeviceIOSession
from isaacteleop.deviceio_trackers import FullBodyTrackerPico
from isaacteleop.oxr import OpenXRSession
from isaacteleop.retargeting_engine.tensor_types.indices import BodyJointPicoIndex

L, R, HEAD, PELVIS = (int(BodyJointPicoIndex.LEFT_ANKLE), int(BodyJointPicoIndex.RIGHT_ANKLE),
                      int(BodyJointPicoIndex.HEAD), int(BodyJointPicoIndex.PELVIS))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=47901)
    ap.add_argument("--hz", type=float, default=50.0)
    args = ap.parse_args()

    os.environ.setdefault("XR_RUNTIME_JSON", os.path.expanduser("~/.cloudxr/openxr_cloudxr.json"))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest = ("127.0.0.1", args.port)

    tracker = FullBodyTrackerPico()
    exts = DeviceIOSession.get_required_extensions([tracker])
    oxr = None
    while oxr is None:
        try:
            oxr = OpenXRSession("fiatlux-body-bridge", exts)
        except RuntimeError as e:
            if "-35" in str(e):   # no headset connected yet
                print("[bridge] waiting for the headset...", flush=True)
                time.sleep(3.0)
            else:
                raise
    period = 1.0 / args.hz
    print(f"[bridge] streaming ankles -> udp://127.0.0.1:{args.port} at {args.hz:.0f} Hz", flush=True)
    live_logged = False
    with DeviceIOSession.run([tracker], oxr.get_handles()) as session:
        while True:
            t0 = time.monotonic()
            session.update()
            pose = tracker.get_body_pose(session)
            if pose is not None and pose.data is not None:
                la = pose.data.joints.joints(L)
                ra = pose.data.joints.joints(R)
                hd = pose.data.joints.joints(HEAD)
                pv = pose.data.joints.joints(PELVIS)
                lp, rp = la.pose.position, ra.pose.position
                lo, ro = la.pose.orientation, ra.pose.orientation
                # placeholder skeleton (pre-immersive): every joint carries one identical pose
                placeholder = (lp.x == rp.x and lp.y == rp.y and lp.z == rp.z
                               and lp.x == hd.pose.position.x and lp.y == hd.pose.position.y)
                if not placeholder:
                    if not live_logged:
                        print("[bridge] LIVE body data flowing", flush=True)
                        live_logged = True
                    pp, po = pv.pose.position, pv.pose.orientation
                    # v2: full ankle positions (xyz each) -- the legs-mirror mode needs them
                    msg = (f"v2 {time.monotonic():.4f} "
                           f"{lp.x:.4f} {lp.y:.4f} {lp.z:.4f} {rp.x:.4f} {rp.y:.4f} {rp.z:.4f} "
                           f"{lo.x:.4f} {lo.y:.4f} {lo.z:.4f} {lo.w:.4f} "
                           f"{ro.x:.4f} {ro.y:.4f} {ro.z:.4f} {ro.w:.4f} "
                           f"{int(la.is_valid)} {int(ra.is_valid)} "
                           f"{pp.x:.4f} {pp.z:.4f} "
                           f"{po.x:.4f} {po.y:.4f} {po.z:.4f} {po.w:.4f} {int(pv.is_valid)}")
                    sock.sendto(msg.encode(), dest)
            dt = time.monotonic() - t0
            if dt < period:
                time.sleep(period - dt)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[bridge] bye")
