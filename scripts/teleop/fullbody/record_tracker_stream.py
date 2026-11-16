"""Record the raw PICO stream -- head, both controllers, all 24 body joints -- to an .npz.

Independent of Isaac Sim: the chain is
    PICO Motion Trackers -> PICO body tracking -> web client (bodyTracking ON)
    -> CloudXR runtime -> OpenXR XR_BD_body_tracking -> isaacteleop DeviceIO -> here.

Isaac Lab's own ``OpenXRDevice`` CANNOT be used for this: its ``TrackingTarget`` enum has only
HAND_LEFT/HAND_RIGHT/HEAD/CONTROLLER_LEFT/CONTROLLER_RIGHT and ``_get_raw_data`` never queries
body joints. The lower body is reachable only through ``isaacteleop.deviceio_trackers``.

RUN IT IN THE ``vr_teleop`` CONDA ENV (which has isaacteleop), not the sim env:

    conda activate vr_teleop
    # CloudXR runtime must be up and the headset client Connected, with body tracking
    # calibrated in the PICO Motion Tracker app. The web client must REQUEST body tracking:
    # bundle.js must carry ``bodyTracking:!0`` (a July patch set it to ``!1`` = off).
    python record_tracker_stream.py --out ~/pico_captures/climb_01.npz --seconds 30

Writes one row per polled frame: monotonic time, per-joint position+quaternion+validity for
all 24 PICO body joints, head pose, and both controller poses + inputs.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

JOINTS = [
    "PELVIS", "LEFT_HIP", "RIGHT_HIP", "SPINE1", "LEFT_KNEE", "RIGHT_KNEE", "SPINE2",
    "LEFT_ANKLE", "RIGHT_ANKLE", "SPINE3", "LEFT_FOOT", "RIGHT_FOOT", "NECK",
    "LEFT_COLLAR", "RIGHT_COLLAR", "HEAD", "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HAND", "RIGHT_HAND",
]  # BodyJointPico indices 0..23; names with gaps (3/6/9) are the PICO spine chain


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = until Ctrl-C")
    ap.add_argument("--rate", type=float, default=50.0)
    args = ap.parse_args()

    from isaacteleop.deviceio import DeviceIOSession
    from isaacteleop.deviceio_trackers import ControllerTracker, FullBodyTrackerPico, HeadTracker

    os.environ.setdefault("XR_RUNTIME_JSON", os.path.expanduser("~/.cloudxr/openxr_cloudxr.json"))
    from isaacteleop.oxr import OpenXRSession

    body, head, ctrl = FullBodyTrackerPico(), HeadTracker(), ControllerTracker()
    trackers = [body, head, ctrl]
    exts = DeviceIOSession.get_required_extensions(trackers)
    print(f"[rec] OpenXR session (exts={exts}) via {os.environ['XR_RUNTIME_JSON']}", flush=True)
    oxr = None
    while oxr is None:
        try:
            oxr = OpenXRSession("fiatlux-tracker-recorder", exts)
        except RuntimeError as exc:
            if "-35" not in str(exc):  # XR_ERROR_FORM_FACTOR_UNAVAILABLE = headset not up yet
                raise
            print("[rec] waiting for the headset...", flush=True)
            time.sleep(3.0)

    rows_t, rows_body, rows_valid, rows_head, rows_ctrl = [], [], [], [], []
    dt, t0, warned = 1.0 / args.rate, time.monotonic(), False
    with DeviceIOSession.run(trackers, oxr.get_handles()) as session:
        print("[rec] recording -- Ctrl-C to stop", flush=True)
        while True:
            session.update()
            now = time.monotonic()
            if args.seconds and now - t0 >= args.seconds:
                break
            pose = body.get_body_pose(session)
            if pose is None or pose.data is None:
                if not warned:
                    print("[rec] body tracking INACTIVE -- is bodyTracking enabled in the "
                          "client and calibrated in the PICO app?", flush=True)
                    warned = True
                time.sleep(dt)
                continue
            js = pose.data.joints
            pos = np.empty((len(JOINTS), 7), np.float32)
            val = np.empty(len(JOINTS), bool)
            for i in range(len(JOINTS)):
                j = js.joints(i)
                p, q = j.pose.position, j.pose.orientation
                pos[i] = (p.x, p.y, p.z, q.w, q.x, q.y, q.z)
                val[i] = bool(j.is_valid)
            rows_t.append(now - t0)
            rows_body.append(pos)
            rows_valid.append(val)
            hp = head.get_head(session)
            rows_head.append(_pose7(hp.data.pose) if hp and hp.data else np.zeros(7, np.float32))
            lc, rc = ctrl.get_left_controller(session), ctrl.get_right_controller(session)
            rows_ctrl.append(np.stack([_ctrl7(lc), _ctrl7(rc)]))
            time.sleep(dt)

    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez(
        out,
        t=np.asarray(rows_t, np.float32),
        body_pose=np.asarray(rows_body, np.float32),      # (T, 24, 7) xyz + wxyz
        body_valid=np.asarray(rows_valid, bool),          # (T, 24)
        head_pose=np.asarray(rows_head, np.float32),      # (T, 7)
        controller_pose=np.asarray(rows_ctrl, np.float32),  # (T, 2, 7) left, right
        joint_names=np.asarray(JOINTS),
    )
    print(f"[rec] wrote {out}: {len(rows_t)} frames, {rows_t[-1] if rows_t else 0:.1f} s", flush=True)
    return 0


def _pose7(p) -> np.ndarray:
    return np.asarray((p.position.x, p.position.y, p.position.z,
                       p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z), np.float32)


def _ctrl7(c) -> np.ndarray:
    if c is None or c.data is None:
        return np.zeros(7, np.float32)
    return _pose7(c.data.pose)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[rec] interrupted -- nothing written (use --seconds for a timed capture)")
