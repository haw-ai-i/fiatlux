"""One-shot 50 Hz body-skeleton capture to CSV, for gait-detector development."""

import os
import sys
import time

from isaacteleop.deviceio import DeviceIOSession
from isaacteleop.deviceio_trackers import FullBodyTrackerPico
from isaacteleop.oxr import OpenXRSession

os.environ.setdefault("XR_RUNTIME_JSON", os.path.expanduser("~/.cloudxr/openxr_cloudxr.json"))
out, dur = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
tracker = FullBodyTrackerPico()
oxr = OpenXRSession("fiatlux-gait-capture", DeviceIOSession.get_required_extensions([tracker]))
rows = 0
with DeviceIOSession.run([tracker], oxr.get_handles()) as s, open(out, "w") as f:
    f.write(
        "t," + ",".join(f"j{i}_{c}" for i in range(24) for c in ("x", "y", "z", "qx", "qy", "qz", "qw", "valid")) + "\n"
    )
    t0 = time.monotonic()
    while time.monotonic() - t0 < dur:
        s.update()
        pose = tracker.get_body_pose(s)
        if pose is not None and pose.data is not None:
            vals = [f"{time.monotonic() - t0:.4f}"]
            for i in range(24):
                j = pose.data.joints.joints(i)
                p, o = j.pose.position, j.pose.orientation
                vals += [
                    f"{p.x:.4f}",
                    f"{p.y:.4f}",
                    f"{p.z:.4f}",
                    f"{o.x:.4f}",
                    f"{o.y:.4f}",
                    f"{o.z:.4f}",
                    f"{o.w:.4f}",
                    str(int(j.is_valid)),
                ]
            f.write(",".join(vals) + "\n")
            rows += 1
        time.sleep(0.02)
print(f"[capture] wrote {rows} rows -> {out}", flush=True)
