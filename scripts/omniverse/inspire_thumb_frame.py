#!/usr/bin/env python3
"""Re-author the Inspire hand's thumb ROTATION joint so the thumb can stand up off the palm, no GPU.

The RH56DFTP manual (2.6.11) gives the thumb rotation as 90-165 deg to the metacarpal plane:
lying nearly in the palm plane at one end, perpendicular to it at the other. In the Unitree G1 +
Inspire USD the yaw axis lies along the fingers while the thumb's rest direction has a 35 deg
component along that same axis, so rotation sweeps the thumb across the palm on a cone that
tops out 55 deg above the palm plane -- it never stands up, and it leans over anything lying on
the palm. This turns the joint axis within the palm plane so it is perpendicular to the thumb's
in-plane rest direction: rotation then pivots the thumb up over its own base, reaching ~86 deg
at the existing 74.5 deg limit. Zero pose and limits are unchanged (both joint frames are rotated
together).

Writes a wrapper USD next to the source that references it and overrides only the joint's
localRot0/localRot1, so the vendor file is untouched.

Usage: python scripts/omniverse/inspire_thumb_frame.py <g1_..._inspire_...usd> [--out PATH]
       [--joint R_thumb_proximal_yaw_joint] [--thumb-dir "-0.015,-0.045,-0.063"]

``--thumb-dir`` is the thumb's rest direction (proximal -> distal) in the palm body's frame,
measured live at zero joint angles; the default is the right hand of the stock asset.
"""

import argparse
import math
import os

import numpy as np

from pxr import Gf, Usd, UsdPhysics

PALM_NORMAL = np.array([-1.0, 0.0, 0.0])  # right_hand_base_link: the palm face is local -x


def quat_to_mat(q: Gf.Quatf | Gf.Quatd) -> np.ndarray:
    q = Gf.Quatd(q)
    w = q.GetReal()
    x, y, z = q.GetImaginary()
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def mat_to_quat(m: np.ndarray) -> Gf.Quatf:
    w = math.sqrt(max(0.0, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2.0
    x = (m[2, 1] - m[1, 2]) / (4 * w)
    y = (m[0, 2] - m[2, 0]) / (4 * w)
    z = (m[1, 0] - m[0, 1]) / (4 * w)
    return Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z)))


def rotation_about(axis: np.ndarray, angle: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)


def elevation_deg(direction: np.ndarray) -> float:
    return math.degrees(math.asin(float(direction @ PALM_NORMAL) / np.linalg.norm(direction)))


# Thumb rotation joint + thumb rest direction (proximal->distal) in the palm frame, both hands
# of the stock asset. Exposed as a constant so callers (e.g. the teleop driver's on-demand
# regeneration) can reuse it without re-typing the measured vectors.
DEFAULT_JOINTS = (
    "R_thumb_proximal_yaw_joint:-0.015,-0.045,-0.063;"
    "L_thumb_proximal_yaw_joint:-0.0088,-0.0453,0.0634"
)


def generate_thumbfix(src: str, out: str | None = None, joint_specs: str = DEFAULT_JOINTS) -> str:
    """Write the re-authored-thumb wrapper USD for ``src`` and return its path.

    Importable so the teleop driver can regenerate the wrapper in-process (``pxr`` is only
    available once Kit is running). ``src`` is the vendor Inspire USD; ``out`` defaults to
    ``<src>_thumbfix.usd`` next to it. The wrapper references the vendor by a RELATIVE path,
    so it is portable across machines as long as it stays beside the vendor file.
    """
    src = os.path.abspath(src)
    out = os.path.abspath(out) if out else src[: -len(".usd")] + "_thumbfix.usd"

    stage = Usd.Stage.Open(src)
    root = stage.GetDefaultPrim()

    # Never Clear+CreateNew over a registered layer -- CreateNew on a path whose (cleared)
    # Sdf.Layer object is still alive crashes Kit natively. Remove the stale file instead.
    if os.path.exists(out):
        os.remove(out)
    wrapper = Usd.Stage.CreateNew(out)
    prim = wrapper.DefinePrim(root.GetPath())
    # Reference the vendor file by a RELATIVE path so the wrapper stays portable when copied
    # elsewhere (e.g. fetched from the HF dataset); an absolute path would bake in this machine's location.
    prim.GetReferences().AddReference("./" + os.path.relpath(src, os.path.dirname(out)))
    wrapper.SetDefaultPrim(prim)
    for key in ("upAxis", "metersPerUnit"):
        if stage.GetMetadata(key) is not None:
            wrapper.SetMetadata(key, stage.GetMetadata(key))

    print(f"{os.path.basename(src)}:")
    for spec in joint_specs.split(";"):
        joint_name, dir_str = spec.split(":")
        joints = [
            prim for prim in stage.Traverse() if prim.GetName() == joint_name and prim.IsA(UsdPhysics.RevoluteJoint)
        ]
        if len(joints) != 1:
            raise ValueError(f"expected exactly one revolute joint named {joint_name!r}, found {len(joints)}")
        joint = UsdPhysics.RevoluteJoint(joints[0])
        axis_index = {"X": 0, "Y": 1, "Z": 2}[str(joint.GetAxisAttr().Get())]
        rot0 = quat_to_mat(joint.GetLocalRot0Attr().Get())
        rot1 = quat_to_mat(joint.GetLocalRot1Attr().Get())
        axis_old = rot0[:, axis_index]

        thumb = np.array([float(v) for v in dir_str.split(",")])
        in_plane = thumb - PALM_NORMAL * float(thumb @ PALM_NORMAL)
        axis_new = np.cross(PALM_NORMAL, in_plane)
        axis_new /= np.linalg.norm(axis_new)
        # Positive joint travel must lift the thumb (the stock model's 0 -> +74.5 deg direction).
        lift = float((rotation_about(axis_new, math.radians(75)) @ thumb) @ PALM_NORMAL)
        if float((rotation_about(-axis_new, math.radians(75)) @ thumb) @ PALM_NORMAL) > lift:
            axis_new = -axis_new

        # Rotate the joint frame (in both bodies, so the zero pose is unchanged) onto axis_new.
        target_local = rot0.T @ axis_new
        unit = np.zeros(3)
        unit[axis_index] = 1.0
        cross = np.cross(unit, target_local)
        sin = np.linalg.norm(cross)
        turn = np.eye(3) if sin < 1e-9 else rotation_about(cross, math.atan2(sin, float(unit @ target_local)))
        rot0_new, rot1_new = rot0 @ turn, rot1 @ turn

        lower, upper = joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()
        body0 = joint.GetBody0Rel().GetTargets()[0].name
        print(f"  {joint_name}: axis in {body0}: {np.round(axis_old, 3)} -> {np.round(axis_new, 3)}")
        for deg in (0.0, upper):
            lifted = rotation_about(axis_new, math.radians(deg)) @ thumb
            print(f"    elevation above the palm plane at {deg:5.1f} deg of yaw: {elevation_deg(lifted):5.1f} deg")
        print(f"    limits unchanged: [{lower:.1f}, {upper:.1f}] deg")
        override = UsdPhysics.RevoluteJoint(wrapper.OverridePrim(joints[0].GetPath()))
        override.CreateLocalRot0Attr().Set(mat_to_quat(rot0_new))
        override.CreateLocalRot1Attr().Set(mat_to_quat(rot1_new))

    wrapper.GetRootLayer().Save()
    print(f"  wrote {out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("--out", default=None, help="wrapper USD path (default: <src>_thumbfix.usd)")
    ap.add_argument(
        "--joints",
        default=DEFAULT_JOINTS,
        help="semicolon-separated JOINT:x,y,z pairs -- the thumb rotation joint and the thumb's rest "
        "direction (proximal -> distal) in the palm body's frame, measured live at zero joint angles. "
        "The default covers both hands of the stock asset.",
    )
    args = ap.parse_args()
    generate_thumbfix(args.src, args.out, args.joints)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
