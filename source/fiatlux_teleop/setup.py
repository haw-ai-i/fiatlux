# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'fiatlux_teleop' extension package.

VR teleoperation on top of the ``fiatlux_task`` benchmark. Teleop-only dependencies live HERE, not in
the benchmark's requirements, so ``fiatlux_task`` installs/runs without them. ``fiatlux_task`` is a
sibling source package (put on PYTHONPATH, like this one), so it is not declared as a pip dependency.
"""

from setuptools import setup

INSTALL_REQUIRES = [
    "onnxruntime",  # NVIDIA SONIC loco-manip policy (Carry/VR teleop). OpenXR/CloudXR come from Isaac Sim.
]

setup(
    name="fiatlux_teleop",
    version="0.1.0",
    packages=["fiatlux_teleop"],
    description="VR teleoperation extension for the Fiatlux benchmark (kept out of fiatlux_task).",
    install_requires=INSTALL_REQUIRES,
    license="Apache-2.0",
    include_package_data=True,
    python_requires=">=3.10",
    zip_safe=False,
)
