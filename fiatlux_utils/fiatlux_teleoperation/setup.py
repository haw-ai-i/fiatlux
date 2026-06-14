from setuptools import find_packages, setup

package_name = "fiatlux_teleoperation"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    package_data={"": ["py.typed"]},
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="johntangz",
    maintainer_email="johntangz@intrinsic.ai",
    description="Utility scripts for FIATLUX teleoperation",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "cartesian_keyboard_teleop = fiatlux_teleoperation.cartesian_keyboard_teleop:main",
            "joint_keyboard_teleop = fiatlux_teleoperation.joint_keyboard_teleop:main",
        ],
    },
)
