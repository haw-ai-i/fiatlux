"""Training bringup for Gazebo-based FIATLUX workflows."""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description() -> LaunchDescription:
    """Launch the standard Gazebo bringup together with training-only tools."""
    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [
                            FindPackageShare("fiatlux_bringup"),
                            "launch",
                            "fiatlux_gz_bringup.launch.py",
                        ]
                    )
                )
            ),
            Node(
                package="fiatlux_training_utils",
                executable="xacro_expander.py",
                parameters=[{"use_sim_time": True}],
            ),
        ]
    )
