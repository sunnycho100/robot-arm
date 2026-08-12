#!/usr/bin/env python3
"""Turn a knob over ROS 2.

    ros2 launch xarmrob knob_turner.launch.py degrees:=115.0

EVERY node gets parameters=[params_file], and that is not boilerplate.
command_xarm and xarm_kinematics declare their calibration as parameters with
HARDCODED DEFAULTS, and one of those defaults disagrees with the yaml: joint_01
maps [-90,0,90] to [880,500,120] in the code and to [120,500,880] in
robot_xarm_info.yaml. Launch the node bare and the base joint is mirrored, the
neutral pose becomes a flat [500]*7 instead of the calibrated one, and the link
lengths revert to nominal. The arm then moves confidently to the wrong place
and nothing reports an error.

Also: scripts/arm.py must NOT be running. command_xarm opens
xarm.Controller('USB') and so does arm.py. One device, one controller.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    params_file = os.path.join(
        get_package_share_directory('xarmrob'), 'config', 'robot_xarm_info.yaml')
    degrees = LaunchConfiguration('degrees')
    pose = LaunchConfiguration('pose')

    return LaunchDescription([
        DeclareLaunchArgument('degrees', default_value='115.0',
                              description='how far to turn the knob, degrees'),
        DeclareLaunchArgument('pose', default_value='grip0',
                              description='which taught pose to grip from'),
        Node(
            package='xarmrob',
            executable='command_xarm',
            name='command_xarm',
            parameters=[params_file],
            output='screen',
        ),
        Node(
            package='xarmrob',
            executable='xarm_kinematics',
            name='xarm_kinematics',
            parameters=[params_file],
            output='screen',
        ),
        Node(
            package='xarmrob',
            executable='knob_turner',
            name='knob_turner',
            parameters=[params_file,
                        {'degrees': degrees, 'pose': pose}],
            output='screen',
        ),
    ])
