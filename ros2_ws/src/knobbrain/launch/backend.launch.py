#!/usr/bin/env python3
"""His preset_system_launch.launch.py with one node removed.

The three that stay are reactive: they compute IK and drive servos when a
message arrives and do nothing otherwise. The one that goes, preset_controller,
is his sequencer, which starts driving all eight knobs the moment the tags
lock. Our brain publishes the same three topics it did, so nothing below this
line knows the difference.

His launch file is untouched and still runs his original demo.
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package='xarmrob', executable='command_xarm',
             name='command_xarm', output='screen'),
        Node(package='xarmrob', executable='xarm_kinematics',
             name='xarm_kinematics', output='screen'),
        Node(package='mobrob', executable='aruco',
             name='aruco_tracker', output='screen'),
        # preset_controller deliberately absent: knobbrain replaces it.
    ])
