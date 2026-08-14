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
        # His aruco.py opens the camera with a bare cv2.VideoCapture(0), which
        # lets OpenCV try GStreamer first. Against the pipewire session running
        # on this Pi that deadlocks in a futex at IMPORT time: the process
        # starts, never registers a node, and /aruco simply never publishes,
        # which reads as a tag problem and is not one. Taking GStreamer out of
        # the running makes the same call fall through to V4L2 and open in
        # 0.2 s. Done here as environment rather than as an edit to his file.
        Node(package='mobrob', executable='aruco',
             name='aruco_tracker', output='screen',
             additional_env={'OPENCV_VIDEOIO_PRIORITY_GSTREAMER': '0'}),
        # preset_controller deliberately absent: knobbrain replaces it.
    ])
