#!/usr/bin/env python3
"""Interactive scene viewer. Run with mjpython (macOS requirement):

    .venv/bin/mjpython view.py           # arm posed at the centre knob
    .venv/bin/mjpython view.py 0         # knob0 / 1 / 2

Drag to orbit, scroll to zoom, double-click a geom to track it.
"""
import sys, time
import mujoco.viewer
import scene

knob = f'knob{sys.argv[1]}' if len(sys.argv) > 1 else 'knob1'
data = mujoco.MjData(scene.MODEL)
scene._pose(data, scene.knob_targets()[knob])
print(f'posed at {knob}; grade: {scene.grade(scene.knob_targets()[knob], target_knob=knob)}')
# launch_passive, not launch: the managed viewer dies with "unknown exception"
# on this macOS setup, the passive one is fine (we pose, we don't simulate)
with mujoco.viewer.launch_passive(scene.MODEL, data) as v:
    while v.is_running():
        time.sleep(0.1)
