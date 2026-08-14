#!/usr/bin/env python3
"""One command for the demo: backend up, TUI in front of you.

    ros2 run knobbrain go

Why this is a wrapper and not a launch file with four nodes in it: ros2 launch
does not forward stdin to the processes it starts, which is the same reason
keyboard teleop has to be run with `ros2 run`. A TUI inside a launch file draws
itself perfectly and then ignores every key. His three nodes also log on every
action, and that text lands on top of the screen, so their output goes to a
file instead.
"""
import os
import signal
import subprocess
import sys

LOG = '/tmp/knobbrain.log'


def main(args=None):
    argv = sys.argv[1:]
    with open(LOG, 'w') as log:
        backend = subprocess.Popen(
            ['ros2', 'launch', 'knobbrain', 'backend.launch.py'],
            stdout=log, stderr=subprocess.STDOUT,
            preexec_fn=os.setsid)          # its own group, so Ctrl-C is ours
    print(f'backend starting, log at {LOG}')
    try:
        return subprocess.call(['ros2', 'run', 'knobbrain', 'brain'] + argv)
    finally:
        # The whole group: ros2 launch spawns the nodes as children and killing
        # only the launcher leaves three orphans holding the camera and the arm.
        try:
            os.killpg(os.getpgid(backend.pid), signal.SIGINT)
            backend.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(os.getpgid(backend.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        print(f'backend stopped. its log is at {LOG}')


if __name__ == '__main__':
    sys.exit(main())
