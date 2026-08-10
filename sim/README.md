# Simulation (runs on a Mac, no hardware needed)

MuJoCo digital twin of the bench: the xArm 1S, the table, and a Boss DS-1
placeholder with 3 knobs. Used to test IK, approach poses, collision, and
(phase 3) the visual-servoing loop before touching the real arm.

- `ik.py` — the course package's own invkin(), lifted from xarmrob with the
  ROS wrapper removed. `solve(xyz)` returns servo counts. Run it directly
  for the round-trip self-check against `scripts/arm.py` FK (0.00 mm).
- `urdfmap.py` — course joint angles -> URDF qpos (offsets +90 / -50.4 /
  +50.4 deg). Run directly to verify (0.63 mm worst over 20 random poses).
- `scene.py` — builds the scene; `grade(target)` returns reachable /
  above_floor / collisions / servo counts. Values marked MEASURE are bench
  placeholders.
- `view.py` — interactive viewer: `.venv/bin/mjpython view.py` (mjpython is
  required on macOS; use launch_passive, the managed viewer crashes).

Setup: `python3 -m venv .venv && .venv/bin/pip install mujoco ikpy numpy
scipy matplotlib pillow`

Arm meshes and URDF adapted from allProgramming/ros2_xarm_1s_demos (MIT,
see LICENSE-xarm-model). Link lengths were corrected to our measured
values (shoulder x 2->10 mm, upper arm 97.75->101 mm); kinematic truth
comes from `scripts/arm.py` and the course package, not the mesh model.
The URDF has no forearm-roll joint (servo 7); the course IK holds it at
zero and `urdfmap.qpos_from` asserts if a pose ever needs it.
