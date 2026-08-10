# ME439 xArm: pedal knob turning

Everything that lives on the Raspberry Pi, pulled off it so it can be version
controlled. Build artifacts are excluded; this is source only.

The robot is a HiWonder xArm 1S, modified by the course into a 6-DOF chain plus a
gripper. It talks USB HID (`0483:5750`) through the Python `xarm` library. The goal
is to reach from neutral and turn a knob on a Boss DS-1 distortion pedal without
disturbing the neighbouring knobs.

## Layout

| Path | What it is |
|---|---|
| `scripts/arm.py` | All motion. read, neutral, teach, save, goto, grip, squeeze, turn |
| `scripts/turn_knob.py` | The whole knob routine as one run, with safety gates |
| `scripts/see.py` | ArUco detection and frame capture from the C270 |
| `scripts/sweep.py` | Force-guided joint offset search (see "what did not work") |
| `scripts/poses.json` | Taught poses, in raw servo counts |
| `cv/` | Course camera calibration for the Logitech C270, plus tag tools |
| `ros2_ws/src/` | The course `xarmrob` packages, Apache-2.0, by Peter Adamczyk |
| `system/` | systemd units for network retry and the optional boot demo |

## Our changes to the course code

The `xarmrob` package is Adamczyk's, unmodified except for one file:

`ros2_ws/src/xarmrob/config/robot_xarm_info.yaml`, where
`bus_servo_neutral_cmds_base_to_tip` was changed to `[500,515,496,488,877,527,388]`,
calibrated by hand for this specific arm with `xarm_cmd_sliders`. The course original
is preserved alongside as `robot_xarm_info.yaml.orig`.

That file's stock neutral commanded the gripper to **90**, which is fully open and
jammed against its mechanical stop. That stalls the servo and browns out the USB bus.
Do not put it back.

Everything under `scripts/` and `system/` is ours.

## Key facts about this hardware

- **Servo IDs base to tip are `[6,5,4,7,3,2,1]`.** Not sequential. ID 7 is the
  course-added forearm rotation, ID 1 is the gripper.
- Positions read back **unsigned 16-bit**; subtract 65536 above 32767.
- **Servo ID 2 turns the knob.** It is joint_56, the last x-rotation, so it spins
  about the gripper's own pointing axis. 90 degrees is 380 counts, and the range is
  -112 to +112.
- Gripper: 90 counts is fully open, 610 fully closed. We cap at 590.
- **`wait=True` in the xarm library is just `time.sleep(duration/1000)`.** Sleeping
  again on top of it leaves the arm stationary half the time, which is what made the
  early motion stutter.
- Motion uses Adamczyk's minimum-jerk profile, `10t^3 - 15t^4 + 6t^5`.
- The Pi has **OpenCV 4.6.0**, the last version with `aruco.Dictionary_get` and
  `estimatePoseSingleMarkers`. Upgrading breaks the course vision code.

## Grip force without a force sensor

The gripper has no sensor, so grip force is measured as the gap between the commanded
servo count and the actual one. The servo is fighting the object by exactly that much.

Holding an object plateaus: actual stops rising while commands keep climbing.

```
commanded 540  actual 503  squeeze 40
commanded 560  actual 507  squeeze 56   <- plateau, that is the knob
```

Closing on air does not plateau; actual tracks the command all the way to the cap.
`turn_knob.py` refuses to turn when there is no plateau, because turning without a
grip puts the torque into shoving the robot instead.

## What did not work, and why

Kept deliberately, since the course asks for a description of what failed.

- **Force-guided position search.** Sweeping joint offsets and hill-climbing on grip
  force returns a completely flat 18-24 counts, which is just the two fingers touching
  each other. Force carries no information until the fingers already touch the target,
  so it cannot acquire a target from outside contact range. Re-teaching by hand is
  faster and exact.
- **Reading the knob pointer by brightness.** Locks onto the specular brushed-metal
  streak on the cap, not the pointer tab, giving an answer ~180 degrees wrong while
  reporting healthy contrast. Specular highlights are set by lighting geometry, not by
  the object, so they are not fiducials.
- **ArUco drift detection with the tag on a nearby object.** Faithfully reported 1.6 mm
  of drift while the grip was missing entirely, because the tag was not attached to the
  pedal. Template matching a feature on the pedal itself gave 0.94-0.99 confidence and
  is the better tool.

## Deploying back to a Pi

```bash
rsync -a scripts/ pi@<host>:~/
rsync -a cv/ pi@<host>:~/cv/
rsync -a ros2_ws/src/ pi@<host>:~/ros2_ws/src/
```

Then rebuild the ROS packages:

```bash
cd ~/ros2_ws && colcon build --symlink-install
```

`system/` files go to `/usr/local/bin/` and `/etc/systemd/system/`, then
`systemctl daemon-reload` and enable `netretry.timer`.

`arm-demo.service` runs the demo at boot but only when armed by a flag file
(`~/DEMO_ALWAYS` or `~/DEMO_ON_BOOT`). Leave it unarmed: armed, the arm moves 45
seconds after **every** power-on, which is a hazard if anyone else plugs the Pi in.

## Centring on the knob: uncalibrated visual servoing

Bench data showed the grip fails from sideways error: 4 mm of lateral offset
makes the knob eject from the jaws, while height barely matters. Taught poses
kept missing by about that much, because a hobby arm does not return to a
commanded position exactly.

The fix is uncalibrated visual servoing (Piepmeier's thesis is the canonical
source; Peter Corke's Robotics, Vision and Control covers it as image-based
visual servoing). The idea: instead of calibrating the camera and the arm well
enough to compute the right move in advance, close the loop in the image. The
camera sees both the gripper and the knob in the same frame; the arm nudges
itself until they line up. Calibration error stops mattering because every
approach ends with the camera confirming alignment, not assuming it.

The whole controller:

1. Once, at the pre-grasp pose: jog the arm a small step in x, then y, and
   measure how many pixels the gripper moved in the image. Those two
   measurements form a 2x2 matrix J, the mm-to-pixel exchange rate.
2. Loop: measure the pixel error e between gripper and knob, move the arm by
   0.4 * inv(J) * e (capped per step), look again. Stop when the error is
   under ~2 px or after 6 iterations. Typically converges in 3 to 5.

The 0.4 damping factor means each step only cancels part of the error, which
makes the loop converge even if J is off by 30 to 50 percent.

Jog size is a tuning knob, default 5 mm. To be evaluated in simulation and
again on the real arm: if 5 mm disturbs the scene or overshoots, drop to 3 mm
or less; if the pixel displacement is too small to measure reliably against
detection noise (about half a pixel), increase it. The right size is the
smallest jog whose pixel displacement is clearly above noise.

Why not classical hand-eye calibration (cv2.calibrateHandEye): it needs 15 to
20 accurate end-effector poses, and this arm's positioning error would poison
the result. The servoing loop sidesteps the problem entirely.
