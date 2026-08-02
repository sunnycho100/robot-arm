# Knob Turning Workflow

How to reproduce the DS-1 knob turn from scratch. Written 2026-07-29.

Goal: the xArm reaches from neutral, grips a knob on the Boss DS-1 pedal, turns it,
turns back, and releases, without disturbing the neighboring knobs.

## Files

On the Mac, in `~/Documents/me439/`:

| File | What it does |
|---|---|
| `arm.py` | All motion. read, neutral, teach, save, goto, grip, squeeze, turn, list |
| `knob.py` | Finds the knobs by itself, reads each white pointer, grades the camera setup |
| `camstream.py` | Live camera in a browser, with a sharpness score, for aiming the camera |
| `turn_knob.py` | Closed-loop turn. Measures the knob, re-turns by what is left, records the run |
| `runs.py` | Compares recorded runs so the setup improves on evidence |
| `see.py` | ArUco tag detection from the C270 webcam. Also importable as `look()` and `snap()` |

All live on the Pi at `~/`. Redeploy after editing:

```bash
scp ~/Documents/me439/{arm,knob,camstream,turn_knob,runs,see}.py pi@<ip>:~/
```

Saved poses live on the Pi at `~/poses.json`, knob positions at `~/knobs.json`,
and each recorded run in `~/runs/<timestamp>/`.

## Connect

```bash
ssh arm2
```

Aliases in `~/.ssh/config`: `arm2` is the hotspot address, `armhome` is
the home-network address, `arm` is me439S25.local.

**Both the Mac and the Pi must be on the same network.** This is the single most common
failure. Check with `ipconfig getifaddr en0`. The Pi does not answer ping on every
network, so test with `nc -z -G 4 <host> 22`, not ping.

**Do not trust `me439S25.local`.** Every Pi in the class is flashed with that hostname,
so mDNS resolves to whichever one answers first. In a room of ten, that is a coin flip
and you could drive a classmate's arm.

Instead find it by MAC, which is unique and never changes:

```bash
cd ~/Documents/me439 && ./findpi ssh
```

`findpi` sweeps whatever subnet the Mac is on and matches the Pi's wifi MAC
`<the Pi's wifi MAC>`. Works at home, on the hotspot, anywhere, no editing. It relies on
the Pi answering ARP, which it does even where it ignores ping.

For access across different networks, sign the Pi into a RealVNC account (VNC Server
icon in the Pi's taskbar, then Sign in). That carries VNC only, not SSH, so run commands
in a terminal inside the VNC desktop.

## Physical setup, do this first

1. **Tape the pedal down.** 
2. **Tape the arm base down.** 
3. **Turn a light on.** Tag detection went from 40/40 frames to 1/20 purely from the
   room getting dimmer.
4. Put the ArUco tag **on the pedal**, not on a loose object beside it. A tag on
   something else measures that something else.

Steps 1 and 2 are not optional. The reaction torque of turning the knob is enough to
slide the robot, and once the arm and the pedal move relative to each other, every
taught pose is wrong.

## Network insurance installed on the Pi

**`netretry.timer`** runs every 30 seconds. If the Pi is not connected it rescans and
tries `hotspot`, then `eduroam`, then `201`. This exists because NetworkManager gives up
too easily when the Pi boots before the hotspot is broadcasting, which is the most
likely reason the connection failed in class on 2026-07-29.

Check what it has been doing:

```bash
journalctl -t netretry -n 10
```

An empty log is good, it means the Pi has never lost its connection.

**`arm-demo.service`** can run the demo at boot with no network, screen, or laptop, but
is deliberately left **unarmed**. It does nothing unless a flag file exists:

```bash
touch ~/DEMO_ON_BOOT     # one single run, then disarms itself
touch ~/DEMO_ALWAYS      # every boot, until the file is removed
```

Armed, the arm moves 45 seconds after every power-on. That is a hazard if anyone else
plugs the Pi in, so keep it unarmed unless deliberately using it, and **test it at home
before relying on it**. The motion path has never been verified from a boot.

An `eduroam` profile is configured (PEAP, MSCHAPv2, identity `<your-netid>@wisc.edu`,
priority 60) but untested, since eduroam is out of range at home. Turn the hotspot off
on campus or it wins at priority 100.

## Quick demo, no setup needed

Shows the motion without gripping anything, so nothing can shove the base. Runs off
the pose already saved in `poses.json`, no teaching required.

```bash
python3 ~/arm.py neutral && python3 ~/arm.py goto grip0 && python3 ~/arm.py turn -60 && python3 ~/arm.py turn 60 && python3 ~/arm.py neutral
```

Check the arm is alive first. Seven numbers back means good, an error means replug USB:

```bash
python3 ~/arm.py read
```

## Run it

One line, start to end:

```bash
python3 ~/arm.py neutral && python3 ~/arm.py teach grip0 && python3 ~/turn_knob.py --force
```

It parks at neutral, cuts servo torque so you can hand-pose the fingers around a knob,
and the moment you press Enter it saves that pose and runs the full sequence:

```
camera check -> neutral -> approach -> grip -> turn 115 -> turn back -> release -> neutral
```

If the pose is already taught and nothing has moved, just:

```bash
python3 ~/turn_knob.py --force
```

Flags: `--blind` skips the camera entirely, `--force` skips only the drift stop,
`--deg=115` turn angle, `--squeeze=35` grip force, `--pose=grip0` which pose to use.

## Reading the grip output

This is the part worth understanding. The gripper has no force sensor, so grip force is
measured as the **gap between the commanded count and the actual count**. The servo is
fighting the object by exactly that much.

Holding the knob looks like this. Actual stops climbing while commands keep rising:

```
commanded 500  actual 479  squeeze 21
commanded 540  actual 500  squeeze 40
commanded 560  actual 504  squeeze 56   <- plateau at 504, that is the knob
```

Closing on air looks like this. Actual tracks the command all the way to the limit:

```
commanded 560  actual 542  squeeze 22
commanded 590  actual 562  squeeze 28
NOT HOLDING: reached the close limit at 590 with only 28 counts of force.
```

The script stops on the second case rather than turning, because turning without a grip
puts the torque into shoving the robot.

## If something goes wrong

Ctrl-C, then:

```bash
python3 ~/arm.py neutral
```

Never leave the gripper clamped. A stalled servo is what browns out the USB bus. If the
arm vanishes from `lsusb`, unplug and replug the USB cable and check
`python3 ~/arm.py read`.

| Symptom | Cause | Fix |
|---|---|---|
| "closed on nothing" | pedal moved since teaching | re-teach `grip0` |
| "tag moved N mm" stop | real drift, or weak detection | more light, or `--force` |
| no tags found | tag clipped by frame edge, or too small in view | whole tag plus its white border must be inside the frame |
| arm feels stiff in teach | normal | these servos are heavily geared, stiff but movable is correct |
| roll out of range | turn too big from this grip | grip at a different angle, or turn in two bites |

## Facts worth not rediscovering

- **Servo IDs base to tip are `[6,5,4,7,3,2,1]`.** Not sequential. ID 7 is the
  course-added forearm rotation, ID 1 is the gripper.
- Positions read back **unsigned 16-bit**. Subtract 65536 when the raw value is above 32767.
- **Servo ID 2 turns the knob.** It is joint_56, the last x-rotation in the chain, so it
  spins about the gripper's own pointing axis. 90 degrees is 380 counts.
- Roll range is **-112 to +112 degrees**, so the most you get from a grip at 0 is about 115.
- **Do not pre-roll the wrist to buy more range.** The gripper hangs well off the roll
  axis, so rolling swings the fingers far enough to lose the knob entirely.
- Gripper calibration: 90 counts is fully open, 610 fully closed. The script caps at 590.
- **`wait=True` in the xarm library is just `time.sleep(duration/1000)`.** Sleeping again
  on top of it makes the arm stand still half the time, which is what made early motion
  stutter.
- Motion uses Adamczyk's minimum jerk profile, `10t^3 - 15t^4 + 6t^5`. Zero velocity and
  zero acceleration at both ends.
- **GUI nodes need VNC, not SSH.** tkinter and `cv2.imshow` die silently with no display.
- The Pi has **OpenCV 4.6.0**, the last version with `aruco.Dictionary_get` and
  `estimatePoseSingleMarkers`. Do not upgrade it, the course code will stop working.
- Camera is a **Logitech C270**, which is the exact model the course calibration files
  were measured on, so `cam_matrix.p` is valid as-is.

## Not solved yet

- **Camera to robot base transform.** Without it, vision can report that the world moved
  but cannot correct for it. Two routes: a second tag on the robot base (Adamczyk's L24
  addendum), or touch the tag with the gripper at three or four spots and solve from the
  correspondences.
- **Tag size.** Everything scales linearly with it. Measure the black square edge in mm
  and pass `--size` in meters to `see.py`, or the distances stay wrong by a constant factor.
- **Whether the knob has ever actually turned.** Never confirmed. The one turn that ran
  probably put its torque into sliding the base instead.
- **Angle calibration.** Still the values inherited from the teammate. All of the above
  works in raw servo counts, which deliberately avoids depending on it.

---

# Camera calibration and the closed-loop turn (added 2026-08-02)

This is the reusable part. A new bench, a moved camera, a different pedal: run
these four steps and the rig re-derives its own numbers. Nothing here needs a
value copied from a previous session.

## 1. Aim the camera

```bash
ssh pi@<ip> 'setsid nohup python3 ~/camstream.py >/tmp/camstream.log 2>&1 </dev/null &'
open http://<ip>:8080
```

A green box marks the middle of the frame with a **sharpness** score above it.

- Put the pedal inside the box. The score only measures what is in the box, so
  with the pedal outside it you are scoring the table.
- **Only the pedal.** The arm does not need to be in frame at all, and every pixel
  spent on it is resolution not spent on the knobs.
- As near straight down as clears the arm's swing. Mount on the far side from the
  arm.
- Move it up and down and watch the score. It climbs as the pedal fills the box,
  and **falls once you pass the C270's fixed-focus limit**. That is how the focus
  distance gets found rather than guessed. Around 40 cm worked here.
- No light directly behind the lens. That puts a specular hot spot on the knob
  tops, which is what wrecked an earlier attempt at reading them.
- **Tape the mount and the cable** when you like it. Every number below is tied to
  that exact position.

Only one process can hold the camera, so stop the stream before anything else:

```bash
ssh pi@<ip> 'pkill -f "[c]amstream"'
```

## 2. Calibrate

```bash
ssh pi@<ip> 'python3 ~/knob.py calibrate'
```

It finds the knobs itself and grades the view:

```
knobs found: 3
  knob1: at (834,442)  cap 32px  round 0.76  sharp 1156  pointer 352 deg (contrast 19.1)  ~450mm away
  knob2: at (786,484)  cap 34px  round 0.89  sharp  627  pointer  18 deg (contrast 17.0)  ~424mm away
  knob3: at (826,542)  cap 34px  round 0.76  sharp  767  pointer 357 deg (contrast 25.7)  ~424mm away
SETUP OK
```

Anything wrong gets an arrow saying what to change: `SMALL, move the camera
closer`, `SOFT, past the fixed-focus limit, back off a little`, `SQUASHED, too
side-on, get more overhead`, `POINTER NOT FOUND, check lighting and glare`.

**Then look at the annotated frame** at `~/images/knobs.jpg`. The circles must sit
on the knobs and each red line on its white pointer. Do not skip this. Every
vision failure on this project reported a healthy-looking number while being
wrong, and drawing the detection back on the frame is what caught all of them.

Knob names are assigned top to bottom, so they stay stable unless the pedal is
rearranged. Distance assumes a 10 mm metal cap: measure yours and edit `CAP_MM`.

## 3. Turn

```bash
ssh pi@<ip> 'python3 ~/turn_knob.py --deg=-90'
```

It measures the knob, grips, turns, releases, parks, measures again, and repeats
until it is within `--tol` (default 8 degrees) or it stops making progress.

Which knob the gripper is on is **not configured**: whichever one moves on the
first bite is the target, and the others become controls. If they move too, the
pedal shifted.

Useful flags: `--tol`, `--tries`, `--squeeze`, and `--open-loop` to do a single
uncorrected bite for comparison in the writeup.

## 4. Read the results

```bash
ssh pi@<ip> 'python3 ~/runs.py'            # every run, newest last
ssh pi@<ip> 'python3 ~/runs.py <stamp>'    # bite by bite
```

Each run leaves `~/runs/<stamp>/` with `run.avi` of the whole attempt, a still per
stage, and `log.json` of every commanded and achieved number.

**Tracking is the number to watch**: the fraction of the commanded roll the knob
actually followed. 100% means the grip is driving the knob. Low means it is
slipping in the jaws, and the grip force reading *cannot see that*, so this is the
only place it shows up. Falling tracking means fix the pose or the gripper, not
add force.

## Why the loop exists

Open loop does not work here, and the failure is invisible from the servos:

```
commanded -90  ->  knob followed -88     good
commanded +90  ->  knob followed +19     slipped
commanded +69  ->  knob followed +28     slipped
```

All three reported `holding at 579, squeeze 70 counts` with no slip detected. The
force check samples while the fingers close, so it is structurally blind to the
knob rotating inside the jaws once torque is applied, and it will keep certifying
a grip that is quietly failing.

The general lesson, and it has now bitten three separate times on this project: a
check that runs at a different moment, or under a different load, than the thing
it certifies is not a check.

## Do not wind the wrist up before gripping

Tempting, since it would let one bite cover the full range. It does not work. The
gripper hangs well off the roll axis, so pre-rolling swings the fingers off the
knob and the grip closes on air. The taught roll is used as-is, capping one bite
near 106 degrees, and the loop covers the rest.

## When the arm vanishes mid-run

```
OSError: read error        ... then ...  OSError: open failed
```

Check the bus first:

```bash
ssh pi@<ip> 'lsusb | grep 0483:5750'
ssh pi@<ip> 'dmesg | tail -5'
```

`usb 1-1.2: USB disconnect` means the controller dropped off, which is a power
event, not software. Squeezing is the largest current draw in the run, so a
sagging battery browns the controller out mid-grip. Check the power switch and
the battery, then power-cycle the arm. `turn_knob.py` logs the battery voltage at
the start of every run so this is diagnosable afterwards instead of looking random.

---

# Height floor and live coverage (added 2026-08-02)

## Height floor: stop the gripper pressing the pedal

Driving the gripper into the pedal shoves it, and once the pedal moves every
taught pose is wrong and the camera calibration with it. `move()` now refuses any
pose below a taught floor.

The floor is taught by **putting the arm at the lowest acceptable height**, not by
typing a number in millimetres:

```bash
python3 ~/arm.py goto grip0
python3 ~/arm.py floor -5        # floor = here, minus 5 mm
python3 ~/arm.py height          # current height and clearance
```

Height comes from the course's own forward kinematics, lifted out of
`xarm_kinematics.py` into `arm.py` so it runs without ROS. **Its absolute accuracy
is unverified and does not matter**: the floor is measured with the same function
as the check, so both carry the same calibration error and it cancels. What matters
is that it is consistent and monotonic, which it is, at about 0.37 mm per count of
shoulder.

A refused move looks like this and returns `False` rather than throwing:

```
REFUSED: that pose puts the gripper at z=61 mm, below the 70 mm floor.
```

## Live coverage: calibration you can watch

```bash
python3 ~/arm.py neutral && python3 ~/align.py base    # arm must be AWAY
python3 ~/align.py watch --knob=knob1                  # then open port 8080
```

Then hand-pose the arm in another terminal (`arm.py teach grip0`) and watch the
percentage climb. `align.py watch` holds the camera; teaching does not use it, so
they run side by side.

Measured behaviour, with the arm parked versus over the target:

```
away      knob1   1%   knob2  1%   knob3  0%
at pose   knob1 100%   knob2 12%   knob3  4%
away      knob1   1%   knob2  2%   knob3  0%
```

Specific to the target, near zero on the neighbours, and it returns to where it
started. Raw readings repeat to about 1 point.

### Two things this got wrong first, both worth keeping

**The gripper DARKENS the knob, it does not whiten it.** The white stickers are on
the gripper's sides; from directly overhead you see its black top and its shadow.
The ring went 58% bright to 0%. A metric keyed on whiteness would have read the
correct position as "further away". Coverage is therefore keyed on **change**, which
works whether the occluder is pale or dark, and survives moving the camera to a
slant.

**Do not normalise by each knob's own baseline.** It looks natural and it is wrong.
A knob whose ring is only 20% bright turns an ordinary change into a full-scale
reading, and the metric reported **100% occlusion on an untouched knob**. A fixed
span is correct here because the raw measurement is stable to about a point.

### What coverage is not

It is not proof of a good grip. The fingers should straddle the knob, not bury it,
so 100% is not the goal. Record the value from a grip that actually worked:

```bash
python3 ~/align.py mark
```

That becomes the number to aim for, learned from a working example rather than
assumed.
