#!/usr/bin/env python3
"""Turn a pedal knob to a target angle, watching the knob and re-trying.

  turn_knob.py [--deg=115] [--tol=8] [--tries=4] [--squeeze=70] [--open-loop]

Every run records itself into ~/runs/<timestamp>/: a video of the whole attempt,
a still at each stage, and log.json with every commanded and achieved number. One
line per run is appended to ~/runs/index.jsonl, so successive attempts can be
compared and the setup actually improved instead of guessed at.


Why this is a loop and not a single turn
----------------------------------------
Open loop does not work on this rig, and the failure is invisible from the servos.
Measured in one session:

    commanded -90  ->  knob followed -88     good
    commanded +90  ->  knob followed +19     slipped
    commanded +69  ->  knob followed +28     slipped

All three reported `holding at 579, squeeze 70 counts` with no slip detected. The
grip force check samples while closing, so it is structurally blind to the knob
rotating inside the jaws once torque is applied. It will keep certifying a grip
that is quietly failing.

So the servo angle is not evidence that the knob moved. The knob is. Each pass
measures what actually happened and re-turns by what is left, which converges
whether the grip slips 0% or 70%, and stops rather than grinding when it is
clearly getting nowhere.


Why it does not wind the wrist up first
---------------------------------------
Tempting, since it would let one bite cover the full range. It does not work: the
gripper hangs well off the roll axis, so pre-rolling swings the fingers clean off
the knob and the grip closes on air. The taught pose's roll is used as-is, which
caps a single bite at about 106 degrees, and the loop covers the rest.
"""
import json, os, sys, threading, time
import numpy as np
import cv2
import arm as A
import knob
import strategy          # bite sizing, shared with the simulation

flag = lambda k, d: next((type(d)(a.split('=')[1])
                          for a in sys.argv[1:] if a.startswith(f'--{k}=')), d)
POSE = flag('pose', 'grip0')
TARGET = flag('deg', 115.0)          # degrees to move the knob, signed, image frame
TOL = flag('tol', 8.0)               # close enough, in degrees
TRIES = flag('tries', 4)
SQUEEZE = flag('squeeze', A.GRIP_FORCE)
OPEN_LOOP = '--open-loop' in sys.argv[1:]      # one bite, no checking. For comparison.

RUNS = os.path.expanduser('~/runs')
DEAD_BITE = 5.0                      # a bite that moves the knob less than this is dead
LOW_BATTERY = 7.2                    # below this, expect brownouts under grip load


class Session:
    """Owns the camera for the whole run.

    One process only can hold the device, so the video, the stills and the angle
    measurements all have to come from the same capture. A background thread keeps
    the newest frame available and writes the video at the same time.
    """

    def __init__(self, outdir, fps=10.0):
        self.dir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.cam = cv2.VideoCapture(0)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
        ok, frame = False, None
        for _ in range(8):
            ok, frame = self.cam.read()
        if not ok:
            self.cam.release()
            raise SystemExit('no frame from the camera. Is camstream.py holding it? '
                             'Stop it with:  pkill -f "[c]amstream"')
        self._frame = frame
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.video = cv2.VideoWriter(os.path.join(outdir, 'run.avi'),
                                     cv2.VideoWriter_fourcc(*'MJPG'), fps,
                                     (frame.shape[1], frame.shape[0]))
        self._t = threading.Thread(target=self._pump, args=(fps,), daemon=True)
        self._t.start()

    def _pump(self, fps):
        while not self._stop.is_set():
            ok, f = self.cam.read()
            if ok:
                with self._lock:
                    self._frame = f
                self.video.write(f)
            time.sleep(1.0 / fps)

    def latest(self):
        with self._lock:
            return self._frame.copy()

    def still(self, label):
        cv2.imwrite(os.path.join(self.dir, f'{label}.jpg'), self.latest())

    def angles(self):
        """Pointer angle per knob, from the newest frame."""
        return knob.measure(self.latest())

    def close(self):
        self._stop.set()
        self._t.join(timeout=2)
        self.video.release()
        self.cam.release()


def wrap(d):
    return (d + 180) % 360 - 180


def pick_knob(before, after):
    """Whichever knob actually moved is the one the gripper is on.

    Nothing needs to be configured: the first bite identifies the target, and the
    others double as controls. A knob that moves while the gripper is elsewhere
    means the pedal shifted, which is worth knowing.
    """
    moved = {k: abs(wrap(after[k]['angle'] - before[k]['angle'])) for k in before}
    best = max(moved, key=moved.get)
    return (best, moved) if moved[best] >= DEAD_BITE else (None, moved)


def one_bite(deg):
    """Grip, roll by deg, release, park. Returns the roll actually commanded.

    The jaws are narrowed BEFORE the arm goes down. Simulation showed the
    fingers splay to 88 mm when open against knobs 24 mm apart, so descending
    open drives them into both neighbours; the count itself is a bench
    measurement (see arm.PRECLOSE), only the direction came from simulation.
    """
    A.preclose()
    A.approach(A.counts_of(A.poses()[POSE]), speed=120)
    cmd, lag, holding = A.squeeze(SQUEEZE)
    if not holding:
        A.release()
        A.move(A.NEUTRAL, speed=150)
        return None
    ok = A.turn_by(deg, speed=200)
    A.release()
    A.move(A.NEUTRAL, speed=150)
    return deg if ok else None


def main():
    stamp = time.strftime('%Y%m%d-%H%M%S')
    sess = Session(os.path.join(RUNS, stamp))
    plan = strategy.Plan(TARGET, tol=TOL)
    log = {'stamp': stamp, 'target': TARGET, 'tol': TOL,
           'open_loop': OPEN_LOOP, 'squeeze': SQUEEZE, 'bites': []}
    target_knob, dead = None, 0

    try:
        # Squeezing is the biggest current draw in the run, and a sagging pack
        # browns the controller out and drops it off the USB bus mid-grip. That
        # has happened. Recording the voltage makes it diagnosable afterwards
        # instead of looking like a random crash.
        try:
            log['battery'] = v = float(A.arm.getBatteryVoltage())
            print(f'battery {v:.2f} V')
            if v < LOW_BATTERY:
                print(f'  battery is low. Expect brownouts under grip load, '
                      f'which drop the arm off USB. Charge or swap it.',
                      file=sys.stderr)
        except Exception as e:
            print(f'could not read the battery ({e})', file=sys.stderr)

        A.move(A.NEUTRAL, speed=150)
        sess.still('00_start')
        start = sess.angles()
        if not start:
            raise SystemExit('no knobs seen. Run:  python3 knob.py calibrate')
        for k, v in sorted(start.items()):
            print(f'  {k}: {v["angle"]:6.1f} deg  contrast {v["contrast"]:5.2f}')
        log['start'] = {k: v['angle'] for k, v in start.items()}

        for i in range(1, (1 if OPEN_LOOP else TRIES) + 1):
            now = sess.angles()
            done = wrap(now[target_knob]['angle'] - start[target_knob]['angle']) \
                if target_knob else 0.0
            remaining = TARGET - done
            print(f'\n=== bite {i}: {done:+.0f} of {TARGET:+.0f} done, '
                  f'{remaining:+.0f} to go ===')
            if not OPEN_LOOP and abs(remaining) <= TOL:
                print('within tolerance, stopping')
                break

            # Bite sizing comes from strategy.py, which the simulation drives
            # too, so the logic tuned there is the logic that runs here rather
            # than a second copy that drifts. It divides by the measured grip
            # efficiency, so the KNOB moves the remaining amount rather than
            # the wrist, and aims deliberately short so any error undershoots.
            #
            # Capping the bite at the remaining travel, which is what this line
            # used to do, cannot work through a slipping grip: at the measured
            # 20 to 35 percent tracking the wrist has to turn three to five
            # times further than the knob needs to go.
            # The CAMERA is the authority on how far the knob has come: it is
            # re-read every pass and cannot drift, whereas a running total
            # inside the plan would accumulate every rejected or missed
            # reading. So the plan's progress is overwritten from the camera
            # here, and the plan is left to do only what it is for, which is
            # deciding how big the next bite should be.
            plan.target, plan.done_deg = TARGET, done
            plan.why = ''
            bite = plan.next_bite()
            if bite is None:
                print(f'stopping: {plan.why}')
                break
            before = now
            got = one_bite(bite)
            sess.still(f'{i:02d}_after_bite')
            after = sess.angles()

            if got is None:
                print('the grip failed, so nothing was turned', file=sys.stderr)
                log['bites'].append({'n': i, 'commanded': bite, 'achieved': None,
                                     'note': 'grip failed'})
                dead += 1
            else:
                if target_knob is None:
                    target_knob, moved = pick_knob(before, after)
                    if target_knob is None:
                        raise SystemExit(
                            f'no knob moved measurably: {moved}. The gripper is '
                            f'not on a knob, or the turn slipped completely.')
                    print(f'the gripper is on {target_knob}')
                achieved = wrap(after[target_knob]['angle']
                                - before[target_knob]['angle'])
                # A pointer reader can lock onto the wrong feature and report a
                # rotation the wrist could not have produced. Learning from
                # that would poison the estimate and send the next bite the
                # wrong way, so an impossible reading is discarded.
                if plan.implausible(bite, achieved):
                    print(f'  ignoring an impossible reading: commanded '
                          f'{bite:+.0f} but the pointer claims {achieved:+.0f}',
                          file=sys.stderr)
                    log['bites'].append({'n': i, 'commanded': bite,
                                         'achieved': achieved,
                                         'note': 'rejected as impossible'})
                    continue
                plan.record(bite, achieved)      # learns the grip efficiency
                track = achieved / bite if bite else 0.0
                print(f'commanded {bite:+.0f}, knob followed {achieved:+.0f} '
                      f'({track*100:.0f}% tracking)')
                others = {k: round(wrap(after[k]['angle'] - before[k]['angle']), 1)
                          for k in after if k != target_knob}
                print(f'  other knobs moved {others}  (should be near zero)')
                log['bites'].append({'n': i, 'commanded': bite,
                                     'achieved': achieved, 'tracking': track,
                                     'others': others})
                dead = dead + 1 if abs(achieved) < DEAD_BITE else 0

            if dead >= 2:
                print('two passes in a row moved nothing. Stopping rather than '
                      'grinding: re-teach the pose or improve the grip.',
                      file=sys.stderr)
                break

        end = sess.angles()
        sess.still('99_end')
        log['end'] = {k: v['angle'] for k, v in end.items()}
        if target_knob:
            final = wrap(end[target_knob]['angle'] - start[target_knob]['angle'])
            log.update(knob_turned=target_knob, final=final,
                       error=TARGET - final, ok=abs(TARGET - final) <= TOL)
            print(f'\n{target_knob}: asked {TARGET:+.0f}, got {final:+.0f}, '
                  f'error {TARGET - final:+.0f} deg over '
                  f'{len(log["bites"])} bite(s)')
            print('WITHIN TOLERANCE' if log['ok'] else 'MISSED')
    finally:
        # Park the arm if the link is still alive, but never let a dead link stop
        # the run being written to disk. The controller dropped off the USB bus
        # mid-squeeze once, and the cleanup then threw on top of it and lost the
        # whole log, which is exactly when the log is worth most.
        try:
            A.release()
            A.move(A.NEUTRAL, speed=150)
        except Exception as e:
            print(f'could not park the arm: {e.__class__.__name__}: {e}\n'
                  f'  Check the arm has power and is still on the USB bus:\n'
                  f'    lsusb | grep 0483:5750', file=sys.stderr)
            log['cleanup_failed'] = f'{e.__class__.__name__}: {e}'
        sess.close()
        json.dump(log, open(os.path.join(sess.dir, 'log.json'), 'w'), indent=1)
        with open(os.path.join(RUNS, 'index.jsonl'), 'a') as f:
            f.write(json.dumps({k: log.get(k) for k in
                                ('stamp', 'target', 'open_loop', 'squeeze',
                                 'knob_turned', 'final', 'error', 'ok')}) + '\n')
        print(f'run recorded in {sess.dir}  (run.avi, stills, log.json)')


if __name__ == '__main__':
    main()
