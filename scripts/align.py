#!/usr/bin/env python3
"""How much of each knob the gripper is covering, live, while you pose the arm.

  align.py watch [--knob=knob1]     live view + coverage, hold the camera open
  align.py                          print coverage once and exit

Open http://<pi-ip>:8080 to watch. Pair it with teaching in another terminal:
hand-pose the arm and watch the number climb. That is calibration you can see,
instead of guessing and finding out at squeeze time.


The metric
----------
Coverage is how much the ring around a knob has CHANGED from how it looks with the
arm parked away. Zero means the knob is as clear as it was; one means the gripper
is fully over it.

It is deliberately not "how white is it". Measured on this rig, moving the gripper
over a knob sent that ring from 58% bright to 0%, because from directly overhead
you see the black top of the gripper and its shadow, not the white stickers on its
sides. Keying on whiteness would have read that as moving further away. Keying on
CHANGE gets it right whether the occluder is pale or dark, so it survives the
camera being moved to a slant where the stickers do face the lens.

Neighbouring knobs barely move (21->16%, 15->16%), so the signal is specific to
the knob being approached and the others act as controls.

**Angle-independent.** It is an overlap in the image, so overhead or slanted both
work, and nothing here changes if the camera moves.

What it is NOT: proof of a good grip. The fingers should straddle the knob, not
bury it, so maximum coverage is not the goal. The number to aim for is whatever it
reads when a grip actually WORKS, which is why `mark` records it from a working
example rather than assuming 100% is best.
"""
import json, os, sys, time
import numpy as np, cv2
import knob, live

GOOD = os.path.expanduser('~/align_target.json')
BASE = os.path.expanduser('~/align_base.json')
BRIGHT = 150            # a ring pixel this bright is not bare knob any more
SPAN = 0.40             # change in bright-fraction that counts as fully covered
HOLD_LEAD = 0.5         # seconds before Enter to take the pose from


def brightness(frame, k):
    """Fraction of this knob's ring that is bright, 0 to 1. Raw, not coverage."""
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cx, cy, r0 = k['cx'], k['cy'], k['cap_r'] + 2
    r1 = max(r0 + 4, int(round(k['cap_r'] * 1.8)))
    vals = []
    for r in range(r0, r1):
        for d in range(0, 360, 3):
            t = np.deg2rad(d)
            y, x = int(cy + r * np.sin(t)), int(cx + r * np.cos(t))
            if 0 <= y < grey.shape[0] and 0 <= x < grey.shape[1]:
                vals.append(grey[y, x])
    return float(np.mean(np.array(vals) >= BRIGHT)) if vals else 0.0


def all_brightness(frame, knobs):
    return {n: brightness(frame, k) for n, k in knobs.items()}


def all_coverage(frame, knobs, base):
    """0 = knob as clear as the baseline, 1 = fully occluded by the gripper.

    Scaled by a FIXED span, not by each knob's own baseline. Dividing by the
    baseline looks natural and is wrong: a knob whose ring is only 20% bright to
    begin with then turns a 20-point change into a full-scale reading, and this
    metric read 100% occlusion on an untouched knob because of it. The raw
    measurement is stable to about 1 point, so a fixed span is safe.
    """
    now = all_brightness(frame, knobs)
    return {n: float(np.clip(abs(base[n] - now[n]) / SPAN, 0, 1)) for n in now}


def draw(frame, knobs, cov, target, note=''):
    out = frame.copy()
    ref = json.load(open(GOOD)) if os.path.exists(GOOD) else {}
    for n, k in knobs.items():
        is_t = (n == target)
        c = (0, 255, 255) if is_t else (255, 255, 0)
        r1 = max(k['cap_r'] + 6, int(round(k['cap_r'] * 1.8)))
        cv2.circle(out, (k['cx'], k['cy']), k['cap_r'] + 2, c, 1)
        cv2.circle(out, (k['cx'], k['cy']), r1, c, 2 if is_t else 1)
        label = f'{n} {cov[n]*100:.0f}%'
        if is_t and n in ref:
            label += f'  (want {ref[n]*100:.0f}%)'
        cv2.putText(out, label, (k['cx'] - 40, k['cy'] - r1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0) if is_t else (200, 200, 200), 2 if is_t else 1)
    if note:
        cv2.putText(out, note, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2)
    return out


def teach(name, target):
    """Pose the arm by hand with the camera live, then save pose AND coverage.

    One process holds both the arm and the camera. They are different devices, so
    that is fine, and it means the number on screen and the pose being saved are
    the same instant. Running them as two processes invites saving a pose that was
    judged from a frame taken before or after it.
    """
    import threading
    import arm as A

    knobs = json.load(open(knob.CONFIG)) if os.path.exists(knob.CONFIG) else {}
    base = json.load(open(BASE)) if os.path.exists(BASE) else None

    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    state = {'frame': None}
    live.serve(lambda: state['frame'])

    stop = threading.Event()
    # Rolling history of the pose while it is being held. Reading only after Enter
    # is pressed captures the arm AFTER the hand comes off it, and with torque off
    # it sags immediately. That saves a pose lower than the one that was posed,
    # which is how a grip aimed at the middle of the knob ends up near its base.
    from collections import deque
    history = deque(maxlen=40)

    def sample():
        while not stop.is_set():
            try:
                history.append((time.time(), A.read()))
            except Exception:
                pass
            time.sleep(0.1)

    def keep_off():
        # The board drops the unload command if anything else talks to it, so
        # resend it for as long as the arm is being posed.
        while not stop.is_set():
            A.arm.servoOff()
            time.sleep(0.4)

    def stream():
        while not stop.is_set():
            ok, f = cam.read()
            if not ok:
                continue
            cov = all_coverage(f, knobs, base) if (knobs and base) else \
                {n: 0.0 for n in knobs}
            note = (f'{target}: {cov.get(target, 0)*100:.0f}% covered'
                    if knobs and base else
                    ('no knobs calibrated yet' if not knobs else 'no baseline yet'))
            state['frame'] = draw(f, knobs, cov, target, note)
            time.sleep(0.12)

    threading.Thread(target=keep_off, daemon=True).start()
    threading.Thread(target=stream, daemon=True).start()
    threading.Thread(target=sample, daemon=True).start()

    print(f'torque off. Watch http://<pi-ip>:8080 while you pose the arm.')
    print(f'Press Enter to save this pose as "{name}".')
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        stop.set(); cam.release()
        sys.exit('\ncancelled, nothing saved')
    t_enter = time.time()
    stop.set()
    time.sleep(0.4)
    settled = A.read()

    # Take the pose from shortly BEFORE Enter, i.e. while it was still held.
    held = next((p for t, p in reversed(history) if t < t_enter - HOLD_LEAD), None)
    pose = held if held else settled
    if held:
        sag = [b - a for a, b in zip(held, settled)]
        dz = (A.endpoint(settled)[2] - A.endpoint(held)[2]) * 1000
        print(f'held  {held}')
        print(f'after {settled}   (you let go)')
        print(f'sag   {sag}   -> gripper dropped {dz:+.1f} mm')
        if abs(dz) > 1.5:
            print(f'  saving the HELD pose, not the sagged one. Reading after Enter '
                  f'is what put earlier grips low on the knob.')

    frame = state['frame']
    cov = all_coverage(cam.read()[1], knobs, base) if (knobs and base) else {}
    cam.release()

    entry = {'counts': pose}
    if cov:
        entry['coverage'] = cov
    p = A.poses()
    p[name] = entry
    json.dump(p, open(A.POSES, 'w'), indent=1)
    print(f'saved {name} = {pose}')
    if cov:
        print('coverage at this pose: '
              + '  '.join(f'{n} {v*100:.0f}%' for n, v in sorted(cov.items())))

    print('\nchecking the grip has hold of something...')
    _, lag, holding = A.squeeze(A.GRIP_FORCE)
    print('GOOD, the fingers are on the object.' if holding else
          'WARNING: this pose does not survive the run force (see above).')
    t = A.read(); t[A.GRIP] = pose[A.GRIP]
    A.move(t, speed=200)
    if cov and holding:
        json.dump(cov, open(GOOD, 'w'), indent=1)
        print(f'grip worked, so this coverage is now the target ({GOOD})')


def main():
    flag = lambda k, d: next((type(d)(a.split('=')[1])
                              for a in sys.argv[1:] if a.startswith(f'--{k}=')), d)
    target = flag('knob', 'knob1')
    cmd = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else ''

    if cmd == 'teach':
        # Deliberately works with no calibration: posing the arm and saving it is
        # useful even before the camera can see the knobs, and refusing here would
        # block the one thing that always works.
        pose_name = sys.argv[2] if len(sys.argv) > 2 else 'grip0'
        return teach(pose_name, target)

    knobs = json.load(open(knob.CONFIG)) if os.path.exists(knob.CONFIG) else {}
    if not knobs:
        sys.exit('no knobs calibrated. Run:  python3 knob.py calibrate')
    if target not in knobs:
        sys.exit(f'{target} is not in knobs.json ({list(knobs)}). '
                 f'Run:  python3 knob.py calibrate')

    if cmd == 'base':
        # Must be run with the arm parked clear of the pedal, or every later
        # reading is measured against a knob that was already half covered.
        b = all_brightness(knob.grab(), knobs)
        json.dump(b, open(BASE, 'w'), indent=1)
        print('baseline (arm must be AWAY): '
              + '  '.join(f'{n} {v*100:.0f}%' for n, v in sorted(b.items())))
        return

    if not os.path.exists(BASE) and cmd != 'watch':
        sys.exit('no baseline yet. Park the arm clear of the pedal and run:\n'
                 '  python3 arm.py neutral && python3 align.py base')

    if cmd != 'watch':
        base = json.load(open(BASE))
        cov = all_coverage(knob.grab(), knobs, base)
        for n in sorted(cov):
            print(f'  {n:8s} {cov[n]*100:5.1f}% covered'
                  + ('   <- target' if n == target else ''))
        if cmd == 'mark':
            json.dump(cov, open(GOOD, 'w'), indent=1)
            print(f'recorded as the good-grip reference in {GOOD}')
        return

    # watch: hold the camera and stream, so the arm can be posed by hand
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    state = {'frame': None}
    live.serve(lambda: state['frame'])

    if os.path.exists(BASE):
        base = json.load(open(BASE))
        print('using the saved baseline')
    else:
        print('no baseline saved. KEEP THE ARM CLEAR for 3 seconds...')
        for _ in range(20):
            ok, f = cam.read()
        base = all_brightness(f, knobs)
        json.dump(base, open(BASE, 'w'), indent=1)
        print('baseline: ' + '  '.join(f'{n} {v*100:.0f}%'
                                       for n, v in sorted(base.items())))

    print(f'watching {target}. Pose the arm by hand and watch the number.')
    print('Ctrl-C to stop and free the camera.')
    best, last = 0.0, 0.0
    try:
        while True:
            ok, f = cam.read()
            if not ok:
                continue
            cov = all_coverage(f, knobs, base)
            best = max(best, cov[target])
            state['frame'] = draw(f, knobs, cov, target,
                                  f'{target}: {cov[target]*100:.0f}%  best {best*100:.0f}%')
            if abs(cov[target] - last) > 0.03:
                print(f'  {target} {cov[target]*100:5.1f}%   best {best*100:5.1f}%')
                last = cov[target]
            time.sleep(0.15)
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        print('\ncamera released')


if __name__ == '__main__':
    main()
