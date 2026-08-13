#!/usr/bin/env python3
"""The grip-and-turn sequence, written once, driven over any transport.

There are two ways to reach this arm. `arm.py` opens the servo bus over HID USB
and talks to it directly, which is what the bench tools and the current demo
use. The course's ROS 2 graph reaches the same servos through `command_xarm`,
which owns the USB itself and exchanges topics with everything else. Both
cannot run at once: one USB device, one controller.

So the sequence lives here and the transport is an argument. A Backend has to
provide six things, and nothing about knobs or bites appears in it:

    taught()            the taught pose, as servo counts
    counts()            where the arm is now, as servo counts
    approach(counts)    fly there, slowly, stopping on contact  -> bool
    squeeze(want)       close until it is pushing  -> (command, force, holding)
    release()           open the jaws
    preclose()          narrow them without gripping
    roll_by(deg)        turn the wrist about its own axis  -> bool
    park()              back to neutral

Writing it twice was the obvious alternative and it is how the logic drifts:
the whole reason strategy.py and regrip.py import nothing but numpy is that a
rule tuned in one place has to be the rule that runs in the other.
"""
import sys

import numpy as np

import ik
import regrip


def try_grip(backend, offset, want_force):
    """Fly to the taught pose plus `offset` mm and squeeze. -> (score, holding).

    score is the squeeze force as a fraction of what was asked for, which is
    the only positional signal either transport has: it falls off as the jaws
    stop straddling the knob. A pose the arm cannot reach scores zero rather
    than raising, because the search should route around it, not stop.
    """
    try:
        target, _ = ik.nudged(backend.taught(), float(offset[0]),
                              float(offset[1]), 0.0)
    except ValueError as e:
        print(f'  offset ({offset[0]:+.1f},{offset[1]:+.1f}) mm unreachable: {e}',
              file=sys.stderr)
        return 0.0, False
    backend.preclose()
    if not backend.approach(target):
        return 0.0, False
    _, force, holding = backend.squeeze(want_force)
    # The miss that actually happens is straight DOWN: the fingers ride the
    # top of the cap and the jaws slide off the metal cone to the travel limit
    # without holding. Seen on video, and the xy search cannot fix it by
    # construction. So before searching sideways, try lower. Two small steps,
    # contact-checked, floor-guarded; on a backend with no vertical nudge the
    # loop just never runs.
    for _ in range(2):
        if holding or not hasattr(backend, 'lower'):
            break
        backend.release()
        if not backend.lower(2.0):
            break
        _, force, holding = backend.squeeze(want_force)
    return min(1.0, force / max(want_force, 1)), holding


def find_grip(backend, want_force, start=None, log=print):
    """Search around the taught pose until something grips. -> offset or None.

    Runs only once a grip has already missed, so it costs nothing on a good
    pose. Measured against the model, 16 directions per error size: a taught
    pose 4 mm out went from 6/16 to 16/16, one 6 mm out from 0/16 to 15/16.

    It does NOT stop at the first grip that holds. That one sits at the edge of
    where the jaws still catch, and re-flying to it held only 8 times in 20
    under the arm's scatter; letting the search finish centres it and the same
    reuse then holds 19 in 20.
    """
    origin = np.zeros(2) if start is None else np.asarray(start, float).copy()
    s = regrip.Search()
    log(f'the grip missed. Searching around the taught pose '
        f'(up to {regrip.MAX_REACH:.0f} mm, {regrip.MAX_TRIES} tries)')
    while True:
        off = s.next_offset()
        if off is None:
            break
        here = np.asarray(off) + origin
        score, holding = try_grip(backend, here, want_force)
        s.record(off, score, holding)
        log(f'  ({here[0]:+5.1f}, {here[1]:+5.1f}) mm  force {score:4.0%}'
            f'  {"HOLDS" if holding else "-"}')
        backend.release()
    if not s.holding:
        log(f'no grip found: {s.why}')
        return None
    best = origin + np.asarray(s.best, float)
    log(f'best grip at ({best[0]:+.1f}, {best[1]:+.1f}) mm from the taught '
        f'pose, in {s.attempts} tries')
    return best


def one_bite(backend, degrees, want_force, offset=None, log=print):
    """Grip, roll by `degrees`, release, park. -> (rolled, offset).

    `rolled` is the roll actually commanded, or None if nothing was turned.
    `offset` comes back so the caller can reuse it: a taught pose that was
    4 mm out on this bite is still 4 mm out on the next one, and re-flying to
    a known offset costs one squeeze instead of a fresh search.
    """
    offset = np.zeros(2) if offset is None else np.asarray(offset, float)
    _, holding = try_grip(backend, offset, want_force)
    if not holding:
        found = find_grip(backend, want_force, start=offset, log=log)
        if found is None:
            backend.release()
            backend.park()
            return None, offset
        offset = found
        # The search let go of everything it tried, so take the winning pose
        # again for real. Worth a few goes: the offset is known good and the
        # only thing in the way is the arm's own scatter, so a miss here is bad
        # luck rather than bad aim, and one squeeze is cheaper than a search.
        for _ in range(3):
            _, holding = try_grip(backend, offset, want_force)
            if holding:
                break
            backend.release()
        if not holding:
            log('the search found a grip but the arm could not land on it '
                'three times running. Something is moving.')
            backend.release()
            backend.park()
            return None, offset
    ok = backend.roll_by(degrees)
    backend.release()
    backend.park()
    return (degrees if ok else None), offset


if __name__ == '__main__':
    # A backend made of arithmetic: no arm, no ROS, no camera. Enough to prove
    # the sequence itself behaves, on either transport.
    class Fake:
        """A knob `err` mm from the taught pose, in a sloppy arm."""

        def __init__(self, err=(0.0, 0.0), scatter=1.5, seed=0,
                     hold_mm=4.0, falloff=9.0):
            self.taught_counts = [524, 475, 620, 505, 551, 500, 500]
            self.err = np.asarray(err, float)
            self.scatter, self.hold_mm, self.falloff = scatter, hold_mm, falloff
            self.rng = np.random.default_rng(seed)
            self.at = list(self.taught_counts)
            self.squeezes = self.rolls = 0

        def taught(self):
            return list(self.taught_counts)

        def counts(self):
            return list(self.at)

        def approach(self, counts):
            self.at = list(counts)
            return True

        def squeeze(self, want):
            self.squeezes += 1
            here = (ik._arm().endpoint(self.at)[:2] * 1000.0
                    + self.rng.normal(0, self.scatter, 2))
            base = ik._arm().endpoint(self.taught_counts)[:2] * 1000.0
            d = float(np.linalg.norm(here - (base + self.err)))
            force = int(round(want * max(0.0, 1.0 - d / self.falloff)))
            return 570, force, d <= self.hold_mm

        def preclose(self):
            return True

        def release(self):
            return True

        def roll_by(self, deg):
            self.rolls += 1
            return True

        def park(self):
            return True

    quiet = lambda *a, **k: None
    print(f'{"taught pose off by":>19} {"turned":>8} {"squeezes":>9}')
    env = {}
    for err in (0, 2, 4, 6, 8, 14):
        won, sq = 0, []
        for k in range(12):
            ang = 2 * np.pi * k / 12
            f = Fake(err=(err * np.cos(ang), err * np.sin(ang)), seed=k)
            rolled, off = one_bite(f, 90.0, 70, log=quiet)
            won += rolled is not None
            sq.append(f.squeezes)
        env[err] = won / 12
        print(f'{err:16.0f} mm {won:6d}/12 {int(np.median(sq)):9d}')

    for err in (0, 2, 4):
        assert env[err] >= 0.9, f'only turned {env[err]:.0%} at {err} mm'
    assert env[6] >= 0.75, f'only turned {env[6]:.0%} at 6 mm'
    assert env[14] <= 0.35, f'turned {env[14]:.0%} at 14 mm, past the leash'

    # A knob that never grips must never be rolled. Turning the wrist while
    # holding nothing is the failure this whole sequence exists to avoid: it
    # looks exactly like success from the servos.
    f = Fake(err=(60.0, 60.0))
    rolled, _ = one_bite(f, 90.0, 70, log=quiet)
    assert rolled is None and f.rolls == 0, \
        f'rolled the wrist {f.rolls} times while holding nothing'
    print(f'\na knob that is not there: {f.squeezes} squeezes, wrist never '
          f'rolled')

    # The offset has to come back out, or every bite pays for the search
    # again. 7 mm rather than 4: at 4 the first attempt sometimes lands inside
    # the hold radius on its own, no search runs, and a zero offset is then the
    # right answer rather than a failure.
    f = Fake(err=(7.0, 0.0), seed=2)
    rolled, off = one_bite(f, 90.0, 70, log=quiet)
    assert rolled is not None and np.linalg.norm(off) > 1.0, \
        f'recovered but reported no offset: {off}'
    first = f.squeezes
    f.squeezes = 0
    rolled2, off2 = one_bite(f, 90.0, 70, offset=off, log=quiet)
    assert rolled2 is not None, 'the learned offset did not grip on reuse'
    assert f.squeezes < first, \
        f'reuse cost {f.squeezes} squeezes against {first} for the search'
    print(f'first bite {first} squeezes, next bite {f.squeezes} reusing '
          f'({off[0]:+.1f}, {off[1]:+.1f}) mm')
    print('\nturn_core self-checks passed')
