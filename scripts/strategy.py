#!/usr/bin/env python3
"""How hard to bite, and when to stop. No simulator, no hardware, no OpenCV.

This is the part of the run that decides things, kept deliberately free of
everything it decides about. It imports nothing but numpy, so the same code
runs inside the MuJoCo cycle on a laptop and inside turn_knob.py on the Pi.
That matters more than it sounds: logic tuned in simulation is worthless if
the robot ends up running a second, subtly different copy of it.

    plan = Plan(target=90.0)
    while not plan.done:
        bite = plan.next_bite()          # what to command the wrist
        moved = <turn the knob and measure the pointer>
        plan.record(bite, moved)
    plan.why                             # why it stopped, in words

Everything here is a pure function of numbers that came from measurements, so
it can be tested without a robot, and it is (see the self-check at the end).
"""
import numpy as np

TOL = 8.0            # degrees; inside this the knob has arrived
MAX_BITES = 8        # see below: the wrist's per-bite limit sets the floor
# A bite can command at most BITE_LIMIT degrees of wrist, so a grip that
# transmits `eff` moves at most BITE_LIMIT*eff of knob per bite. Reaching a
# 90 degree target through a 20 percent grip therefore needs at least
# 90/(100*0.2) = 5 bites no matter how good the estimate is, and 6 left no
# room for measurement noise. The budget has to be set from that arithmetic,
# not chosen to look tidy.
BITE_LIMIT = 100.0   # the wrist joint's usable range in one command
MIN_BITE = 4.0       # a smaller move is not worth the wear
DEAD_BITE = 3.0      # a bite moving the knob less than this achieved nothing
EFF_FLOOR = 0.12     # never assume the grip is worse than this
EFF_BLEND = 0.6      # weight on a new efficiency estimate
AIM = 0.90           # aim this fraction of the way, so errors undershoot


def wrap(d):
    """Fold an angle difference into [-180, 180)."""
    return (float(d) + 180.0) % 360.0 - 180.0


class Plan:
    """Tracks one knob's progress toward a target rotation."""

    def __init__(self, target, efficiency=None, tol=TOL, max_bites=MAX_BITES):
        self.target = float(target)
        self.tol = tol
        self.max_bites = max_bites
        self.efficiency = efficiency      # None until the first bite teaches us
        self.done_deg = 0.0
        self.bites = []
        self.why = ''

    # ---- state ---------------------------------------------------------
    @property
    def remaining(self):
        return self.target - self.done_deg

    @property
    def arrived(self):
        return abs(self.remaining) <= self.tol

    @property
    def done(self):
        return bool(self.arrived or self.why)

    # ---- decisions -----------------------------------------------------
    def next_bite(self):
        """Degrees to command the wrist, or None if the run should stop.

        Divided by the efficiency estimate so the KNOB moves the remaining
        amount rather than the wrist, then aimed deliberately short.

        Aiming short is what replaces a cap on the bite size. Capping each
        bite at a multiple of the remaining travel looks safer but is not: a
        grip that transmits 35 percent needs nearly 3x the remaining travel,
        so the cap binds first and the measured efficiency is never used.
        Undershooting by a fixed fraction gives the same protection against a
        wrong estimate while keeping it, and leaves any error on the safe side.
        """
        # A reason to stop, once set, has to actually stop the run. Without
        # this, record() could mark the knob dead and next_bite() would hand
        # out another bite regardless, so a knob that never moved was reported
        # as merely "short after 6 bites" and got polished five more times.
        if self.why:
            return None
        if self.arrived:
            self.why = 'arrived'
            return None
        if len(self.bites) >= self.max_bites:
            self.why = (f'still {self.remaining:+.0f} deg short after '
                        f'{self.max_bites} bites')
            return None
        eff = max(self.efficiency if self.efficiency else 1.0, EFF_FLOOR)
        bite = AIM * self.remaining / eff
        bite = float(np.clip(bite, -BITE_LIMIT, BITE_LIMIT))
        if abs(bite) < MIN_BITE:
            self.why = 'the next bite would be too small to be worth making'
            return None
        return bite

    def record(self, commanded, moved):
        """Log what the pointer says actually happened, and learn from it."""
        moved = wrap(moved)
        self.done_deg += moved
        self._learn(commanded, moved)
        self.bites.append(dict(n=len(self.bites) + 1,
                               commanded=round(float(commanded), 1),
                               moved=round(moved, 1),
                               total=round(self.done_deg, 1),
                               efficiency=(None if self.efficiency is None
                                           else round(self.efficiency, 3))))
        if self.arrived:
            self.why = 'arrived'
        elif abs(moved) < DEAD_BITE:
            # Distinguished from "short" on purpose: short means bite again,
            # dead means the jaws are spinning on the cap or the knob is at
            # its end stop, and biting again only polishes it.
            self.why = ('the knob stopped moving: the jaws are slipping on it '
                        'or it has reached its end stop')
        return self.why

    def _learn(self, commanded, moved):
        if abs(commanded) < 1e-6:
            return
        seen = moved / commanded
        if seen <= 0:              # went nowhere, or backwards: not a ratio
            return
        self.efficiency = (seen if self.efficiency is None
                           else (1 - EFF_BLEND) * self.efficiency
                           + EFF_BLEND * seen)


def summary(plan):
    return (f'{plan.done_deg:+.1f} of {plan.target:+.0f} deg in '
            f'{len(plan.bites)} bite(s)'
            + ('' if plan.arrived else f' ({plan.why})'))


if __name__ == '__main__':
    # A knob that transmits `eff` of the wrist's rotation, exactly.
    def simulate(target, eff, **kw):
        p = Plan(target, **kw)
        while True:
            b = p.next_bite()
            if b is None:
                return p
            p.record(b, b * eff)

    print(f'{"grip":>6} {"target":>7} {"bites":>6} {"arrived":>8} {"error":>7}')
    for eff in (1.0, 0.7, 0.5, 0.35, 0.25, 0.15, 0.08):
        p = simulate(90.0, eff)
        print(f'{eff:6.0%} {90:7.0f} {len(p.bites):6d} {str(p.arrived):>8} '
              f'{p.remaining:+6.1f}d')
        if eff >= 0.15:
            assert p.arrived, f'a {eff:.0%} grip should reach 90 deg'

    # never overshoot, in either direction: aiming short is the whole point
    for eff in (0.15, 0.35, 0.6, 1.0):
        for target in (30.0, 90.0, 150.0, -90.0):
            p = simulate(target, eff)
            assert abs(p.done_deg) <= abs(target) + p.tol, \
                f'overshot: {p.done_deg:.1f} past {target} at {eff:.0%} grip'
    print('\nno overshoot at any grip quality, in either direction')

    # a knob that will not move must be called dead, not merely short
    p = simulate(90.0, 0.0)
    assert not p.arrived and 'slipping' in p.why, f'dead knob reported {p.why!r}'
    assert len(p.bites) == 1, f'a dead knob took {len(p.bites)} bites to spot'
    print('a knob that will not move is spotted on the first bite')

    # a wrong starting estimate must be corrected, not trusted
    p = Plan(90.0, efficiency=0.9)          # told the grip is great; it is not
    while True:
        b = p.next_bite()
        if b is None:
            break
        p.record(b, b * 0.3)
    assert p.arrived, 'a bad starting estimate should be corrected, not fatal'
    assert p.efficiency < 0.5, f'never learned: still believes {p.efficiency:.2f}'
    print(f'a wrong starting estimate (90%) corrected to {p.efficiency:.0%} '
          f'and still arrived in {len(p.bites)} bites')

    # the estimate must survive noisy measurements
    rng = np.random.default_rng(0)
    worst = 0
    for trial in range(200):
        eff = rng.uniform(0.2, 0.9)
        p = Plan(rng.choice([60.0, 90.0, 120.0, -90.0]))
        while True:
            b = p.next_bite()
            if b is None:
                break
            p.record(b, b * eff + rng.normal(0, 2.0))    # 2 deg reading noise
        assert p.arrived, f'failed at {eff:.0%} grip with noisy readings'
        worst = max(worst, len(p.bites))
    print(f'200 random grips (20-90%) with 2 deg of reading noise: all arrived, '
          f'worst {worst} bites')
    print('strategy self-checks passed')
