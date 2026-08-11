#!/usr/bin/env python3
"""Turn several knobs in a row, and keep biting until each one arrives.

Two things this adds over a single cycle.

Which knob. A job names a knob by its role ("tone"), not by a taught pose.
The role maps to a position on the pedal, the pedal's pose comes from the
tags, and the camera confirms which knob is which. Move the pedal and the
same job still targets the same knob.

Trial and error. The knob slips inside the jaws under torque: bench data has
commanded +90 producing +19 and commanded +69 producing +28, while the force
reading says the grip is fine the whole time. So one bite is never enough and
the size of the next bite cannot be guessed in advance. After each bite the
pointer says how far the knob really went, and that ratio becomes the estimate
used to size the next one. The estimate is deliberately blended rather than
replaced, and bites shrink as the target approaches, because the last thing
wanted near the target is a confident overshoot.

    python3 sequence.py                 # the default three-knob job list
    python3 sequence.py --slip 0.25     # pretend the grip is much worse
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / 'scripts'))
import scene
import cycle
import strategy          # the bite-sizing brain, shared with the Pi
from strategy import Plan, wrap

# Bite sizing, the stop rules and the efficiency estimate all live in
# scripts/strategy.py, which imports nothing but numpy so the Pi runs the same
# code. Anything tuned here would otherwise have to be re-tuned there, and the
# two copies would quietly diverge.

# Which knob each role sits on, left to right along the pedal's own axis.
# On the DS-1 the row is tone, level, dist; rename here, not in the code.
ROLES = {'tone': 'knob0', 'level': 'knob1', 'dist': 'knob2'}


class Session:
    """Runs a job list, carrying what it learns from knob to knob."""

    def __init__(self, slip=0.35, noise_px=0.5, seed=0, verbose=True):
        self.slip = slip
        self.noise_px = noise_px
        self.seed = seed
        self.verbose = verbose
        self.efficiency = {}       # knob -> commanded-to-achieved ratio
        self.log = []

    def say(self, *a):
        if self.verbose:
            print(*a)

    def turn_to(self, role, degrees):
        """Bite until this knob has moved `degrees`, or give up with a reason."""
        knob = ROLES.get(role, role)
        rec = dict(role=role, knob=knob, want=degrees, bites=[], ok=False)
        c = cycle.Cycle(noise_px=self.noise_px, seed=self.seed)
        c.slip = self.slip

        try:
            c.knob_angle(knob)
        except KeyError:
            rec['why'] = f'no knob for role {role!r}'
            self.say(f'{role}: {rec["why"]}')
            c.cam.close()
            return rec

        # Carry the efficiency learned on earlier knobs as a starting guess:
        # they share a gripper, so the first bite on knob three should not be
        # as ignorant as the first bite on knob one.
        plan = Plan(degrees, efficiency=self.efficiency.get('shared'))
        self.say(f'\n{role} ({knob}): want {degrees:+.0f} deg'
                 + ('' if plan.efficiency is None
                    else f' (starting from a {plan.efficiency:.0%} grip estimate)'))

        while True:
            bite = plan.next_bite()
            if bite is None:
                break
            before = c.knob_angle(knob)
            r = c.run(knob, bite, verbose=False)
            if not r['stages'] or not all(s['ok'] for s in r['stages'][:5]):
                last = r['stages'][-1]
                plan.why = f'{last["name"]}: {last["note"]}'
                self.say(f'  stopped: {plan.why}')
                break
            moved = wrap(c.knob_angle(knob) - before)
            was = plan.efficiency
            plan.record(bite, moved)
            self.say(f'  bite {len(plan.bites)}: commanded {bite:+6.1f}, knob '
                     f'moved {moved:+6.1f}, total {plan.done_deg:+6.1f} of '
                     f'{degrees:+.0f}   grip {plan.efficiency or 1:.0%}'
                     + ('' if was is None else f' (was {was:.0%})'))

        if plan.efficiency:
            self.efficiency['shared'] = plan.efficiency
            self.efficiency[knob] = plan.efficiency
        rec.update(bites=plan.bites, ok=plan.arrived,
                   done=round(plan.done_deg, 1),
                   error=round(plan.remaining, 1), why=plan.why)
        self.say(f'  -> {"reached" if rec["ok"] else "gave up"}: '
                 f'{strategy.summary(plan)}')
        c.cam.close()
        self.log.append(rec)
        return rec

    def run(self, jobs):
        """jobs: [(role, degrees), ...]. -> the per-knob records."""
        out = [self.turn_to(role, deg) for role, deg in jobs]
        done = sum(1 for r in out if r['ok'])
        self.say(f'\n{done} of {len(out)} knobs reached target')
        return out


if __name__ == '__main__':
    slip = 0.35
    if '--slip' in sys.argv:
        slip = float(sys.argv[sys.argv.index('--slip') + 1])

    scene.reset()
    print(f'grip efficiency in this run: {slip:.0%} '
          f'(the knob turns this fraction of what the wrist does)')
    s = Session(slip=slip)
    jobs = [('tone', 90.0), ('level', -60.0), ('dist', 120.0)]
    out = s.run(jobs)
    assert all(r['ok'] for r in out), 'the nominal job list should complete'

    # The point of the learning: later knobs need fewer bites than the first,
    # because the efficiency estimate carries over as a starting guess.
    first, last = out[0], out[-1]
    print(f'\nbites: {[len(r["bites"]) for r in out]} '
          f'(first knob {len(first["bites"])}, last {len(last["bites"])})')

    # A grip so poor that the target cannot be reached must be reported, not
    # silently accepted. This is the case that matters on the bench.
    print('\nwith a hopeless grip (4%):')
    bad = Session(slip=0.04, verbose=False)
    r = bad.turn_to('level', 120.0)
    print(f'  {r["done"]:+.0f} of {r["want"]:+.0f} deg after {len(r["bites"])} '
          f'bites -> {"reached" if r["ok"] else "reported: " + r.get("why","")}')
    assert not r['ok'], 'a 4 percent grip should not reach a 120 degree target'

    # and a healthy grip should need very few bites
    print('\nwith a perfect grip (100%):')
    good = Session(slip=1.0, verbose=False)
    r = good.turn_to('level', 90.0)
    print(f'  reached in {len(r["bites"])} bite(s)')
    assert r['ok'] and len(r['bites']) <= 2, \
        f'a perfect grip should take one or two bites, took {len(r["bites"])}'

    # A knob that does not move at all must be called out as dead rather than
    # merely "short": the two need different responses on the bench. Short
    # means bite again; dead means the jaws are spinning on the cap or the
    # knob has hit its end stop, and biting again just polishes it.
    print('\nwith a knob that will not move at all (0%):')
    dead = Session(slip=0.0, verbose=False)
    r = dead.turn_to('level', 90.0)
    print(f'  {len(r["bites"])} bite(s), reported: {r.get("why","")}')
    assert not r['ok'], 'a knob that never moves should not report success'
    assert 'slipping' in r.get('why', '') or 'end stop' in r.get('why', ''), \
        f'a dead knob was reported as {r.get("why")!r}, not as dead'
    assert len(r['bites']) <= 2, \
        f'a dead knob should be spotted immediately, took {len(r["bites"])} bites'
    print('\nsequence self-checks passed')
