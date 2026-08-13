#!/usr/bin/env python3
"""Where to try next when the grip misses. No simulator, no hardware, no OpenCV.

The taught pose is posed by hand and the arm lands about 1.5 mm from wherever
it is sent, so the first attempt at a knob can simply be in the wrong place.
Until now that was the end of the road: `one_bite()` squeezed, found it was
holding nothing, and gave up, and after two of those the run stopped.

This decides where to move instead. It is a pattern search, deliberately, and
not a gradient step: the only thing either the bench or the model can hand back
is ONE NUMBER per attempt (how hard the servo is pushing), and getting a
direction out of a scalar means sampling around the current point rather than
differentiating it.

    s = Search()
    while True:
        off = s.next_offset()            # (dx, dy) mm from the taught pose
        if off is None: break
        s.record(off, force, holding)    # what that pose actually gripped
    s.best                               # the offset to keep

Same file runs in the simulation and on the Pi, for the reason strategy.py is
written the same way: logic tuned against a model is worth nothing if the robot
ends up running a second, subtly different copy of it.
"""
import numpy as np

STEP0 = 4.0          # mm, the first ring's radius, and the hunt's step outward
STEP_MIN = 1.0       # mm, below this the arm's own scatter swamps the step
SHRINK = 0.5         # how much the ring tightens once something has held
MAX_REACH = 12.0     # mm, never wander further than this from the taught pose
MAX_TRIES = 13       # each try is an approach and a squeeze: ~40 seconds on
                     # the real arm, so the flat-miss bailout below matters
                     # more than this ceiling does.
# A score this good stops the search early. It is a HOLD margin, not a target:
# the force fades over about 9 mm and the jaws lose the knob at 4, so 0.65 is
# roughly 3 mm off the axis, comfortably inside. Set at 0.80 the search almost
# never cleared the bar and ran its whole budget every time, turning a recovery
# into a two-minute event for no extra reliability.
GOOD_ENOUGH = 0.65

# The search runs coarse to fine, and the order matters more than the pattern.
#
# It used to tighten the ring whenever a lap found nothing, which is right for
# refining a grip that already works and exactly backwards for finding one that
# does not: a pose 8 mm out was answered by looking 1.5 mm from where it had
# already failed. Recovery went from 24/24 at 4 mm to 6/24 at 8 mm because of
# it. So while nothing has held, the ring WIDENS; the moment something holds,
# it tightens around that.
def _ring(n):
    a = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.stack([np.cos(a), np.sin(a)], axis=1)


HUNT_POINTS = 6      # a hexagon: the sparsest ring that leaves no gap wider
FINE_POINTS = 8      # than the distance a knob can be grabbed from


class Search:
    """Hunts for an offset from the taught pose that actually grips.

    Scores are whatever the caller can measure, as long as bigger is better and
    it is comparable between attempts. On the Pi that is the squeeze force in
    servo counts, normalised; in the model it is how squarely the jaws straddle
    the knob.
    """

    def __init__(self, step=STEP0, max_tries=MAX_TRIES, reach=MAX_REACH,
                 good_enough=GOOD_ENOUGH):
        self.step = float(step)
        self.max_tries = int(max_tries)
        self.reach = float(reach)
        self.good_enough = float(good_enough)
        self.centre = np.zeros(2)        # best offset found so far
        self.best_score = -np.inf
        self.best = np.zeros(2)
        self.holding = False
        self.tried = []                  # [(offset, score, holding)]
        self.why = ''
        self._queue = []                 # what is left of the current lap
        self._radius = 0.0               # how far out the hunt has reached
        self._started = False

    # ---- state ---------------------------------------------------------
    @property
    def done(self):
        return bool(self.why)

    @property
    def attempts(self):
        return len(self.tried)

    # ---- decisions -----------------------------------------------------
    def next_offset(self):
        """The next (dx, dy) in mm to try, or None if the search should stop."""
        if self.why:
            return None
        if self.holding and self.best_score >= self.good_enough:
            self.why = 'found a grip worth keeping'
            return None
        if len(self.tried) >= self.max_tries:
            self.why = (f'gave up after {self.max_tries} attempts. Re-teach '
                        f'the pose: the search only covers a few mm and this '
                        f'is further out than a nudge can fix')
            return None
        # A miss the search CAN fix is sideways, and a sideways miss has a
        # gradient: probes on the knob side score visibly higher. Timed on the
        # bench, a vertical miss (a limp-taught pose droops ~7 mm under load)
        # has no gradient at all, every probe of a full ring reads the same
        # ~33%, and the search ground through ten minutes of 40-second probes
        # learning nothing. Seven flat probes say the problem is not in this
        # plane: stop and say so, instead of finishing the lap on principle.
        if (not self.holding and len(self.tried) >= 1 + HUNT_POINTS
                and self.best_score < 0.4):
            self.why = ('nothing anywhere near the pose even begins to grip: '
                        'the miss is probably vertical, which this search '
                        'cannot fix. Re-teach, then verify with  arm.py settle')
            return None
        if not self._started:
            self._started = True
            return np.zeros(2)           # the taught pose gets the first go
        if not self._queue:
            if not self._new_lap():
                return None
        return self._queue.pop(0)

    def _new_lap(self):
        """Lay out the next ring. False when there is nowhere left worth going."""
        for _ in range(8):                       # bounded: each pass changes a radius
            if self.holding:
                # Refining: tighten around the pose that actually gripped.
                self.step *= SHRINK
                if self.step < STEP_MIN:
                    self.why = (f'the ring shrank to {self.step:.1f} mm, finer '
                                f'than the arm can place itself')
                    return False
                ring = [self.centre + self.step * d for d in _ring(FINE_POINTS)]
            else:
                # Hunting: widen. The knob is not where we were told it is.
                self._radius += STEP0
                if self._radius > self.reach:
                    self.why = (f'searched out to {self.reach:.0f} mm and found '
                                f'no grip: re-teach the pose, it is further out '
                                f'than a nudge can fix')
                    return False
                ring = [self._radius * d for d in _ring(HUNT_POINTS)]
            # Past the leash is not worth an approach and a squeeze. There
            # used to be a de-duplicator here too, skipping places already
            # tried. It read as an obvious saving and measured as nothing at
            # all: 131 of 144 recoveries either way, 5.83 mean attempts against
            # 5.82. Deleted rather than kept as untested cleverness.
            self._queue = [p for p in ring
                           if np.linalg.norm(p) <= self.reach + 1e-9]
            if self._queue:
                # Try the promising side of the ring FIRST. A miss still
                # carries information: the squeeze force falls off with
                # distance, so the best-scoring attempt so far points roughly
                # where the knob is. This changes no attempt's position and
                # adds none, it only changes the order, which is worth having
                # because the budget usually runs out mid-ring.
                lead = self._hint()
                if lead is not None:
                    self._queue.sort(key=lambda p: -float(np.dot(
                        p / max(np.linalg.norm(p), 1e-9), lead)))
                return True
        self.why = 'nowhere left to try'
        return False

    def _hint(self):
        """Unit vector toward the best-scoring attempt so far, or None."""
        scored = [(s, np.asarray(o)) for o, s, _ in self.tried
                  if np.isfinite(s) and np.linalg.norm(o) > 1e-9]
        if not scored:
            return None
        best = max(scored, key=lambda t: t[0])[1]
        n = np.linalg.norm(best)
        return best / n if n > 1e-9 else None

    def record(self, offset, score, holding):
        """What that offset actually produced.

        A grip that HOLDS always beats one that does not, whatever the scores
        say. Ranking on the raw number alone lets a hard shove against the
        pedal, which reads as plenty of force, outrank a real but gentle grip
        on the knob.
        """
        offset = np.asarray(offset, dtype=float)
        score = float(score)
        self.tried.append((tuple(offset), score, bool(holding)))
        better = ((holding, score) > (self.holding, self.best_score))
        if better:
            self.holding, self.best_score, self.best = bool(holding), score, offset
            # Only re-centre on a pose that actually held. Chasing the best
            # NUMBER walks the search toward whatever reads hardest, which off
            # the knob is the pedal.
            if holding:
                self.centre = offset
                self._queue = []         # re-lap around the new centre
                self.step = STEP0 * 2    # halved on the first refining lap
        return self.why

    def summary(self):
        where = f'({self.best[0]:+.1f}, {self.best[1]:+.1f}) mm'
        if not self.holding:
            return f'no grip found in {self.attempts} attempts ({self.why})'
        return (f'gripped at {where} after {self.attempts} attempt(s)'
                + ('' if not self.why else f' [{self.why}]'))


if __name__ == '__main__':
    # A knob that grips well within GRIP_R of its axis and not at all outside.
    # The score falls off with distance, which is what the squeeze force does.
    GRIP_R, FALLOFF = 4.0, 9.0

    def bench(true_err, noise=0.0, seed=0, **kw):
        """The taught pose is `true_err` mm off. Can the search find the knob?"""
        rng = np.random.default_rng(seed)
        s = Search(**kw)
        while True:
            off = s.next_offset()
            if off is None:
                break
            landed = off + rng.normal(0, noise, 2)      # the arm's own scatter
            d = np.linalg.norm(landed - np.asarray(true_err, dtype=float))
            score = max(0.0, 1.0 - d / FALLOFF)
            s.record(off, score, d <= GRIP_R)
        return s

    # 1. a pose that is already right must not go wandering
    s = bench((0.0, 0.0))
    assert s.holding and s.attempts == 1, \
        f'a good pose took {s.attempts} attempts: {s.summary()}'
    print(f'a pose that already works is left alone: {s.summary()}')

    # 2. the recovery envelope, which is the number that matters
    print(f'\n{"true error":>11} {"recovered":>10} {"attempts (median)":>18}')
    envelope = {}
    for err in (0, 2, 4, 6, 8, 10, 12, 16):
        ok, tries = 0, []
        for k in range(24):
            ang = 2 * np.pi * k / 24
            s = bench((err * np.cos(ang), err * np.sin(ang)), noise=1.5, seed=k)
            ok += s.holding
            if s.holding:
                tries.append(s.attempts)
        envelope[err] = ok / 24
        med = f'{int(np.median(tries))}' if tries else '-'
        print(f'{err:8.0f} mm {ok:7d}/24 {med:>18}')

    # A hand-taught pose is off by a few mm, not a centimetre, so the bar is
    # set where the real errors live and the tail is reported rather than
    # promised. Past the leash the search is SUPPOSED to fail: wandering across
    # the pedal looking for a knob is worse than saying "re-teach this".
    for err in (0, 2, 4, 6):
        assert envelope[err] >= 0.95, \
            f'only recovered {envelope[err]:.0%} of {err} mm errors'
    assert envelope[8] >= 0.75, \
        f'only recovered {envelope[8]:.0%} of 8 mm errors'
    assert envelope[16] <= 0.10, \
        (f'recovered {envelope[16]:.0%} of 16 mm errors, well past the '
         f'{MAX_REACH:.0f} mm leash: the search is wandering')

    # 3. without the search, one shot only: this is what it is worth
    solo = {}
    for err in (0, 2, 4, 6, 8):
        ok = sum(bench((err*np.cos(2*np.pi*k/24), err*np.sin(2*np.pi*k/24)),
                       noise=1.5, seed=k, max_tries=1).holding
                 for k in range(24))
        solo[err] = ok / 24
    print(f'\n{"true error":>11} {"one shot":>10} {"with search":>13}')
    for err in (0, 2, 4, 6, 8):
        print(f'{err:8.0f} mm {solo[err]:9.0%} {envelope[err]:12.0%}')
    assert solo[6] < 0.5 < envelope[6], \
        'the search has to beat a single attempt where it matters'

    # 4. it must stop, and say why, rather than grinding
    s = bench((40.0, 40.0), noise=1.5)          # nowhere near the knob
    assert not s.holding and s.why, 'a hopeless search did not stop cleanly'
    assert s.attempts <= MAX_TRIES, f'{s.attempts} attempts past the budget'
    print(f'\na knob that is not there: {s.summary()}')

    # 6. The leash is about where it will GO, not what it finds. The try budget
    #    runs out before 12 mm anyway, so a recovery rate cannot test this:
    #    raising MAX_REACH to 60 changed no outcome at all and the check stayed
    #    green. Ask the search directly instead.
    # A generous budget on purpose: with the normal 13 tries the budget runs
    # out before the leash does, so a default Search never exercises the leash
    # at all. That is also why raising MAX_REACH to 60 changed no outcome.
    far = Search(reach=12.0, max_tries=500)
    walked = []
    while True:
        off = far.next_offset()
        if off is None:
            break
        walked.append(np.linalg.norm(off))
        far.record(off, 0.0, False)          # nothing ever grips
    assert max(walked) <= 12.0 + 1e-6, \
        f'proposed a pose {max(walked):.1f} mm out, past its own 12 mm leash'
    assert far.why and 'Re-teach' in far.why or 're-teach' in far.why, \
        f'a search that found nothing should say to re-teach, not {far.why!r}'
    print(f'never proposed further than {max(walked):.0f} mm, then said to '
          f're-teach')

    # 7. Force is not the same thing as a grip. Driving the fingers into the
    #    pedal beside the knob reads as plenty of squeeze and holds nothing,
    #    and ranking on the number alone walks the search straight into it.
    #
    #    The decoy has to be BOTH sampled before the real grip AND score higher
    #    than it, or the ranking rule is never actually exercised. An earlier
    #    version put the knob where the ring happened to look first and scored
    #    it 1.0, so the right answer won for the wrong reason and inverting the
    #    rule changed nothing.
    def with_decoy():
        s = Search()
        while True:
            off = s.next_offset()
            if off is None:
                return s
            if np.linalg.norm(off - np.array([4.0, 0.0])) < 1.0:
                s.record(off, 0.95, False)       # jammed against the pedal
                continue
            d = np.linalg.norm(off - np.array([0.0, 6.0]))
            s.record(off, max(0.0, 1.0 - d / FALLOFF), d <= GRIP_R)
    s = with_decoy()
    scores = {tuple(np.round(o, 1)): (sc, h) for o, sc, h in s.tried}
    assert (4.0, 0.0) in scores and scores[(4.0, 0.0)][0] > 0.8, \
        'the decoy was never sampled, so this proves nothing'
    assert s.holding, 'the decoy beat a real grip outright'
    assert np.linalg.norm(np.asarray(s.best) - np.array([0.0, 6.0])) < 4.5, \
        (f'ended at {np.round(s.best,1)}, drawn to the high-force pose that '
         f'held nothing rather than to the knob at (0, +6)')
    print('a hard shove that grips nothing does not outrank a real grip')

    # 8. Refining has to bottom out. Given an endless budget and a grip that
    #    never quite clears the bar, the ring halves forever; below the arm's
    #    own scatter every further lap is measuring noise. The normal 13-try
    #    budget hides this, which is why it gets its own generous-budget test.
    endless = Search(max_tries=10_000, good_enough=2.0)   # never satisfied
    while True:
        off = endless.next_offset()
        if off is None:
            break
        endless.record(off, 0.5, True)                    # always holds, never enough
    assert 'shrank' in endless.why, \
        f'the refine loop stopped for the wrong reason: {endless.why!r}'
    # Bound it by the arithmetic, not by a round number: the ring starts at
    # 2*STEP0 and halves, so it bottoms out in log2(2*STEP0/STEP_MIN) laps of
    # FINE_POINTS each. Checking only that it stopped SOMEWHERE lets the floor
    # be lowered to a micron and the message still reads 'shrank'.
    laps = int(np.ceil(np.log2(2 * STEP0 / STEP_MIN)))
    assert endless.attempts <= 1 + laps * FINE_POINTS, \
        (f'took {endless.attempts} attempts to bottom out, more than the '
         f'{1 + laps * FINE_POINTS} that halving from {2*STEP0:.0f} mm down to '
         f'{STEP_MIN:.0f} mm allows: the floor is lower than it claims')
    print(f'refining bottoms out at {STEP_MIN:.0f} mm after '
          f'{endless.attempts} attempts, rather than chasing noise')
    print('\nregrip self-checks passed')
