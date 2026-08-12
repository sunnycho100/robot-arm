#!/usr/bin/env python3
"""Record what a run actually did, in one format, from either transport.

turn_knob.py has always written every attempt into ~/runs/<timestamp>/ and
appended a line to ~/runs/index.jsonl, which is what runs.py reads and what
makes "is the setup getting better or worse" a question with an answer rather
than an opinion. The ROS node had none of it, so a run over topics left no
evidence at all.

The schema here is turn_knob.py's, deliberately and exactly, so runs.py can
compare a run driven over USB against one driven over ROS without knowing or
caring which it is reading. `transport` is the only field that differs, and it
is additive.

    r = Run(target=115.0, tol=8.0, squeeze=70, transport='ros')
    r.start(angles)                       # pointer angles before anything
    r.bite(1, commanded=60, achieved=15, others={...})
    r.finish(angles, knob='knob2')
    print(r.path)
"""
import json
import os
import time

RUNS = os.path.expanduser('~/runs')


class Run:
    """One attempt at a knob, written to disk as it happens.

    Written as it happens, not at the end, because the failure that matters
    most is the one that kills the process: the controller dropped off the USB
    bus mid-squeeze once, and a log assembled only at exit would have lost
    exactly the run worth reading.
    """

    def __init__(self, target, tol=8.0, squeeze=70, open_loop=False,
                 transport='direct', root=RUNS, stamp=None):
        self.stamp = stamp or time.strftime('%Y%m%d-%H%M%S')
        self.root = root
        self.dir = os.path.join(root, self.stamp)
        os.makedirs(self.dir, exist_ok=True)
        self.log = {'stamp': self.stamp, 'target': float(target),
                    'tol': float(tol), 'open_loop': bool(open_loop),
                    'squeeze': squeeze, 'transport': transport, 'bites': []}
        self._flush()

    @property
    def path(self):
        return os.path.join(self.dir, 'log.json')

    def _flush(self):
        with open(self.path, 'w') as f:
            json.dump(self.log, f, indent=1)

    def note(self, **kw):
        """Anything worth knowing about the conditions: battery, offsets, why."""
        self.log.update(kw)
        self._flush()

    def start(self, angles):
        self.log['start'] = {k: float(v['angle']) for k, v in angles.items()}
        self._flush()

    def bite(self, n, commanded, achieved=None, note=None, others=None,
             tracking=None):
        entry = {'n': int(n), 'commanded': float(commanded)}
        entry['achieved'] = None if achieved is None else float(achieved)
        if tracking is not None:
            entry['tracking'] = float(tracking)
        if others:
            entry['others'] = {k: round(float(v), 1) for k, v in others.items()}
        if note:
            entry['note'] = note
        self.log['bites'].append(entry)
        self._flush()

    def still(self, label, frame):
        """Save a frame beside the log. Never let a bad write kill the run."""
        try:
            import cv2
            cv2.imwrite(os.path.join(self.dir, f'{label}.jpg'), frame)
        except Exception:
            pass

    def finish(self, angles=None, knob=None):
        if angles:
            self.log['end'] = {k: float(v['angle']) for k, v in angles.items()}
        if knob and 'start' in self.log and 'end' in self.log:
            final = _wrap(self.log['end'][knob] - self.log['start'][knob])
            self.log.update(knob_turned=knob, final=final,
                            error=self.log['target'] - final,
                            ok=abs(self.log['target'] - final) <= self.log['tol'])
        self._flush()
        # One line per run, so a dozen attempts are a table rather than a dozen
        # files to open.
        try:
            os.makedirs(self.root, exist_ok=True)
            with open(os.path.join(self.root, 'index.jsonl'), 'a') as f:
                f.write(json.dumps({k: self.log.get(k) for k in
                                    ('stamp', 'target', 'open_loop', 'squeeze',
                                     'transport', 'knob_turned', 'final',
                                     'error', 'ok')}) + '\n')
        except Exception:
            pass
        return self.log


def _wrap(d):
    return (float(d) + 180.0) % 360.0 - 180.0


if __name__ == '__main__':
    import shutil
    import tempfile

    tmp = tempfile.mkdtemp()
    r = Run(115.0, transport='ros', root=tmp, stamp='test')
    r.note(battery=8.1)
    r.start({'knob1': {'angle': 10.0}, 'knob2': {'angle': 90.0}})

    # A log has to be readable even if the process dies here, because the run
    # that dies mid-way is the one worth reading.
    mid = json.load(open(r.path))
    assert mid['target'] == 115.0 and mid['battery'] == 8.1, mid
    assert mid['bites'] == [], 'bites should be empty before any are recorded'
    print(f'log is on disk and parseable before the run finishes')

    r.bite(1, commanded=60, achieved=15, tracking=0.25,
           others={'knob1': 0.2})
    r.bite(2, commanded=None if False else 90, achieved=None,
           note='grip failed')
    r.finish({'knob1': {'angle': 10.2}, 'knob2': {'angle': -155.0}},
             knob='knob2')

    got = json.load(open(r.path))
    assert len(got['bites']) == 2, got['bites']
    assert got['bites'][1]['achieved'] is None, 'a failed bite must record None'
    # 90 -> -155 is +115 the short way round, which is the whole point of
    # wrapping: read naively it looks like -245 and the run reports a miss.
    assert abs(got['final'] - 115.0) < 0.001, f'final {got["final"]}'
    assert got['ok'] is True, got
    print(f'wrap handled: 90 -> -155 deg reads as {got["final"]:+.0f}, ok={got["ok"]}')

    index = [json.loads(l) for l in
             open(os.path.join(tmp, 'index.jsonl'))]
    assert len(index) == 1 and index[0]['transport'] == 'ros', index
    print(f'index line: {index[0]}')

    # The schema has to match what runs.py already reads, or a ROS run is
    # invisible to the tool that compares runs.
    need = {'stamp', 'target', 'tol', 'open_loop', 'squeeze', 'bites',
            'start', 'end', 'knob_turned', 'final', 'error', 'ok'}
    missing = need - set(got)
    assert not missing, f'runs.py expects fields that are missing: {missing}'
    print(f'schema carries every field runs.py reads, plus transport')

    shutil.rmtree(tmp)
    print('\nrunlog self-checks passed')
