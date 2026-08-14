#!/usr/bin/env python3
"""The TUI: drawing it, and reading what was typed. No ROS, no camera, no IO.

render() returns a string and parse() returns a tuple, so the whole interface
can be tested by asserting on text. Nothing here knows what a robot is.

    python3 screen.py      # self-test
"""
import dial

W = 71
NAMES = ['volume', 'treble', 'high mid', 'low mid', 'bass', 'level',
         'drive', 'gain']


def _rule(text=''):
    if not text:
        return '/' * W
    return '//  ' + text.ljust(W - 8) + '  //'


def render(knobs, status):
    """knobs: [{'name','now','target','last'}] with None where unknown.
       status: {'zero','tol','amp','camK','camT','arm'}"""
    L = ['/' * W,
         _rule(f"K N O B   B R A I N".ljust(30)
               + f"zero {status.get('zero', '--:--')}"
               + f"  tol {int(dial.TOL_DEG)}"
               + f"  amp {status.get('amp', 'WAITING')}"),
         '/' * W, '',
         '    #  knob        now  target     0 |--- one cell = one bite ---| 300',
         '   ' + '-' * (W - 5)]

    for i, k in enumerate(knobs, 1):
        now = k.get('now')
        tgt = k.get('target')
        last = k.get('last')
        L.append('   {:2d}  {:<10s} {:>4s}  {:>6s}     {}  {:>18s}'.format(
            i, k['name'],
            '  ??' if now is None else f'{now:.0f}',
            '.' if tgt is None else f'{tgt:.0f}',
            dial.bar(0 if now is None else now, tgt),
            '.' if last is None else f'{last:+.0f}'))

    L += ['   ' + '-' * (W - 5),
          '    cam K  {}      cam T  {}      arm  {}'.format(
              status.get('camK', '?'), status.get('camT', '?'),
              status.get('arm', 'PARKED')),
          '',
          '/' * W,
          _rule('enter  [knob] [degrees]        e.g.   2 75'),
          _rule('degrees as multiples of 25:  25  50  75  100  125  150'),
          _rule('state  re-read     zero  recalibrate     q  quit'),
          '/' * W, '']
    return '\n'.join(L)


def box(title, lines):
    out = ['/' * W, _rule(title), '/' * W]
    out += ['    ' + s for s in lines]
    return '\n'.join(out)


def parse(line, n=8):
    """-> ('turn', index, degrees, is_multiple_of_25) | ('state',) |
          ('zero',) | ('quit',) | ('error', why)"""
    parts = line.strip().lower().split()
    if not parts:
        return ('error', '')
    head = parts[0]
    if head in ('q', 'quit', 'exit'):
        return ('quit',)
    if head in ('state', 's'):
        return ('state',)
    if head in ('zero', 'z', 'c'):
        return ('zero',)
    if len(parts) != 2:
        return ('error', 'type a knob number and a target, like:  2 75')
    try:
        i, deg = int(parts[0]), float(parts[1])
    except ValueError:
        return ('error', f'did not understand "{line.strip()}"')
    if not 1 <= i <= n:
        return ('error', f'knob number must be 1 to {n}')
    if not dial.in_range(deg):
        return ('error', f'{deg:.0f} is outside the knob travel '
                         f'(0 to {dial.FULL_TRAVEL:.0f})')
    return ('turn', i, deg, abs(deg % dial.BITE_DEG) < 1e-6)


def nearest_multiples(deg):
    lo = math.floor(deg / dial.BITE_DEG) * dial.BITE_DEG
    return lo, lo + dial.BITE_DEG


import math  # noqa: E402  (used only by nearest_multiples, kept beside it)


def _selftest():
    knobs = [{'name': n, 'now': None, 'target': None, 'last': None}
             for n in NAMES]
    knobs[1].update(now=50.0, target=75.0, last=23.0)
    knobs[4].update(now=25.0, target=25.0, last=26.0)
    out = render(knobs, {'zero': '00:14', 'amp': 'LOCKED', 'camK': '8/8 knobs',
                         'camT': 'tags 2,4', 'arm': 'PARKED'})
    assert 'K N O B   B R A I N' in out
    assert '[##-.........]' in out, 'treble owes one bite'
    assert '   +23' in out
    assert 'high mid' in out and 'gain' in out
    assert out.count('\n') > 20
    # every line fits the terminal the demo will run in
    assert max(len(s) for s in out.splitlines()) <= 80, 'a line wraps'

    assert parse('2 75') == ('turn', 2, 75.0, True)
    assert parse('2 60') == ('turn', 2, 60.0, False), 'accepted, but flagged'
    assert parse('q')[0] == 'quit'
    assert parse('state')[0] == 'state'
    assert parse('c')[0] == 'zero'
    assert parse('9 25')[0] == 'error', 'no ninth knob'
    assert parse('2 400')[0] == 'error', 'past the hard stop'
    assert parse('banana')[0] == 'error'
    assert parse('2')[0] == 'error'
    assert nearest_multiples(60.0) == (50.0, 75.0)

    # an unread knob must not render as if it were at zero
    assert '  ??' in render([{'name': 'volume', 'now': None, 'target': None,
                              'last': None}], {})

    print('screen: 15 assertions pass')


if __name__ == '__main__':
    _selftest()
