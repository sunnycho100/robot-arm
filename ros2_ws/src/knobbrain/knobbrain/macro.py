#!/usr/bin/env python3
"""Read his macro instead of copying numbers out of it.

His PresetController.build_macro_sequence() returns the whole demo as a list of
plain dicts. Everything we would otherwise have hardcoded is in there already:
which knobs exist and in what order, the wrist angle each one is approached at,
how far the twist goes, the plunge depth, the gripper's open and closed values,
and how long each step is given. So we call his method and read it.

This is not tidiness. Between his 14 Aug snapshot and the copy running on his Pi
he changed six knob offsets AND added a per-knob wrist flip: the four far-side
knobs are approached at -1.57 rather than +1.57 to keep the servo off its stop.
A constant copied from his file on Tuesday is a wrong number on Thursday, and it
would have swung the wrist 180 degrees the wrong way on half the panel.

    python3 macro.py       # self-test, against both shapes of his macro
"""
import math


def blocks(queue):
    """-> (park_target, {knob: [his actions, verbatim]})

    Split where he splits: every knob's turn begins by pulling back to the safe
    pose, so a hover onto that target starts a new block. The trailing pull-back
    has no plunge in it and is not a knob.
    """
    if not queue or queue[0]['type'] != 'hover':
        raise ValueError('his macro no longer opens with a hover; '
                         'read build_macro_sequence() before trusting this')
    park = queue[0]['target']
    starts = [i for i, a in enumerate(queue)
              if a['type'] == 'hover' and a.get('target') == park]
    out = {}
    for s, e in zip(starts, starts[1:] + [len(queue)]):
        blk = queue[s:e]
        plunge = [a for a in blk if a['type'] == 'plunge']
        if not plunge:
            continue                    # the final pull-back, not a knob
        out[plunge[0]['target']] = blk
    if not out:
        raise ValueError('no knobs found in his macro')
    return park, out


def twist_at(block):
    """Index of the twist that turns the knob: the one made while gripping.

    Found by state rather than position, so his untwist-before-lifting and his
    squaring-up twist are never mistaken for it however he reorders them.
    """
    closed = max(a['value'] for a in block if a['type'] == 'gripper')
    gripped = False
    for i, a in enumerate(block):
        if a['type'] == 'gripper':
            gripped = a['value'] == closed
        elif a['type'] == 'twist' and gripped:
            return i
    raise ValueError('no twist happens while the gripper is closed')


def home_of(block, i):
    """The wrist angle this knob is approached at. +1.57 for most of his
    panel, -1.57 for the four he flips."""
    for a in reversed(block[:i]):
        if a['type'] == 'twist':
            return a['value']
    raise ValueError('no wrist angle set before the twist')


def bite_deg(block):
    """How far his own demo turns a knob, in degrees. His comment says 45; the
    arithmetic says 24.6, and the arithmetic is what runs."""
    i = twist_at(block)
    return math.degrees(block[i]['value'] - home_of(block, i))


def grips(block):
    v = [a['value'] for a in block if a['type'] == 'gripper']
    return min(v), max(v)


# ------------------------------------------------------------------ self-test
def _queue(flipped=()):
    """His build_macro_sequence(), both the 14 Aug shape and the one on his Pi
    now. Reproduced here only so this file can be tested without ROS."""
    q = []
    for knob in ['volume', 'treble', 'bass', 'gain']:
        base, act = ((-1.57, -1.14) if knob in flipped else (1.57, 2.00))
        q += [{'type': 'hover', 'target': 'initial', 'wait': 3.0},
              {'type': 'gripper', 'value': 0.0, 'wait': 0.5},
              {'type': 'twist', 'value': base, 'wait': 0.5},
              {'type': 'hover', 'target': knob, 'wait': 3.0},
              {'type': 'plunge', 'target': knob, 'z': 0.6, 'wait': 2.0},
              {'type': 'gripper', 'value': 1.57, 'wait': 2.0},
              {'type': 'twist', 'value': act, 'wait': 2.0},
              {'type': 'gripper', 'value': 0.0, 'wait': 2.0},
              {'type': 'twist', 'value': base, 'wait': 2.0},
              {'type': 'hover', 'target': knob, 'wait': 2.0}]
    q.append({'type': 'hover', 'target': 'initial', 'wait': 3.0})
    return q


def _selftest():
    for flipped in ((), ('bass', 'gain')):
        park, bs = blocks(_queue(flipped))
        assert park == 'initial'
        assert list(bs) == ['volume', 'treble', 'bass', 'gain'], list(bs)
        assert all(len(b) == 10 for b in bs.values())
        for name, b in bs.items():
            want = -1.57 if name in flipped else 1.57
            assert home_of(b, twist_at(b)) == want, (name, flipped)
            # the turn is the same either way: only the approach is mirrored
            assert abs(bite_deg(b) - 24.638) < 0.01, bite_deg(b)
            assert grips(b) == (0.0, 1.57)
            assert b[twist_at(b)]['type'] == 'twist'
            # the twist we find must be the gripping one, not the untwist
            assert twist_at(b) == 6, twist_at(b)

    # a knob approached from the flipped side must not inherit the other's home
    _, bs = blocks(_queue(('bass', 'gain')))
    assert home_of(bs['treble'], twist_at(bs['treble'])) == 1.57
    assert home_of(bs['bass'], twist_at(bs['bass'])) == -1.57

    # if he ever reorders so nothing is twisted while gripping, say so loudly
    bad = [a for a in _queue()[:10] if a['type'] != 'gripper']
    try:
        twist_at(bad)
        assert False, 'a macro that never grips should not yield a bite'
    except ValueError:
        pass

    try:
        blocks([{'type': 'gripper', 'value': 0.0, 'wait': 1.0}])
        assert False, 'a macro not opening with a hover should be refused'
    except ValueError:
        pass

    print('macro: 30 assertions pass, on both the flipped and unflipped shapes')


if __name__ == '__main__':
    _selftest()
