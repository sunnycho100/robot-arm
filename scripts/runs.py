#!/usr/bin/env python3
"""Compare recorded turn_knob runs, so the setup gets improved on evidence.

  runs.py            table of every run, newest last
  runs.py <stamp>    the bite-by-bite detail of one run

Tracking is the number to watch: what fraction of the commanded roll the knob
actually followed. 100% means the grip is driving the knob. A low number means it
is slipping in the jaws, which the grip force reading cannot see, so this is the
only place it shows up. If tracking is falling run over run, the pose or the
gripper needs attention, not more force.
"""
import json, os, sys, glob

RUNS = os.path.expanduser('~/runs')


def load(stamp):
    return json.load(open(os.path.join(RUNS, stamp, 'log.json')))


def detail(stamp):
    log = load(stamp)
    print(f'{stamp}   target {log["target"]:+.0f}   '
          f'{"open loop" if log["open_loop"] else "closed loop"}   '
          f'squeeze {log["squeeze"]}')
    print(f'  knob {log.get("knob_turned", "?")}')
    for b in log['bites']:
        if b.get('achieved') is None:
            print(f'  bite {b["n"]}: commanded {b["commanded"]:+6.0f}   '
                  f'{b.get("note", "failed")}')
        else:
            print(f'  bite {b["n"]}: commanded {b["commanded"]:+6.0f}   '
                  f'followed {b["achieved"]:+6.0f}   '
                  f'tracking {b["tracking"]*100:4.0f}%   '
                  f'others {b.get("others", {})}')
    if 'final' in log:
        print(f'  final {log["final"]:+.0f}, error {log["error"]:+.0f}, '
              f'{"OK" if log["ok"] else "MISSED"}')


def table():
    rows = [json.loads(l) for l in
            open(os.path.join(RUNS, 'index.jsonl'))] if \
        os.path.exists(os.path.join(RUNS, 'index.jsonl')) else []
    if not rows:
        print(f'no runs recorded yet in {RUNS}')
        return
    print(f'{"when":16s} {"mode":6s} {"knob":7s} {"target":>7s} {"got":>7s} '
          f'{"error":>7s} {"track":>6s}  bites')
    for r in rows:
        try:
            log = load(r['stamp'])
            got = [b['tracking'] for b in log['bites'] if b.get('tracking')]
            track = f'{sum(got)/len(got)*100:5.0f}%' if got else '    -'
            n = len(log['bites'])
        except Exception:
            track, n = '    -', 0
        f = r.get('final')
        e = r.get('error')
        print(f'{r["stamp"]:16s} '
              f'{"open" if r.get("open_loop") else "closed":6s} '
              f'{str(r.get("knob_turned") or "-"):7s} '
              f'{r.get("target", 0):+7.0f} '
              f'{(f"{f:+.0f}" if f is not None else "-"):>7s} '
              f'{(f"{e:+.0f}" if e is not None else "-"):>7s} '
              f'{track:>6s}  {n}'
              f'{"" if r.get("ok") else "   <- missed"}')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        detail(sys.argv[1])
    else:
        table()
