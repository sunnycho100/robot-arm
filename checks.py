#!/usr/bin/env python3
"""Run every self-check in dependency order and say which layer broke.

    python3 checks.py              # all of them
    python3 checks.py -v           # show each check's own output
    python3 checks.py aruco knobs  # only stages whose name contains these

The order matters: it is the order the real pipeline depends on. Kinematics
before scene, scene before servoing, camera compat before anything that reads
a frame. When something fails, the FIRST red line is the one to fix; later
failures are usually its consequences, so fix upward from the bottom of the
stack rather than starting wherever the symptom appeared.

Each run appends its results to checks.log (gitignored) so a number that
quietly drifts is visible as a change rather than a surprise.
"""
import json
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).parent
PY = HERE / 'sim' / '.venv' / 'bin' / 'python'

# (stage, directory, module, what passing actually proves)
STAGES = [
    ('ik',      'scripts', 'ik.py',
     'course invkin round-trips against arm.py FK, mm nudges land, '
     'unreachable raises'),
    ('strategy', 'scripts', 'strategy.py',
     'bites converge through a slipping grip and an off-axis camera, '
     'never overshooting'),
    ('regrip',  'scripts', 'regrip.py',
     'the search recovers a mis-taught pose, and stops rather than wandering'),
    ('knob',    'scripts', 'knob.py selftest',
     'the BENCH knob finder and pointer reader, on a synthetic pedal'),
    ('urdfmap', 'sim', 'urdfmap.py',
     'course joint angles land on the same pose in the URDF'),
    ('scene',   'sim', 'scene.py',
     'knob geometry sane, collision detection actually detects, marker on tip'),
    ('regripsim', 'sim', 'regrip_sim.py',
     'that same search against real IK, collisions and arm scatter'),
    ('compat',  'cv',  'compat.py',
     'aruco works on this OpenCV, intrinsics load and scale'),
    ('aruco',   'cv',  'aruco_locate.py',
     'tags recover corners under warp and give millimetres'),
    ('knobs',   'cv',  'knobs2.py',
     'classical finder still reaches 6 of 9 bench frames, no false positives'),
    ('gripper', 'cv',  'gripper.py',
     'the gripper is findable in a real frame, which servoing needs'),
    ('pointer', 'cv',  'pointer.py',
     'pointer angle within 3 degrees, signed rotation correct across the wrap'),
    ('simcam',  'sim', 'simcam.py',
     'sim camera matches the bench optics, projection agrees with them'),
    ('servo',   'sim', 'servo_center.py',
     'centring converges from 10 mm out, and survives a wrong Jacobian'),
    ('cycle',   'sim', 'cycle.py',
     'find, centre, pre-close, descend, turn, verify runs on all three knobs'),
    ('sequence', 'sim', 'sequence.py',
     'several knobs in a row, biting until each arrives despite a slipping grip'),
]

# Stages needing a model download or torch; skipped unless asked for by name
SLOW = {
    ('knobs_ml', 'cv', 'knobs_ml.py',
     'ML tier fires and the two tiers together cover 8 of 9 frames'),
    ('edge', 'sim', 'edge_cases.py',
     'the pedal moved, rotated, raised and reshaped still gets turned'),
}


def run(stage, folder, module, claim, verbose=False):
    t0 = time.time()
    p = subprocess.run([str(PY)] + module.split(), cwd=HERE / folder,
                       capture_output=True, text=True)
    dt = time.time() - t0
    ok = p.returncode == 0
    mark = 'PASS' if ok else 'FAIL'
    print(f'  [{mark}] {stage:9s} {dt:5.1f}s  {claim}')
    if verbose or not ok:
        body = (p.stdout + p.stderr).strip().splitlines()
        for line in (body if verbose else body[-12:]):
            print(f'         | {line}')
    return ok, dt


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    verbose = '-v' in sys.argv
    if not PY.exists():
        print(f'no venv at {PY}\n'
              f'create it:  python3 -m venv sim/.venv && '
              f'sim/.venv/bin/pip install mujoco ikpy numpy scipy matplotlib '
              f'pillow opencv-contrib-python')
        return 1
    stages = list(STAGES) + sorted(SLOW)
    if args:
        stages = [s for s in stages if any(a in s[0] for a in args)]
    else:
        stages = list(STAGES)          # slow ones only when named
    print(f'running {len(stages)} checks, in dependency order\n')
    results, failed = {}, []
    for stage, folder, module, claim in stages:
        ok, dt = run(stage, folder, module, claim, verbose)
        results[stage] = dict(ok=ok, seconds=round(dt, 1))
        if not ok:
            failed.append(stage)
    with open(HERE / 'checks.log', 'a') as f:
        f.write(json.dumps({'when': time.strftime('%Y-%m-%d %H:%M:%S'),
                            'results': results}) + '\n')
    print()
    if failed:
        print(f'{len(failed)} failed: {", ".join(failed)}')
        print(f'fix "{failed[0]}" first; the later ones usually depend on it.')
        return 1
    print(f'all {len(stages)} passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
