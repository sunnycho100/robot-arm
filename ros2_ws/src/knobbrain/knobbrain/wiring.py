#!/usr/bin/env python3
"""Check the ROS wiring without ROS, by reading the source rather than running it.

brain.py imports rclpy and his xarmrob package, so it cannot even be imported
on a laptop. That is not a reason to ship it unchecked. The claims below are
the ones that would be expensive to discover on the bench with an audience
watching, and every one of them is visible in the syntax tree:

  - his sequencer is cancelled, so tag lock does not start an eight-knob run
  - we publish his three topics and invent no others
  - our bite is his macro in his order, not a reordering that happens to work
  - the launch file starts three nodes and not the fourth
  - our copy of the detector has not drifted from the one under test

    python3 wiring.py
"""
import ast
import hashlib
import pathlib
import py_compile
import subprocess
import sys

HERE = pathlib.Path(__file__).parent
PKG = HERE.parent
REPO_DETECTOR = PKG.parents[2] / 'cv' / 'ampknobs.py'

def tree(name):
    src = (HERE / name).read_text()
    py_compile.compile(str(HERE / name), doraise=True)
    return ast.parse(src), src


def calls_in(fn):
    """Every self._x(...) inside a function, in source order."""
    out = []
    for n in ast.walk(fn):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id == 'self'):
            out.append(n.func.attr)
    return out


def main():
    t, src = tree('brain.py')
    fns = {n.name: n for n in ast.walk(t) if isinstance(n, ast.FunctionDef)}
    cls = next(n for n in ast.walk(t) if isinstance(n, ast.ClassDef))

    assert 'PresetController' in [b.id for b in cls.bases], \
        'Brain must subclass his controller, not copy his math'

    body = ast.get_source_segment(src, fns['__init__'])
    assert 'self.timer.cancel()' in body, \
        'his 1 s timer is still running: it would drive all eight knobs the ' \
        'moment the tags lock'
    assert 'self.action_queue = []' in body, 'his queue is still populated'
    assert body.index('super().__init__()') < body.index('self.timer.cancel()'), \
        'the cancel must follow his constructor, which is what creates it'

    # exactly his three topics, and nothing new subscribed
    assert src.count('create_publisher') == 0, \
        'publishers are inherited from him; a new one means a second interface'
    assert src.count('create_subscription') == 0, \
        '/aruco is inherited; a new subscription is a new dependency'
    for pub in ('pub_endpoint', 'pub_gripper', 'pub_wrist'):
        assert pub in src, f'{pub} unused: we are not driving his stack'

    # the bite REPLAYS his block rather than re-implementing it
    bite = ast.get_source_segment(src, fns['bite'])
    assert 'self.blocks[name]' in bite, 'the bite must come from his macro'
    assert 'macro.twist_at' in bite and 'macro.home_of' in bite, \
        'the twist and the approach angle must be found in his block, not typed'
    assert 'wrist_target' in bite, 'the twist size must come from dial.py'
    # every step type he can emit is handled, so a new one is an error and not
    # a silently skipped move
    do = ast.get_source_segment(src, fns['_do'])
    for step in ('hover', 'plunge', 'gripper', 'twist'):
        assert f"'{step}'" in do, f'his {step} step is not handled'
    assert 'raise ValueError' in do, 'an unknown step must be loud'

    # NOTHING of his is copied. He changed six offsets and added a per-knob
    # wrist flip between his snapshot and his Pi; a number frozen here would
    # have swung the wrist the wrong way on four of the eight knobs.
    #
    # Checked by meaning rather than by digits. A bare substring search for
    # "0.6" also matches our own knob-matching threshold, which has nothing to
    # do with his plunge depth, and a test that cries wolf gets deleted.
    HIS = {1.57, -1.57, 2.0, -1.14, 0.6}
    for n in ast.walk(t):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant):
            assert n.value.value not in HIS, \
                f'module constant {ast.unparse(n.targets[0])} holds one of his ' \
                f'numbers ({n.value.value}); read it off his macro instead'
    # And none of his numbers anywhere in the motion path, in any form. Checking
    # only for a bare argument misses `1.57 if value is None else value`, which
    # is an IfExp, so walk the whole subtree of the two functions that move the
    # arm and look at every constant in them.
    for fname in ('bite', '_do'):
        for n in ast.walk(fns[fname]):
            if isinstance(n, ast.Constant) and isinstance(n.value, float):
                assert n.value not in HIS, \
                    f'{fname}() contains {n.value}, which is his. It has to ' \
                    f'come from his macro, or it goes stale when he retunes.'
    for name in ('dist lev', 'initial'):
        assert name not in src, f'{name!r} is his, and comes from his macro'
    assert 'macro.blocks' in src, 'his macro is never read'

    # the launch file drops his sequencer and keeps the rest
    lt = (PKG / 'launch' / 'backend.launch.py').read_text()
    execs = [n.value for n in ast.walk(ast.parse(lt))
             if isinstance(n, ast.keyword) and n.arg == 'executable']
    names = [e.value for e in execs]
    assert names == ['command_xarm', 'xarm_kinematics', 'aruco'], names
    assert 'preset_controller' not in [n.strip() for n in names], \
        'his sequencer is in our launch file'

    # entry points the demo command depends on
    st = (PKG / 'setup.py').read_text()
    assert 'brain=knobbrain.brain:main' in st
    assert 'go=knobbrain.go:main' in st
    assert "'launch/backend.launch.py'" in st, \
        'the launch file is not installed, so ros2 launch will not find it'

    # The detector copy has not drifted from the one the checks exercise. Only
    # checkable beside the repo: on a Pi this package is deployed alone, and
    # there is nothing to compare against.
    if REPO_DETECTOR.exists():
        a = hashlib.sha256((HERE / 'ampknobs.py').read_bytes()).hexdigest()
        b = hashlib.sha256(REPO_DETECTOR.read_bytes()).hexdigest()
        assert a == b, ('the packaged detector differs from pi/cv/ampknobs.py. '
                        'Copy it across before deploying, or the bench and the '
                        'demo are running different code.')
    else:
        print('  (no repo alongside: detector drift check skipped)')

    # Everything that does not need ROS must import BOTH ways: as loose files,
    # which is how these self-checks run, and as an installed package, which is
    # how ros2 run loads them. A bare `import dial` satisfies the first and
    # fails the second, and the failure only shows up on the Pi.
    for m in ('dial', 'screen', 'cams', 'eyes', 'macro'):
        py_compile.compile(str(HERE / f'{m}.py'), doraise=True)
        __import__(m)                       # loose, from this directory
    # In a FRESH interpreter, and from a directory where the loose copies are
    # not importable. In this process they are already in sys.modules, so a
    # bare `import dial` inside a package module would find the cached one and
    # the check would pass on a package that cannot actually be installed.
    probe = ('import knobbrain.dial, knobbrain.screen, knobbrain.cams, '
             'knobbrain.eyes, knobbrain.macro')
    r = subprocess.run([sys.executable, '-c', probe], cwd=str(PKG),
                       capture_output=True, text=True)
    assert r.returncode == 0, (
        'a module imports its siblings in a way that only works when they are '
        'loose files in the cwd. ros2 run loads them as an installed package '
        'and will fail:\n' + r.stderr.strip().splitlines()[-1])
    for m in ('brain', 'go'):
        py_compile.compile(str(HERE / f'{m}.py'), doraise=True)

    print("wiring: 26 assertions pass")
    return 0


if __name__ == '__main__':
    sys.path.insert(0, str(HERE))
    sys.exit(main())
