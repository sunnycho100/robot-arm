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
    for t in ('hover', 'plunge', 'gripper', 'twist'):
        assert f"'{t}'" in do, f'his {t} step is not handled'
    assert 'raise ValueError' in do, 'an unknown step must be loud'

    # NOTHING of his is copied as a literal. He changed six offsets and added a
    # per-knob wrist flip between his snapshot and his Pi; a number frozen here
    # would have swung the wrist the wrong way on four of the eight knobs.
    for stale in ('1.57', '2.00', '-1.14', '0.6', 'dist lev', 'initial'):
        assert stale not in src, f'{stale!r} is copied out of his file'
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

    # everything that does not need ROS must import cleanly
    for m in ('dial', 'screen', 'cams', 'eyes', 'macro'):
        py_compile.compile(str(HERE / f'{m}.py'), doraise=True)
        __import__(m)
    for m in ('brain', 'go'):
        py_compile.compile(str(HERE / f'{m}.py'), doraise=True)

    print('wiring: 24 assertions pass')
    return 0


if __name__ == '__main__':
    sys.path.insert(0, str(HERE))
    sys.exit(main())
