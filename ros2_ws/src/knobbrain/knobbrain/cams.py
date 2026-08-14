#!/usr/bin/env python3
"""Pick the knob camera without guessing at device numbers.

His aruco node runs `cv2.VideoCapture(0)` at import, so it owns /dev/video0 the
moment it loads. Ours has to be the other one, and it has to be the other one
across reboots, when video numbering is not stable. So choose by the USB serial
in /dev/v4l/by-id and exclude whatever video0 currently points at.

    python3 cams.py        # self-test, then list what is plugged in now
"""
import glob
import os

BY_ID = '/dev/v4l/by-id'


def choose(entries, taken):
    """entries: [(by-id link, real device)] -> (link, real) | (None, why)

    Only -video-index0 nodes are real streams; index1 is a metadata node that
    opens fine and never returns a frame, which is a confusing way to fail.
    """
    cams = [(l, r) for l, r in entries if l.endswith('-video-index0')]
    free = [(l, r) for l, r in cams if r != taken]
    if not cams:
        return None, 'no camera found at all'
    if not free:
        return None, (f'only one camera, and {taken} is held by the aruco '
                      f'node. The knob camera needs its own.')
    if len(free) > 1:
        return None, ('more than one spare camera:\n  '
                      + '\n  '.join(f'{r}  {os.path.basename(l)}'
                                    for l, r in free)
                      + '\nset "camera" in ~/.knobbrain.json to one of these')
    return free[0]


def serial_of(link):
    """usb-046d_0825_4DC82940-video-index0 -> 4DC82940. Two identical C270s
    still differ here, so a camera is always identifiable even when the model
    is not."""
    base = os.path.basename(link).rsplit('-video-index', 1)[0]
    return base.rsplit('_', 1)[-1]


def scan():
    return sorted((p, os.path.realpath(p)) for p in glob.glob(BY_ID + '/*'))


def knob_camera(taken='/dev/video0'):
    return choose(scan(), os.path.realpath(taken))


def swapped(entries, taken, expected_serial):
    """Have the two cameras traded places since last time?

    We cannot choose which camera becomes /dev/video0: that is plug and probe
    order, and his aruco node hardcodes index 0 rather than a stable path. So
    the honest thing is not to fix it silently but to notice it, because the
    symptom otherwise is his node staring at a close-up of a knob and never
    locking, which looks like a tag problem and is not.
    """
    real = os.path.realpath(taken)
    for link, r in entries:
        if r == real and link.endswith('-video-index0'):
            got = serial_of(link)
            if expected_serial and got != expected_serial:
                return (f'{taken} is camera {got}, but the tag camera was '
                        f'{expected_serial} last time. The two have swapped: '
                        f'unplug both and plug the TAG camera in first.')
            return None
    return None


def _selftest():
    A = ('/dev/v4l/by-id/usb-046d_0825_AAA-video-index0', '/dev/video0')
    Am = ('/dev/v4l/by-id/usb-046d_0825_AAA-video-index1', '/dev/video1')
    B = ('/dev/v4l/by-id/usb-046d_0825_BBB-video-index0', '/dev/video2')
    Bm = ('/dev/v4l/by-id/usb-046d_0825_BBB-video-index1', '/dev/video3')
    C = ('/dev/v4l/by-id/usb-046d_0825_CCC-video-index0', '/dev/video4')

    assert choose([A, Am, B, Bm], '/dev/video0') == B, 'takes the free one'
    link, why = choose([A, Am], '/dev/video0')
    assert link is None and 'only one camera' in why
    link, why = choose([], '/dev/video0')
    assert link is None and 'no camera' in why
    link, why = choose([A, Am, B, Bm, C], '/dev/video0')
    assert link is None and 'more than one' in why, 'ambiguity is not guessed'
    # the metadata node must never be chosen even when it is the only spare
    link, why = choose([A, Am, Bm], '/dev/video0')
    assert link is None, 'index1 is not a camera'

    assert serial_of(A[0]) == 'AAA'
    assert swapped([A, Am, B, Bm], '/dev/video0', 'AAA') is None
    msg = swapped([A, Am, B, Bm], '/dev/video0', 'BBB')
    assert msg and 'swapped' in msg, 'a trade must be named, not absorbed'
    assert swapped([A, Am, B, Bm], '/dev/video0', None) is None, 'first run'

    print('cams: 10 assertions pass')
    found = scan()
    print(f'  plugged in here now: {len(found)} by-id entries')
    for l, r in found:
        print(f'    {r}  {os.path.basename(l)}')


if __name__ == '__main__':
    _selftest()
