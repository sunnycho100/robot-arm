#!/usr/bin/env python3
"""Find the base rotation that actually centres the gripper on the knob.

Grip force is the sensor. Try a few base offsets, squeeze at each, and see which
one produces real force rather than the fingers meeting each other. This is the
self-debugging idea applied to positioning instead of to the turn.
"""
import sys
sys.path.insert(0, '/home/pi')
import arm

g = arm.counts_of(arm.poses()['grip0'])
rows, best = [], (None, -1)

for d in (-24, -12, 0, 12, 24):
    t = list(g)
    t[2] = g[2] + d                     # servo_02 is the elbow: height and reach
    arm.approach(t, speed=140, via=0.85)
    _, lag, holding = arm.squeeze(70)
    rows.append((d, lag, holding))
    if lag > best[1]:
        best = (d, lag)
    arm.release(380)

print('\n===== RESULT =====')
print('elbow offset | force | verdict')
for d, lag, holding in rows:
    verdict = 'HOLDS' if holding else 'air'
    print(f'   {d:+4d}     |  {lag:3d}  | {verdict}')
print(f'\nbest: offset {best[0]:+d} with {best[1]} counts')
arm.move(arm.NEUTRAL, speed=150)
