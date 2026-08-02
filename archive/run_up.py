import xarm, time, json
a = xarm.Controller("USB")
IDS = [6,5,4,7,3,2]
def rd(s):
    p=a.getPosition(s); return p-65536 if p>32767 else p

traj = json.load(open('/home/pi/up.json'))
for c in traj:
    assert all(0 <= v <= 1000 for v in c), "command out of range"

cur = [rd(i) for i in IDS]
print("current  ", cur)
print("reorient ", traj[0])
# slow reorientation into the IK pose
for k in range(1, 31):
    f = k/30
    step = [int(round(cur[j] + (traj[0][j]-cur[j])*f)) for j in range(6)]
    a.setPosition(list(zip(IDS, step)), duration=300, wait=True); time.sleep(0.04)
time.sleep(0.5)
print("reoriented", [rd(i) for i in IDS])

# rise
print("rising 76mm -> 150mm")
for c in traj:
    a.setPosition(list(zip(IDS, c)), duration=250, wait=False); time.sleep(0.15)
time.sleep(0.8)
print("final    ", [rd(i) for i in IDS])
