import xarm, time, json, sys
a = xarm.Controller("USB")
IDS = [6,5,4,7,3,2]          # joint01..joint56
def rd(s):
    p=a.getPosition(s); return p-65536 if p>32767 else p

traj = json.load(open('/home/pi/traj.json'))
for c in traj:
    assert all(0 <= v <= 1000 for v in c), "command out of range"

cur = [rd(i) for i in IDS]
print("from", cur, "-> start", traj[0])

# ease into the trajectory start
for k in range(1, 21):
    f = k/20
    step = [int(round(cur[j] + (traj[0][j]-cur[j])*f)) for j in range(6)]
    a.setPosition(list(zip(IDS, step)), duration=250, wait=True); time.sleep(0.03)

print("running trajectory,", len(traj), "waypoints")
for c in traj:
    a.setPosition(list(zip(IDS, c)), duration=200, wait=False); time.sleep(0.12)

time.sleep(0.6)
# back to the vertical pose
vert = [500, 507, 342, 505, 277, 500]
now = [rd(i) for i in IDS]
for k in range(1, 26):
    f = k/25
    step = [int(round(now[j] + (vert[j]-now[j])*f)) for j in range(6)]
    a.setPosition(list(zip(IDS, step)), duration=250, wait=True); time.sleep(0.03)
time.sleep(0.6)
print("final", [rd(i) for i in IDS])
