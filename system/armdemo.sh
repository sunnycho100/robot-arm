#!/bin/bash
# Run the arm demo on boot, but ONLY when armed by a flag file.
# This is the no-network no-screen fallback: arm it while SSH still works, then
# trigger the demo in class by power-cycling the Pi. Nothing else is needed.
#
#   touch ~/DEMO_ALWAYS    every boot runs the demo. Use this for class, so you
#                          get unlimited runs from power-cycling alone.
#   touch ~/DEMO_ON_BOOT   one single run, then disarms itself.
#
# Remove the flag when you are done, or the arm moves on every future boot.
if [ -f /home/pi/DEMO_ALWAYS ]; then
  :
elif [ -f /home/pi/DEMO_ON_BOOT ]; then
  rm -f /home/pi/DEMO_ON_BOOT     # removed FIRST so a crash cannot loop
else
  exit 0
fi
sleep 45                          # time to stand clear and be ready
cd /home/pi || exit 1
{
  echo "=== boot demo $(date) ==="
  python3 arm.py neutral      || exit 1
  python3 arm.py goto grip0   || exit 1
  python3 arm.py turn -60     || exit 1
  python3 arm.py turn 60      || exit 1
  python3 arm.py neutral
  echo "=== done ==="
} >> /home/pi/demo.log 2>&1
