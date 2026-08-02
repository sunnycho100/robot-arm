#!/bin/bash
# Keep trying every known network, in the order Sunghwan wants, until one sticks.
# NetworkManager gives up too easily when the Pi boots before a network exists,
# which is the most likely reason the connection failed in class on 2026-07-29.
nmcli -t -f STATE general | grep -q "^connected" && exit 0
nmcli device wifi rescan 2>/dev/null
sleep 3
for c in eduroam UWNet hotspot 201; do
    nmcli -t -f NAME connection show | grep -qx "$c" || continue
    if nmcli connection up "$c" 2>/dev/null; then
        logger -t netretry "joined $c"
        exit 0
    fi
done
logger -t netretry "no network available"
