#!/bin/sh

#https://stackoverflow.com/questions/192292/how-best-to-include-other-scripts
DIR=$(dirname $0)

if [ ! -d "$DIR" ]; then DIR="$PWD"; fi
. "$DIR/utils.sh"


devices=$(getDevices)
if [ -n "$1" ]; then
  devices=$1
fi
for device in $devices
  do
    status=$(resetWifi $device)
    echo "$device $status"
    status=$(adbtcpip $device)
    echo $status
    status=$(forwardPort $device)
    echo "port forward result: $status"

    ip=$(getWlanIP $device)
    echo "$device $ip"
#    setPIN $device 3 5 7 6 4 0
#    pin=$(getPIN $device)
#    echo "pin: $pin"
#    for key in $pin; do
#      echo "adb -s $device shell input keyevent $key"
#    done
  done
