#!/bin/sh

#https://stackoverflow.com/questions/192292/how-best-to-include-other-scripts
DIR="${BASH_SOURCE%/*}"
if [[ ! -d "$DIR" ]]; then DIR="$PWD"; fi
. "$DIR/utils.sh"


devices=$(getDevices)
if [ -n "$1" ]; then
  devices=$1
fi
for device in $devices
  do
    ip=$(getIP $device)
    echo "$device $ip"
    setPIN $device 3 5 7 6 4 0
    pin=$(getPIN $device)
    echo "pin: $pin"
    for key in $pin; do
      echo "adb -s $device shell input keyevent $key"
    done
  done
