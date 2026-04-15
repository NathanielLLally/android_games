#!/bin/sh
#
#  resolution 2560x1600
#
declare -i limit=0 n=0
if [ -n "$1" ]; then limit=$1; fi
while [ \( $(($limit-$n)) -gt 0 \) -o \( $limit -eq 0 \) ]; do
  adb shell input swipe 1263 46 1263 46 84
  sleep 2
  adb shell input swipe 1959 1465 1959 1465 99
  sleep 1
  adb shell input swipe 1563 780 1563 780 106
  sleep 37.6
  adb shell input swipe 2116 1313 2116 1313 116
  sleep 1

  n+=1
  spl=$(bc <<< "scale=3;$SECONDS/$n")
  remain=$(bc <<< "scale=0; ($limit-$n)*$spl")
  now=$(date "+%r %a")
  eta=$(date "+%r %a" --date="$remain seconds")
  output="$now $spl s per loop, $n loops"
  if [ limit -gt 0 ]; then
    output+=" of $(($limit - $n)), eta $eta ($remain)s"
  fi
  echo $output
done
