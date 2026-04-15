#!/bin/sh
#
#  resolution 2560x1600
#
declare -i limit=3156 n=0 display_freq=1
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
  if [ $(($SECONDS%$display_freq)) -eq 0 ]; then
  spl=$(bc <<< "scale=3; $SECONDS/$n")
  remain=$(bc <<< "scale=3; ($limit-$n)*$spl")
  now=$(date "+%r %a")
  eta=$(date "+%r %a" --date="$remain seconds")
  output="$now $spl s per loop, $n loops"
  if [ $limit -gt 0 ]; then
    rh=$(bc <<< "scale=0;$remain/3600")
    rmin=$(bc <<< "scale=0;$remain%3600")
    rs=$(bc <<< "scale=0;$remain%60")
    output="$output of $(($limit - $n)) $now, eta $eta ($remain)s"
   # $rh:$rmin:$rs"
  fi
  echo $output
  fi
done
