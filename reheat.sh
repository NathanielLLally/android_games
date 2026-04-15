#!/bin/sh
#
#  resolution 2560x1600
#
declare -i limit=0 n=0 once=1 rfreq=3
if [ -n "$1" ]; then limit=$1; fi
while [ \( $(($limit-$n)) -gt 0 \) -o \( $limit -eq 0 \) ]; do
  adb shell input swipe 1286 124 1286 124 101
  sleep 2
  adb shell input swipe 1723 770 1723 770 114
  sleep 33
  n+=1
  spl=$(bc <<< "scale=3;$SECONDS/$n")
  remain=$(bc <<< "scale=0; (($limit-$n)*$spl)/1")
  eta=$(date "+%r %a" --date="$remain seconds")
  output="$spl s per loop, $n loops"
  if [ $once -gt 0 ]; then
    now=$(date "+%r %a")
    echo "begin $now"
    once=0
  fi
  if [ $(($n%$rfreq)) -eq 0 ]; then
    output+=""
  fi
  if [ $limit -gt 0 ]; then
    rmin=$(bc <<< "scale=0;$remain/60") 
    rh=$(bc <<< "scale=0;$rmin/60")
    output+=" of $(($limit - $n)), eta $eta ($rh:$(($rmin%60)):$(($remain%60)))"
  fi
  echo $output
done
