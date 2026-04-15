#!/bin/sh
#
#  resolution 2560x1600
#
declare -i limit=0 n=0 once=1 rfreq=3
if [ -n "$1" ]; then limit=$1; fi
while [ \( $(($limit-$n)) -gt 0 \) -o \( $limit -eq 0 \) ]; do
  adb shell input swipe 1280 23 1280 23 83
  sleep 1.9
  adb shell input swipe 1253 63 1253 63 68
  sleep 2.8
  adb shell input swipe 311 481 311 481 67
  sleep 2.5
  adb shell input swipe 1527 422 1534 859 771
  sleep 0.9
  adb shell input swipe 1260 1148 1260 1148 107
  sleep 0.8
  adb shell input swipe 1754 1274 1754 1096 301
  sleep 1.6
  adb shell input swipe 2027 1481 2027 1481 77
  sleep 3.3
  adb shell input swipe 1711 805 1711 805 109
  sleep 37.1
  adb shell input swipe 2093 1346 2093 1346 79
  sleep 2.0
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
