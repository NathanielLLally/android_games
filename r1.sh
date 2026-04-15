#!/bin/sh
#
#  resolution 2560x1600
#
declare -i limit=0 n=0 once=1 rfreq=3
if [ -n "$1" ]; then limit=$1; fi
while [ \( $(($limit-$n)) -gt 0 \) -o \( $limit -eq 0 \) ]; do
  adb shell input swipe 1250 36 1250 36 109
  sleep 4.1
  adb shell input swipe 926 296 926 296 115
  sleep 1.7
  adb shell input swipe 1992 1479 1992 1479 102
  sleep 2.5
  adb shell input swipe 1684 751 1684 751 78
  sleep 23.2
  adb shell input swipe 1261 190 1261 190 84
  sleep 0.5
  adb shell input swipe 1742 781 1742 781 91
  sleep 1.9
  adb shell input swipe 1263 131 1263 131 133
  sleep 2.5
  adb shell input swipe 1688 750 1686 750 108
  sleep 23.0
  adb shell input swipe 2131 1307 2131 1307 69
  sleep 1.3
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
