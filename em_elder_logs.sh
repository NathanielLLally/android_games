#!/bin/sh
#
#  resolution 2560x1600
#
declare -i limit=0 n=0 once=1 rfreq=3
if [ -n "$1" ]; then limit=$(($1/28+1)); fi
while [ \( $(($limit-$n)) -gt 0 \) -o \( $limit -eq 0 \) ]; do
  adb shell input swipe 298 989 298 989 109
  sleep 4.0
  adb shell input swipe 2062 235 2062 235 140
  sleep 1.5
  adb shell input swipe 2306 729 2306 729 79
  sleep 3.8
  adb shell input swipe 1989 1445 1989 1445 131
  sleep 36.7
  adb shell input swipe 1526 714 1526 714 102
  sleep 1.5
  adb shell input swipe 1989 1445 1989 1445 131
  sleep 19.2
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
    rh=$(bc <<< "scale=3;$rmin/60")
    output+=" of $(($limit - $n)), eta $eta ($rh h)"
  fi
  echo $output
done
