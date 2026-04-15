#!/bin/sh
#
#  resolution 2560x1600
#
declare -i limit=0 n=0 once=1 rfreq=3
declare -i unit_input=28
if [ -n "$1" ]; then limit=$(($1/$unit_input+1)); fi
while [ \( $(($limit-$n)) -gt 0 \) -o \( $limit -eq 0 \) ]; do
  adb shell input swipe 2268 261 2268 261 99
  sleep 9.1
  adb shell input swipe 1594 644 1594 644 77
  sleep 2.9
  adb shell input swipe 1982 1509 1982 1509 93
  sleep 2.0
  adb shell input swipe 2438 76 2438 76 67
  sleep 2.5
  adb shell input swipe 1967 148 1967 148 99
  sleep 9.0

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
    rmin=$(bc <<< "scale=3;$remain/60") 
    rh=$(bc <<< "scale=3;$rmin/60")
    output+=" of $(($limit - $n)), eta $eta ($rh hours)"
  fi
  echo $output
done
