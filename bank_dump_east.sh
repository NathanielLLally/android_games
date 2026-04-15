#!/bin/sh
#
#  resolution 2560x1600
#
declare -i limit=0 n=0 once=1 rfreq=3
declare -i unit_input=28
if [ -n "$1" ]; then limit=$(($1/$unit_input+1)); fi
while [ \( $(($limit-$n)) -gt 0 \) -o \( $limit -eq 0 \) ]; do
  adb shell input swipe 2251 261 2251 261 99
  sleep 8.1
  adb shell input swipe 1609 975 1609 975 123
  sleep 2.7
  adb shell input swipe 1971 1503 1971 1503 115
  sleep 2.1
  adb shell input swipe 2440 84 2440 84 117
  sleep 2.6
  adb shell input swipe 1984 153 1984 153 59
  sleep 9.4
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
