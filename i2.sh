#!/bin/sh
#
#  resolution 1920x1200
#
declare -i limit=0 n=0 once=1 rfreq=3
declare -i unit_input=28
if [ -n "$1" ]; then limit=$(($1/$unit_input+1)); fi
while [ \( $(($limit-$n)) -gt 0 \) -o \( $limit -eq 0 \) ]; do
  adb shell input swipe 694 625 694 625 127
  sleep 1
  adb shell input swipe 1489 233 1489 233 197
  sleep 1
  adb shell input swipe 332 972 332 972 145
  sleep 1
  adb shell input swipe 1637 1062 1637 1062 91
  sleep 47
    n+=1
  while  [ $(($n%13)) -ne 0 ]; do
    adb shell input swipe 706 580 706 580 973
    sleep 3
    adb shell input swipe 1198 573 1198 573 215
    sleep 1
    adb shell input swipe 341 969 341 969 211
    sleep 1
    adb shell input swipe 1667 1063 1667 1063 175
    sleep 47
    n+=1
    echo "$n"
    echo "$(($n%27))"
  done
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
    material=$((($limit-$n-1)*$unit_input))
    output+=" of $(($limit - $n)), eta $eta ($rh hours) $material"
  fi
  echo $output
done
