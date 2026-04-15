#!/bin/sh
#
#  resolution 2560x1600
#
declare -i limit=0 n=0 once=1 rfreq=3
declare -i unit_input=28
if [ -n "$1" ]; then limit=$(($1/$unit_input+1)); fi
while [ \( $(($limit-$n)) -gt 0 \) -o \( $limit -eq 0 \) ]; do
  adb shell input swipe 953 786 953 786 1020
  sleep 1.4
  adb shell input swipe 1280 802 1280 802 61
  sleep 2.1
  adb shell input swipe 647 1274 647 1274 75
  sleep 1.5
  adb shell input swipe 2112 1468 2112 1468 117
  sleep 37.1
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
    material=$((($limit-$n-1)*$unit_input))
    output+=" of $(($limit - $n)), eta $eta ($rh hours) $material"
  fi
  echo $output
done
