#!/bin/sh
#
#  resolution 2560x1600
#
declare -i limit=0 n=0 once=1 rfreq=3
declare -i unit_input=28
if [ -n "$1" ]; then limit=$(($1/$unit_input+1)); fi
    now=$(date "+%r %a")
    echo "begin $now"
while [ \( $(($limit-$n)) -gt 0 \) -o \( $limit -eq 0 \) ]; do

  if [ $once -gt 0 ]; then
#3rd preset
    adb shell input swipe 983 822 983 822 101
    sleep 2.0
    adb shell input swipe 2160 226 2160 226 99
    sleep 2.0

  else
  #last preset 
    adb shell input swipe 974 819 974 819 879
    sleep 0.8
    adb shell input swipe 1213 808 1213 808 78
    sleep 1
  fi

#5th button (yew)
#    adb shell input swipe 617 1486 617 1486 69
#    7th button, willow
  adb shell input swipe 457 1469 457 1469 187
    sleep 2
  if [ $once -gt 0 ]; then
#do incence button
    adb shell input swipe 2095 1218 2095 1218 92
    sleep 2
    once=0
  fi
  adb shell input swipe 1974 1410 1974 1410 61
  sleep 42
  n+=1
  spl=$(bc <<< "scale=3;$SECONDS/$n")
  remain=$(bc <<< "scale=0; (($limit-$n)*$spl)/1")
  eta=$(date "+%r %a" --date="$remain seconds")
  output="$spl s per loop, $n loops"
  if [ $(($n%$rfreq)) -eq 0 ]; then
    output+=""
  fi
  if [ $limit -gt 0 ]; then
    rmin=$(bc <<< "scale=3;$remain/60") 
    rh=$(bc <<< "scale=0;$rmin/60")
    rmi=$(bc <<< "scale=0;$rmin%60")
    material=$((($limit-$n-1)*$unit_input))
    output+=" of $(($limit - $n)), eta $eta ($rh hours $rmi mins) $material"
  fi
  echo $output
done
