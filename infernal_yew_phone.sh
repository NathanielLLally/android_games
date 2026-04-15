#!/bin/sh
declare -i limit=0 n=0
declare -i unit_input=13
if [ -n "$1" ]; then limit=$(($1/$unit_input+1)); fi
while [ \( $(($limit-$n)) -gt 0 \) -o \( $limit -eq 0 \) ]; do
  adb shell input swipe 1406 564 1406 564 955
  sleep 0.8
  adb shell input swipe 1147 581 1147 581 125
  sleep 1
  adb shell input swipe 1850 357 1850 357 140
  sleep 1
  adb shell input swipe 1900 958 1900 958 132
  sleep 35
  n+=1
  spl=$(bc <<< "scale=3;$SECONDS/$n")
  remain=$(bc <<< "scale=0; (($limit-$n)*$spl)/1")
  eta=$(date "+%r %a" --date="$remain seconds")
  output="$spl s per loop, $n loops"
  if [ $limit -gt 0 ]; then
    rmin=$(bc <<< "scale=3;$remain/60") 
    rh=$(bc <<< "scale=3;$rmin/60")

    limit=$(($1/$unit_input+1))
    material=$((($limit-$n-1)*$unit_input))
    output+=" of $(($limit - $n)), eta $eta ($rh hours) $material"
  fi
  echo $output
done
