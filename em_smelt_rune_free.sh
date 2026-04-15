#!/bin/sh
#
#  resolution 2560x1600
#
declare -i limit=0 n=0 once=1 rfreq=3
declare -i unit_input=28
if [ -n "$1" ]; then
  limit=$(($1/$unit_input+1))
  out=$(bc <<< "scale=0;($1 + $1*0.1)/1")
fi
while [ \( $(($limit-$n)) -gt 0 \) -o \( $limit -eq 0 \) ]; do
  adb shell input swipe 1306 1187 1306 1187 112
  #adb shell input swipe 1321 1304 1321 1304 99
  sleep 2
  adb shell input swipe 388 1488 388 1488 122
  sleep 1.5
  if [ $once -gt 0 ]; then
    now=$(date "+%r %a")
    echo "begin $now"
    #select luminite
    adb shell input swipe 453 485 453 485 116
  sleep 1.5
    once=0
  fi

  adb shell input swipe 1980 1472 1980 1472 100
  sleep 53

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
    rh=$(bc <<< "scale=3;$rmin/60")

    limit=$(($1/$unit_input+1))
    material=$((($limit-$n-1)*$unit_input))
    output+=" of $(($limit - $n)), eta $eta ($rh hours) $material"
  fi
  echo $output
done
