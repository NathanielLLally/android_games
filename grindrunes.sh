#!/bin/sh
#
#  resolution 1920x1200
#
declare -i limit=0 n=0 once=1 rfreq=3
declare -i unit_input=60 m=25000
delay=0.17
limit=$(($m/$unit_input+1))
if [ -n "$1" ]; then limit=$(($1/$unit_input+1)); fi
while [ \( $(($limit-$n)) -gt 0 \) -o \( $limit -eq 0 \) ]; do
  adb shell input swipe 327 967 327 967 100
  adb shell input swipe 1459 387 1459 387 100
  sleep $delay
  n+=1
  spl=$(bc <<< "scale=3;$SECONDS/$n")
  nc=2
  totalms=200
  lag=$(bc <<< "scale=3;($spl-$delay-($totalms/1000))/$nc")


  remain=$(bc <<< "scale=0; (($limit-$n)*$spl)/1")
  eta=$(date "+%r %a" --date="$remain seconds")
  output="$spl s per loop, avg lag $lag"
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
    rh=$(bc <<< "scale=0;$rmin/60")
    rmi=$(bc <<< "scale=0;$rmin%60")
    material=$((($limit-$n-1)*$unit_input))
    output+=" of $(($limit - $n)), eta $eta ($rh hours $rmi mins) $material"
  fi
  echo $output
done
