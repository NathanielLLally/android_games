#!/bin/sh
#
#  resolution 2560x1600
#
declare -i limit=0 n=0 once=1 rfreq=3
if [ -n "$1" ]; then limit=$(($1/60)); fi
while [ \( $(($limit-$n)) -gt 0 \) -o \( $limit -eq 0 \) ]; do

  #furnace
  adb shell input swipe 1274 1122 1274 1122 197
  sleep 1.5

  if [ $once -gt 0 ]; then
    now=$(date "+%r %a")
    echo "begin $now"
    adb shell input swipe 152 1321 152 1321 91
    sleep 1.5
    adb shell input swipe 1103 319 1103 319 139
    sleep 1.5
    once=0
  fi

  # begin over empty button
  adb shell input swipe 1963 1479 1963 1479 149
  sleep 146
  n+=1
  spl=$(bc <<< "scale=3;$SECONDS/$n")
  remain=$(bc <<< "scale=0; (($limit-$n)*$spl)/1")
  eta=$(date "+%r %a" --date="$remain seconds")
  output="$spl s per loop, $n loops"
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
