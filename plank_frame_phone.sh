#!/bin/sh
#
#  resolution 2560x1600
#
#
#bank="adb shell input swipe 349 997 349 997 130"
#sawmill="adb shell input swipe 2306 729 2306 729 79"
#sawmill_at_sawmill="adb shell input swipe 1526 714 1526 714 102"
#woodworking="adb shell input swipe 2109 403 2109 403 85"

bank="adb shell input swipe 621 727 621 727 97"
sawmill="adb shell input swipe 1814 410 1814 410 66"
sawmill_at_sawmill="adb shell input swipe 1330 477 1331 473 107"
woodworking="adb shell input swipe 1650 250 1650 250 83"

ready=0
declare -i choice=0
while [ $ready -eq 0 ]; do
  echo "1: bank from sawmill"
  echo "2: sawmill from bank"
  echo "3: sawmill from sawmill"
  echo "4: woodworking bench from bank"
  echo "5: ready to start"
  read choice
  if [ $choice -eq 1 ]; then $($bank); fi
  if [ $choice -eq 2 ]; then $($sawmill); fi
  if [ $choice -eq 3 ]; then $($sawmill_at_sawmill); fi
  if [ $choice -eq 4 ]; then $($woodworking); fi
  if [ $choice -eq 5 ]; then ready=1; fi
done

construct="adb shell input swipe 1944 973 1944 973 99"
preset_11="adb shell input swipe 2044 168 2044 168 125"
preset_12="adb shell input swipe 2113 177 2113 177 157"
preset_13="adb shell input swipe 2175 185 2175 185 133"
declare -i limit=0 n=0 once=1 rfreq=3
if [ -n "$1" ]; then limit=$(($1/(28*4)+1)); fi
while [ \( $(($limit-$n)) -gt 0 \) -o \( $limit -eq 0 \) ]; do
  for i in {1..4}; do
    $($bank)
    sleep 4
    $($preset_12)
    sleep 1
    $($sawmill)
    sleep 5
    $($construct)
    sleep 18
  done

  $($bank)
  sleep 5
  $($preset_13)
  sleep 2
  $($woodworking)
  sleep 5
  $($construct)
  sleep 68

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
    
    materials=$((($limit-$n)*28*4))
    output+=" of $(($limit - $n)), eta $eta ($rh h) materials $materials"
  fi
  echo $output
done
