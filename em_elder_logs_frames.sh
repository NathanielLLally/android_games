#!/bin/sh
#
#  resolution 2560x1600
#
#
#bank="adb shell input swipe 349 997 349 997 130"
#sawmill="adb shell input swipe 2306 729 2306 729 79"
#sawmill_at_sawmill="adb shell input swipe 1526 714 1526 714 102"
#woodworking="adb shell input swipe 2109 403 2109 403 85"

bank="adb shell input swipe 617 1128 617 1128 100"
sawmill="adb shell input swipe 2016 544 2016 544 100"
sawmill_at_sawmill="adb shell input swipe 1416 707 1416 707 100"
woodworking="adb shell input swipe 1787 415 1787 415 114"

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

construct="adb shell input swipe 1989 1445 1989 1445 131"
preset_11="adb shell input swipe 2062 235 2062 235 140"
preset_13="adb shell input swipe 2245 235 2245 235 100"
declare -i limit=0 n=0 once=1 rfreq=3
if [ -n "$1" ]; then limit=$(($1/(28*4)+1)); fi
while [ \( $(($limit-$n)) -gt 0 \) -o \( $limit -eq 0 \) ]; do
  for i in {1..4}; do
    $($bank)
    sleep 5
    $($preset_11)
    sleep 2
    $($sawmill)
    sleep 5
    $($construct)
    sleep 37
    $($sawmill_at_sawmill)
    sleep 2
    $($construct)
    sleep 19.2
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
