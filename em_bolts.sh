#!/bin/sh
#
#  resolution 2560x1600
#
declare -i limit=0 n=0 once=1 rfreq=3 net=0 cost=0 profit=0
if [ -n "$1" ]; then limit=$1; fi
if [ -n "$2" ]; then net=$2; fi
if [ -n "$3" ]; then cost=$3; fi
while [ \( $(($limit-$n)) -gt 0 \) -o \( $limit -eq 0 \) ]; do
  adb shell input swipe 1270 200 1270 200 146
  sleep 2.5
  adb shell input swipe 1965 1469 1965 1469 83
  sleep 1.5
  adb shell input swipe 1555 780 1555 780 107
  sleep 36
  adb shell input swipe 1270 200 1270 200 146
  sleep 1.5
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
    rh=$(bc <<< "scale=0;$rmin/60")
    output+=" of $(($limit - $n)), eta $eta ($rh:$(($rmin%60)):$(($remain%60)))"
  fi
  loops=$(($limit-$n))
  tcost=$(($loops*$cost))
  tnet=$(($loops*$net))
  profit=$(($tnet - $tcost))
  if [ $profit -gt 0 ]; then
    output+=" profit $profit"
  fi
  echo $output
done
