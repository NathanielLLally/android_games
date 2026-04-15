#!/bin/sh
#
#  resolution 2560x1600
#
bank="adb shell input swipe 617 1128 617 1128 100"
sawmill="adb shell input swipe 2016 544 2016 544 100"
sawmill_at_sawmill="adb shell input swipe 1416 707 1416 707 100"
woodworking="adb shell input swipe 1787 415 1787 415 114"
construct="adb shell input swipe 1989 1445 1989 1445 131"
preset_11="adb shell input swipe 2062 235 2062 235 140"
preset_12="adb shell input swipe 2155 225 2155 225 77"
preset_13="adb shell input swipe 2245 235 2245 235 100"
declare -i limit=0 n=0 once=1 rfreq=3
if [ -n "$1" ]; then limit=$(($1/28+1)); fi
while [ \( $(($limit-$n)) -gt 0 \) -o \( $limit -eq 0 \) ]; do
    $($bank)
  sleep 4.0
  $($preset_12)
  sleep 1.5
    $($sawmill)
  sleep 3.8
    $($construct)
  sleep 19.8
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
  echo $output
done
