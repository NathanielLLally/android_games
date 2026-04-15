#!/bin/sh
HEATFREQ=24
START=0
while [ 1 ]; do
  if [ $((($SECONDS-$START)%$HEATFREQ)) -eq 0 ]; then
adb shell input swipe 337 819 337 819 162
sleep 0.6
adb shell input swipe 1471 390 1466 390 175
sleep 1
    echo "superheat $(($SECONDS/60)) min $(($SECONDS%60)) sec"
  else 

    echo "$(($SECONDS/60)) min $(($SECONDS%60)) sec"
    sleep 1
  fi
done
