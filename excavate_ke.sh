#!/bin/sh
#
#culinarum- level 100
#  imperial steel, ancient vis, vellum, blood of orcus
#

KEEPALIVE=5
DISPLAYTIME=60
PETDELAY=1
PETFREQ=600
PETTIME=$PETDELAY
deposit_freq=180
deposit_time=$deposit_freq

while [ 1 ]; do
  if [ $(($SECONDS%$DISPLAYTIME)) -eq 0 ]; then
    echo "$(($SECONDS/60)) min $(($SECONDS%60)) sec"
  fi
  if [ $SECONDS -ge $(($PETTIME-15)) ]; then
    echo "$(($PETTIME-$SECONDS))s until pet"
  fi
  if [ $SECONDS -ge $PETTIME ]; then
    echo "$SECONDS do pet"
    adb shell input swipe 1097 544 1097 544 157
    sleep 4
    adb shell input swipe 1889 901 1889 901 162
    sleep 1.5
    adb shell input swipe 1518 953 1518 953 161
    sleep 1.5
    adb shell input swipe 1046 1075 1046 1075 161
    sleep 1.5
    adb shell input swipe 1276 819 1276 819 175
    sleep 1.5
    adb shell input swipe 1871 313 1871 313 197
    sleep 1.5
    adb shell input swipe 799 560 799 560 180
    deposit_time=$(($SECONDS+$deposit_freq))
    #ranges from 8 seconds to 12
    PETTIME=$(($SECONDS+$PETFREQ+5))
    echo "$(($SECONDS/60)) min $(($SECONDS%60)) sec, pet done, next pet at $(($PETTIME/60)) min $(($PETTIME%60)) sec"
  elif [ $SECONDS -eq $deposit_time ]; then
    adb shell input swipe 1097 544 1097 544 157
    sleep 4
    adb shell input swipe 799 560 799 560 180
    deposit_time=$(($SECONDS+$deposit_freq))
  elif [ $(($SECONDS%$KEEPALIVE)) -eq 0 ]; then
    # currency on 
    adb shell input swipe 1726 968 1726 968 285
    sleep 0.7
  else
    sleep 1
  fi
done

