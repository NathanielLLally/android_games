#!/bin/sh
while [ 1 ]; do
adb shell input swipe 971 358 971 358 109
sleep 2.5
adb shell input swipe 333 1018 333 1018 152
sleep 1.5
adb shell input swipe 499 186 499 186 53
sleep 1.5
adb shell input swipe 1716 984 1716 984 196
sleep 54
#echo $SECONDS
done
