#!/bin/sh
while [ 1 ]; do
adb shell input swipe 767 659 767 659 144
sleep 3.5
adb shell input swipe 1498 249 1498 249 175
sleep 2
adb shell input swipe 1197 549 1197 549 179
sleep 3
adb shell input swipe 1400 1070 1400 1070 161
sleep 18
done
