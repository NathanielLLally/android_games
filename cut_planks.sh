#!/bin/sh
adb shell input swipe 837 640 837 640 162
sleep 2
while [ 1 ]; do
adb shell input swipe 1622 185 1622 185 197
sleep 1
adb shell input swipe 1604 736 1604 736 215
sleep 4
adb shell input swipe 1443 1070 1443 1070 215
sleep 37
adb shell input swipe 401 473 401 473 157
sleep 4
done
