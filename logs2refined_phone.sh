#!/bin/sh
#

while [ 1 ]; do
adb shell input swipe 844 539 844 539 158
sleep 3.6
adb shell input swipe 2038 180 2038 180 175
sleep 2 
adb shell input swipe 1563 548 1563 548 139
sleep 3.7
adb shell input swipe 1695 971 1695 971 140
sleep 36.7
adb shell input swipe 1274 523 1274 523 123
sleep 2
adb shell input swipe 1633 970 1633 970 158
sleep 19.7
done
