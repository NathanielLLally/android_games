#!/bin/sh
#
while [ 1 ]; do
#select and smelt
adb shell input swipe 909 667 909 667 215
sleep 1
adb shell input swipe 347 219 347 219 162
sleep 1
adb shell input swipe 1535 1101 1535 1101 179
sleep 112.917618

BALLS=1
while [ $BALLS -ge 0 ]; do
#select and smelt
adb shell input swipe 948 628 948 628 197
sleep 1
adb shell input swipe 97 1004 97 1004 197
sleep 1.516852
adb shell input swipe 828 219 828 219 175
sleep 1.463504
adb shell input swipe 1542 1134 1542 1134 193
sleep 255.796649
BALLS=`expr $BALLS - 1`

done

done
