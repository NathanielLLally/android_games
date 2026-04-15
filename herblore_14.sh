#!/bin/sh
while [ 1 ]; do
adb shell input swipe 951 118 951 118 157
sleep 2
adb shell input swipe 1754 190 1754 190 232
sleep 1
adb shell input swipe 1467 384 1467 384 179
sleep 1
adb shell input swipe 1502 1074 1502 1074 175
sleep 19
done
