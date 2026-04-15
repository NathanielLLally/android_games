#!/bin/sh
while [ 1 ]; do
adb shell input swipe 817 612 817 612 109
sleep 1.6
adb shell input swipe 1614 268 1614 268 193
sleep 1.5
adb shell input swipe 1271 620 1271 620 122
sleep 1.7
adb shell input swipe 1452 1069 1452 1069 175
sleep 68.3
done
