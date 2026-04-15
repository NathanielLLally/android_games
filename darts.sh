#!/bin/sh
while [ 1 ]; do
adb shell input swipe 980 307 980 307 144
sleep 1.6
adb shell input swipe 1594 1098 1594 1098 250
sleep 1.6
adb shell input swipe 1164 608 1164 608 193
sleep 35
adb shell input swipe 1609 1085 1609 1085 246
sleep 1.5
done
