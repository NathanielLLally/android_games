#!/bin/sh
while [ 1 ]; do
adb shell input swipe 1199 260 1199 260 133
sleep 2
adb shell input swipe 1963 1008 1963 1008 208
sleep 2
adb shell input swipe 1366 515 1366 515 141
sleep 29
adb shell input swipe 1954 880 1954 880 166
sleep 2
done
