#!/bin/sh
while [ 1 ]; do
adb shell input swipe 1138 344 1138 344 116
sleep 1.6
adb shell input swipe 491 1020 491 1020 157
sleep 1.5
adb shell input swipe 667 186 664 185 166
sleep 1.1
adb shell input swipe 1976 1012 1976 1012 173
sleep 53
done
