#!/bin/sh
while [ 1 ]; do
adb shell input swipe 774 668 774 668 144
sleep 2.9
adb shell input swipe 1491 230 1491 230 179
sleep 1.8
adb shell input swipe 1132 477 1132 477 175
sleep 2.5
adb shell input swipe 1423 1061 1423 1061 144
sleep 68.6
done
