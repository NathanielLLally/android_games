#!/bin/sh
while [ 1 ]; do
adb shell input swipe 986 205 986 205 179
sleep 2
adb shell input swipe 1496 237 1496 237 197
sleep 1
adb shell input swipe 1469 394 1469 394 197
sleep 1
adb shell input swipe 1382 1070 1382 1070 193
sleep 20
done
