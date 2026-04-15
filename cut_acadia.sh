#!/bin/sh
while [ 1 ]; do
adb shell input swipe 935 938 935 938 162
sleep 17
adb shell input swipe 951 221 951 221 144
sleep 17
adb shell input swipe 1495 390 1495 390 1078
sleep 1.0
adb shell input swipe 1271 289 1271 289 193
sleep 1.7
adb shell input swipe 778 708 778 708 193
sleep 2.4
adb shell input swipe 1266 1071 1266 1071 197
52
done
