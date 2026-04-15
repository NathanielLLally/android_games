#!/bin/sh
while [ 1 ]; do
adb shell input swipe 1567 388 1567 388 1131
sleep 1.1
adb shell input swipe 1323 280 1323 280 228
sleep 2.9
adb shell input swipe 1354 1071 1354 1071 215
sleep 18
adb shell input swipe 977 872 977 872 180
sleep 18
adb shell input swipe 1580 387 1580 387 1008
sleep 1.0
adb shell input swipe 1366 286 1366 286 158
sleep 2.1

adb shell input swipe 1355 1089 1355 1089 158
sleep 1.5
done
