#!/bin/sh
while [ 1 ]; do
adb shell input swipe 1355 1089 1355 1089 158
sleep 15.6
adb shell input swipe 961 292 961 292 197
sleep 6.9
adb shell input swipe 1564 399 1564 399 898
sleep 1.6
adb shell input swipe 1309 293 1309 293 193
sleep 1.5
done
