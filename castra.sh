#!/bin/sh
while [ 1 ]; do
adb shell input swipe 1567 990 1567 990 144
sleep 2.5
adb shell input swipe 1872 897 1872 897 179
sleep 1.7
adb shell input swipe 1668 893 1668 893 175
sleep 1.0
adb shell input swipe 1458 388 1458 388 157
sleep 85.9
done
