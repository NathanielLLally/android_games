#!/bin/sh
while [ 1 ]; do
adb shell input swipe 1284 661 1284 661 99
sleep 1.7
adb shell input swipe 537 1008 537 1008 182
sleep 3.2
adb shell input swipe 1707 1007 1707 1007 141
sleep 54
done
