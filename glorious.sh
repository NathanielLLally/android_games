#!/bin/sh
while [ 1 ]; do
adb shell input swipe 1221 304 1221 304 132
sleep 1.4
adb shell input swipe 491 1020 491 1020 157
sleep 1.2
adb shell input swipe 1943 990 1943 990 182
sleep 74.5
done
