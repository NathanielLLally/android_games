#!/bin/sh
while [ 1 ]; do
adb shell input swipe 932 746 932 746 109
sleep 2.0
adb shell input swipe 1571 1099 1571 1099 215
sleep 6.4
adb shell input swipe 207 826 207 826 179
sleep 68.9
done
