#!/bin/sh
while [ 1 ]; do
adb shell input swipe 1722 972 1722 972 179
sleep 1
adb shell input swipe 1157 505 1157 505 158
sleep 1
adb shell input swipe 1166 486 1166 486 193
sleep 1
adb shell input swipe 204 817 204 817 179
sleep 29
done
