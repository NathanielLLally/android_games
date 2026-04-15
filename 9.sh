#!/bin/sh
while [ 1 ]; do
adb shell input swipe 1159 608 1159 608 1007
sleep 4.1
adb shell input swipe 913 623 913 623 175
sleep 1.5
adb shell input swipe 973 841 973 841 157
sleep 2.0
adb shell input swipe 1383 1066 1383 1066 162
sleep 13.2
done
