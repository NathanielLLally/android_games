#!/bin/sh
while [ 1 ]; do
adb shell input swipe 996 311 996 311 140
sleep 1.8
adb shell input swipe 1399 1081 1399 1081 140
sleep 2.6
adb shell input swipe 1150 608 1150 608 122
sleep 19.2
adb shell input swipe 1721 976 1721 976 175
sleep 2.2
done
