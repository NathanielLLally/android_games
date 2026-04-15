#!/bin/sh
while [ 1 ]; do
adb shell input swipe 860 482 860 482 144
sleep 2.1
adb shell input swipe 468 1085 468 1085 144
sleep 0.3
adb shell input swipe 1461 386 1461 386 175
sleep 0.7
done
while [ 1 ]; do
adb shell input swipe 1308 481 1308 481 232
sleep 3.0
adb shell input swipe 754 467 754 467 179
sleep 1.4
adb shell input swipe 1840 48 1840 48 210
sleep 1.5
adb shell input swipe 557 478 557 478 175
sleep 5.0
done
