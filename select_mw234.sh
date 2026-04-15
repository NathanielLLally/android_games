#!/bin/sh
while [ 1 ]; do
adb shell input swipe 977 319 977 319 127
sleep 1.7
adb shell input swipe 459 644 459 644 175
sleep 2.3
adb shell input swipe 703 361 703 361 162
sleep 4.0
adb shell input swipe 831 369 831 369 157
sleep 1.9
adb shell input swipe 939 359 939 359 175
sleep 3.4
done
