#!/bin/sh
while [ 1 ]; do
adb shell input swipe 333 821 333 821 197
sleep 0.9
adb shell input swipe 1491 390 1491 390 211
sleep 3.5
done
