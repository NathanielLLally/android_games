#!/bin/sh
while [ 1 ]; do
  adb shell input swipe 861 481 861 481 132
sleep 1.2
adb shell input swipe 491 1020 491 1020 157
sleep 1.2
adb shell input swipe 1976 1012 1976 1012 173
sleep 53
done
