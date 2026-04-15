#!/bin/sh
while [ 1 ]; do
  adb shell input swipe 1056 67 1056 67 18
sleep 4
adb shell input swipe 1704 998 1704 998 141
sleep 3
adb shell input swipe 1322 494 1322 494 74
sleep 37.9
  adb shell input swipe 1056 67 1056 67 18
sleep 1.5
done
