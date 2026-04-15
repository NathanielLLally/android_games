#!/bin/sh
#
while [ 1 ]; do
 echo "cast"
adb shell input swipe 452 1078 452 1078 100
adb shell input swipe 1465 398 1465 398 100
echo "sleep 3"
echo "."
sleep 1
echo "."
sleep 1
echo "."
sleep 1
done
