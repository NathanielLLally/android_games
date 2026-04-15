#!/bin/sh
adb shell input swipe 842 608 842 608 197
sleep 3.8
while [ 1 ]; do
adb shell input swipe 1202 471 1202 471 127
sleep 1.6
adb shell input swipe 1566 185 1566 185 175
sleep 2.5
adb shell input swipe 1871 892 1871 892 180
sleep 1.0
adb shell input swipe 1642 739 1642 739 281
sleep 3.1
adb shell input swipe 1456 1070 1456 1070 197
sleep 2.6
adb shell input swipe 1865 897 1865 897 215
sleep 14.0
adb shell input swipe 1534 885 1534 885 215
sleep 12.0
adb shell input swipe 1526 900 1526 900 215
sleep 9.5
adb shell input swipe 464 481 464 481 193
sleep 4.8
done
