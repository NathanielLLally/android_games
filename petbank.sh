#!/bin/sh
while [ 1 ]; do
adb shell input swipe 1874 894 1869 894 197
sleep 1.1
adb shell input swipe 1513 957 1513 957 193
sleep 2.2
adb shell input swipe 996 1070 996 1070 179
sleep 2.5
adb shell input swipe 1249 824 1249 824 197
sleep 3.6
done
