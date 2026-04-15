#!/bin/sh
while [ 1 ]; do
adb shell input swipe 980 907 980 907 157
sleep 18
adb shell input swipe 970 272 970 272 127
sleep 18
adb shell input swipe 1724 890 1724 890 193
done
