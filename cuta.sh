#!/bin/sh
while [ 1 ]; do
  #south tree
adb shell input swipe 980 907 980 907 157
sleep 18
#top left log
adb shell input swipe 1564 388 1564 388 933
sleep 1.5
adb shell input swipe 1307 288 1307 288 158
sleep 2.3
adb shell input swipe 1270 1066 1270 1066 162
sleep 20
#north tree
adb shell input swipe 970 272 970 272 127
sleep 18
adb shell input swipe 1567 402 1567 402 902
sleep 1.5
adb shell input swipe 1331 297 1331 297 193
sleep 2.0
adb shell input swipe 1386 1065 1386 1065 179
sleep 20
done
