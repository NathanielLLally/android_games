#!/bin/sh
#
#  top down, full in zoom, tree directly to the right 
#  chop tree down, click on stump to move into position
#
#

echo  "top down, full in zoom, tree directly to the right"
echo "chop tree down, click on stump to move into position"
#
N=0
while [ 1 ]; do
adb shell input swipe 1724 978 1724 978 161
#adb shell input swipe 1727 975 1727 975 162
sleep 0.8
adb shell input swipe 1308 620 1308 620 122
#adb shell input swipe 1235 635 1235 635 157
sleep 1.2
adb shell input swipe 1308 620 1308 620 122
#adb shell input swipe 1235 635 1235 635 157
sleep 1
#hotkey 1 right side
adb shell input swipe 1591 824 1591 824 109

#hotkey 1 left side
#adb shell input swipe 213 828 213 828 162
sleep 25
done
