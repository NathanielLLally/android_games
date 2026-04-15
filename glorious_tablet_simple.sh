#!/bin/sh
#
COUNT=0
BAR=0
BARFOLD=0

while [ 1 ]; do
  #anvil
adb shell input swipe 973 359 973 359 127
sleep 2

 echo "depositing ore, selecting bar $BAR"
 #deposit ore over action button #6
 adb shell input swipe 280 1099 280 1099 140
 sleep 1
 #immaculate
 adb shell input swipe 348 637 348 637 144
 sleep 1.2
 #partially folded
 adb shell input swipe 944 217 944 217 158
 sleep 1
  #begin button over a blank right side #6 action bar button
adb shell input swipe 1464 1098 1464 1098 250

 sleep 75
done
