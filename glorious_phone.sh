#!/bin/sh
#
#  resolution 2400x1080
#
#these are for reverse_landscape
  #furnace top of screen over progress box
furnace="adb shell input swipe 1194 58 1194 58 408"
 #deposit ore over action button #6
deposit="adb shell input swipe 491 1020 491 1020 157"
 #immaculate alloy
immaculate="adb shell input swipe 565 591 565 591 173"
 #glorious bar
glorious="adb shell input swipe 1156 197 1156 197 173"
  #begin button over a blank right side #6 action bar button
begin="adb shell input swipe 1943 990 1943 990 182"

#normal landscape
  #furnace top of screen over progress box
furnace="adb shell input swipe 1194 58 1194 58 408"
 #deposit ore over action button #6
deposit="adb shell input swipe 491 1020 491 1020 157"
 #immaculate alloy
immaculate="adb shell input swipe 635 591 635 591 173"
 #glorious bar
glorious="adb shell input swipe 1236 197 1236 197 173"
  #begin button over a blank right side #6 action bar button
begin="adb shell input swipe 1943 990 1943 990 182"

#number of folds completed in the first bar
BARFOLD=0
BARTOTAL=9999
if [ $(($1)) -gt 0 ]; then
  BARFOLD=$1
fi
if [ $(($2)) -gt 0 ]; then
  BARTOTAL=$2
fi

# loop counter
N=0

#total number of bars completed (including partial)
BAR=0

while [ $(bc <<< "scale=0;($BARTOTAL-$BAR)/1") -gt 0 ]; do

  # total loop iterations remaining
  NT=$(((1001-$BARFOLD)/60))

  #  bar progress remainder on last loop 
  R=$((1001-60*$NT-$BARFOLD))
$($furnace)
sleep 2

if [ \( $N -eq 0 \) -o \( $N -eq $(($NT+1)) \) ]; then
 echo "depositing ore, selecting bar $BAR"
 $($deposit)
 sleep 1
 $($immaculate)
 sleep 1.2
 $($glorious)
 sleep 1
fi
$($begin)
# seconds to sleep on loop iteration of partial progress 
S=$(bc <<< "$R * 1.2 + 1")
N=$(($N+1))
if [ $N -eq $(($NT+1)) ]; then
  N=0
  BAR=$(bc <<< "scale=3;$BAR+(1-$BARFOLD/1001)")
  echo "sleep $S"
  sleep $S
  SPB=$(bc <<< "$SECONDS/$BAR")
 echo "seconds per bar $SPB, $(($SPB/60)) min $(($SPB%60)) sec"
  BARFOLD=0
else
  echo "sleep 73"
 sleep 73
fi
  echo "loop $N of $(($NT+1)), bar $BAR, $(($SECONDS/60)) min $(($SECONDS%60)) sec"

done
