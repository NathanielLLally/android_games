#!/bin/sh
#
#  resolution 1920x1200
#
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

  #furnace top of screen over progress box
adb shell input swipe 1217 102 1217 102 126
sleep 2

if [ \( $N -eq 0 \) -o \( $N -eq $(($NT+1)) \) ]; then
 echo "depositing ore, selecting bar $BAR"
 #deposit ore over action button #6
 adb shell input swipe 350 989 350 989 140
 sleep 1
 #immaculate alloy
 adb shell input swipe 435 573 435 573 144
 sleep 1.2
 #glorious bar
 adb shell input swipe 1180 195 1180 195 158
 sleep 1
fi
  #begin button over a blank right side #6 action bar button
adb shell input swipe 1830 988 1830 988 250

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
