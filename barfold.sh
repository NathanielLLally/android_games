#!/bin/sh
BARFOLD=0
BARTOTAL=9999
if [ $(($1)) -gt 0 ]; then
  BARFOLD=$1
fi
if [ $(($2)) -gt 0 ]; then
  BARTOTAL=$2
fi

echo "barfold $BARFOLD bartotal $BARTOTAL"

NT=$(((1001-$BARFOLD)/60))
R=$((1001-60*$NT-$BARFOLD))
N=1
BAR=0
  BAR=$(bc <<< "scale=3;$BAR+(1-$BARFOLD/1001)")
  SPB=$(bc <<< "300/$BAR")

echo $NT
echo $R
S=$(bc <<< "$R * 1.2 + 3")
echo $S
echo "bar $BAR"
 echo "seconds per bar $SPB"

if [ \( $BARFOLD -eq 0 \) -o \( $N -eq $(($NT+1)) \) ]; then
  echo "new bar"
fi
