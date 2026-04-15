#!/bin/sh
# 0 masterwork plate 1 curved 2 untempered 3 armour
USEPB=0
PLATE=0
PLATES=0
PLATETOTAL=9999
FREESPOT=0
if [ $(($1)) -gt 0 ]; then
  PLATE=$1
fi
if [ $(($2)) -gt 0 ]; then
  FREESPOT=$2
fi
if [ $(($3)) -gt 0 ]; then
  PLATETOTAL=$3
fi

POWERBURSTFREQ=120
PBTIMER=0
HEATFREQ=24
PBTIMER=0
DISPLAYTIME=5
START=0
PLATETIMER=0
PLATEFREQ=430
SLEEP=0

backpack=("1464 390" "1565 398" "1633 392" "1721 392" "1466 474" "1564 481" "1641 474" "1727 474" "1473 557" "1571 557" "1641 562" "1734 553" "1461 640" "1566 640" "1653 639" "1729 637" "1469 718" "1562 723" "1645 723" "1734 720" "1484 806" "1562 803" "1642 802" "1722 803" "1464 884" "1565 886" "1637 885" "1728 890")
#platefreq=("430" "430" "470" "40") 
#platefreq=("480" "480" "480" "40") 
platefreq=("530" "565" "620" "105") 

while [ $PLATES -le $PLATETOTAL ]; do
  if [ $SECONDS -ge $PLATETIMER ]; then
    echo "starting next plate"
    #click currency with backpack open in case of a continue dialog
    adb shell input swipe 1721 964 1721 964 197
    sleep 1
    #furnace top of screen over progress box
    adb shell input swipe 974 114 974 114 126
    sleep 2
    #glorious
    adb shell input swipe 467 639 467 639 180
    sleep 1
    if [ $PLATE -eq 0 ]; then
      PLATE=$(($PLATE+1))
      #masterwork plate
      adb shell input swipe 1062 229 1062 229 140
    elif [ $PLATE -eq 1 ]; then
      PLATE=$(($PLATE+1))
      #curved
      adb shell input swipe 703 361 703 361 162
    elif [ $PLATE -eq 2 ]; then
      PLATE=$(($PLATE+1))
      #untempered
      adb shell input swipe 831 369 831 369 157
    elif [ $PLATE -eq 3 ]; then
      PLATE=0
      FREESPOT=$(($FREESPOT+1))
      PLATES=$(($PLATES+1))
      #armour
      adb shell input swipe 939 359 939 359 175
      #with a powerburst
    fi
    sleep 1
    #begin button over a blank right side #6 action bar button
    adb shell input swipe 1464 1098 1464 1098 250
    sleep 6

    #Anvil
    adb shell input swipe 1165 582 1165 582 109
    sleep 1
    PLATETIMER=$(($SECONDS+${platefreq[$(($PLATE-1))]}+14))
    START=$SECONDS
    echo "plate start $(($SECONDS/60)) min $(($SECONDS%60)) sec - $PLATETIMER"
    SLEEP=1
  fi
  if [ $((($SECONDS-$START+$HEATFREQ)%$HEATFREQ)) -eq 0 ]; then
    #Superheat 
    adb shell input swipe 333 821 333 821 197
    sleep 1
    #inventory #1
    #handle mimic tokens
    current="adb shell input swipe ${backpack[$FREESPOT]} ${backpack[$FREESPOT]} 100"
    echo "superheat $(($SECONDS/60)) min $(($SECONDS%60)) sec $current"
    $($current)
    SLEEP=1
  fi
  if [ \( $SECONDS -ge $PBTIMER \) -a \( $PLATE -lt 3 \) -a \( $USEPB -ne 0 \) ]; then
    adb shell input swipe 208 821 208 821 179
    sleep 1
    PBTIMER=$(($SECONDS+$POWERBURSTFREQ+2))
    echo "powerburst $(($SECONDS/60)) min $(($SECONDS%60)) sec"
    SLEEP=1
  fi
  if [ $SLEEP -eq 0 ]; then
    sleep 1
    echo "$SECONDS, $(($SECONDS-$START)), $(($SECONDS/60)) min $(($SECONDS%60)) sec, PLATE=$((PLATE-1))"
  else 
    SLEEP=0
  fi

done
