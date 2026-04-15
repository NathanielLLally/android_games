#!/bin/sh
#set -x
DIR="${BASH_SOURCE%/*}"
if [[ ! -d "$DIR" ]]; then DIR="$PWD"; fi
. "$DIR/utils.sh"

# realme
S="XG9LGEZX6L75QGMJ"

S="0123456789ABCDEF"
S="192.168.0.158:5555"

#blu pure
#S='FAMVRW9D9HDEHEWS'
CMD="adb -s $S shell "

PACKAGE="com.nhn.gunsup"
NOW=$(date '+%H%M%Y%m%d')
DEBUG=true
BASEC="2400 1080"
BASEC="1920 1200"

input() {
  itype=$1
  bx=$(echo $BASEC | cut -d" " -f1)
  by=$(echo $BASEC | cut -d" " -f2)

  px=$(echo $SIZE | cut -d" " -f1)
  py=$(echo $SIZE | cut -d" " -f2)


  tx=$(echo "scale=2;$bx / $px" | bc)
  ty=$(echo "scale=2;$by / $py" | bc)

#  echo "bx $bx px $px ty $ty tx $tx"
#  echo "scale=2;2400 / 1920" | bc

  echo "itype: $itype $2 $3 $4 $5 $6"
  if [ "$itype" = "tap" ]; then
	  x=$2
	  y=$3
  	xx=$(echo "$x / $tx" | bc)
	yy=$(echo "$y / $ty" | bc)
    echo $CMD input tap $xx $yy
   $CMD input tap $xx $yy
  elif [ "$itype" = "swipe" ]; then
	  x1=$2
	  y1=$3
	  x2=$4
	  y2=$5
	  time=$6
  	xx1=$(echo "$x1 / $tx" | bc)
	yy1=$(echo "$y1 / $ty" | bc)
  	xx2=$(echo "$x2 / $tx" | bc)
	yy2=$(echo "$y2 / $ty" | bc)
    echo $CMD input swipe $xx1 $yy1 $xx2 $yy2 $time
    $CMD input swipe $xx1 $yy1 $xx2 $yy2 $time
  fi
}  	  

checkTraffic() {
  if [ -z "$IP" ]; then
    IP=$(getIP $S)
  fi
  conn=$(ssh -i /home/nathaniel/.ssh/wings.pem root@192.168.1.1 "cat /proc/net/nf_conntrack | grep 115.88.123.76")
  if [ -z "$conn" ]; then
    echo "no active connection"
  else
    echo $conn
  fi
}

#  when the laptop sleeps, usb mode reverts to charge only on this device
#  write code for the case where wifi isnt connected
adbWifi() {
  port=5555
  ip=$(getIP $S | sed 's|\r$||')
  echo $ip
  if [ -n "$(echo $ip | grep "insufficient permissions")" ]; then
    echo "we have no adb permissions! check usb mode is mtp, usb debugging is on, and computer is trusted or that device is unlocked at home with smart lock"
  else
    IP=$ip
    echo "found ip for device $S: $ip"
  fi

  # in the event we have usb->adb mode set,
  # turn on adb over wifi
  # this will be our fallback connection to the device
  connected=$(adb devices | grep $port | sed 's|\r$||')
  if [ -z "$connected" ]; then
    echo "not connected"
    adb tcpip $port
    adb connect "$ip"
  fi

  #in the event usb mode is set to charge only, reinstate 
  connected=$(adb devices | grep $port | cut -f1 | sed 's|\r$||') 
  if [ -n "$connected" ]; then
    echo "adb over tcpip is connected"

    #  restore usb mode incase somehow it reverted to charge only
    #
    adb -s $connected shell svc usb setFunction mtp
  fi
}

log() {
  if [ ! -d /tmp/adbdroid ]; then
    mkdir /tmp/adbdroid
  fi
  tag=$1||'log'
  log="/tmp/adbdroid/$PACKAGE.$NOW.$tag.png"
  echo "$log"

  $CMD 'stty raw 2>/dev/null; screencap -p' | sed 's|\r$||' > $log 
}

getPhysicalSize() {
  yx=$($CMD wm size | cut -d' ' -f3 | sed 's|\r$||')
  x=$(echo $yx | cut -dx -f2)
  y=$(echo $yx | cut -dx -f1)
  echo "$x $y"
}

#adbWifi
#echo "IP: $IP"

#conn=$(checkTraffic)


SIZE=$(getPhysicalSize)
echo $SIZE

echo $xy

while [ 1 ]; do
#war button
input tap 1080 556
sleep 4
input tap 1080 556
sleep 4
input tap 1080 556
sleep 4

#input tap 1488 403
input tap 1475 390
sleep 2

done
