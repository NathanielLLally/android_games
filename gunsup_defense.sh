#!/bin/sh
#set -x
DIR="${BASH_SOURCE%/*}"
if [[ ! -d "$DIR" ]]; then DIR="$PWD"; fi
. "$DIR/utils.sh"

# realme
S="XG9LGEZX6L75QGMJ"

#blu pure
#S='FAMVRW9D9HDEHEWS'
CMD="adb -s $S shell "

PACKAGE="com.nhn.gunsup"
NOW=$(date '+%H%M%Y%m%d')
DEBUG=true
BASEC="2400 1080"

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

if [ "$conn" != "no active connection" ]; then
  ip=$(echo $conn | /home/nathaniel/bin/cperl '/src=(\d+\.\d+\.\d+\.\d+)/ && print $1')

  echo "current connection to gunsup server from $ip, exiting"
#  switchDisplayState off $S
#  exit 0
fi

echo "no conn, doing stuff"

switchDisplayState on $S

SIZE=$(getPhysicalSize)

echo $xy

$DEBUG && log 'prestart'


#force stop
$CMD am force-stop $PACKAGE
#guns up icon
#$CMD input tap 734 554
#
# instead use android monkey utility to start app (what if it was left running)
$CMD monkey -p $PACKAGE -c android.intent.category.LAUNCHER 1
sleep 35
$DEBUG && log 'start' 
#back button
$CMD input keyevent 4
sleep 2
#cancel (do you want to quit)

#this only needed if program wasnt stopped
#$CMD input tap 1400 630
#sleep 2
#war button
$DEBUG && log 'warbutton' || sleep 2
sleep 1
input tap 2200 950
#defense button

$DEBUG && log 'swipe' || sleep 2
sleep 1
#input swipe 1000 500 500 500 1000
$DEBUG && log 'defensebutton' || sleep 2
sleep 2
input tap 2315 360
#loadout
$DEBUG && log 'loadout' || sleep 
sleep 1
input tap 2200 975
#moveout
$DEBUG && log 'moveout' || sleep 2
sleep 1
input tap 2200 975
log 'postmoveout'
sleep 20 
#watch the battle
input swipe 2100 500 0 500 350
log 'swipe'
sleep 450
#battle over, continue - avoid pressing move out button
log 'continue'
input tap 2274 990
sleep 2
#accept
$DEBUG && log 'accept' || sleep 2
input tap 2340 990
sleep 1
#$DEBUG && log 'postaccept'

#home
#$CMD input keyevent 3
#
switchDisplayState off $S
