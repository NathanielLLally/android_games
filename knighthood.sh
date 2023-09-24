#!/bin/sh

DIR="${BASH_SOURCE%/*}"
if [[ ! -d "$DIR" ]]; then DIR="$PWD"; fi
. "$DIR/utils.sh"


S="XG9LGEZX6L75QGMJ"
CMD="/usr/bin/adb -s $S shell "

PACKAGE="com.king.knightsrage"
NOW=$(date '+%H%M%Y%m%d')
DEBUG=true

checkTraffic() {
  conn=$(ssh -i /home/nathaniel/.ssh/wings.pem root@192.168.1.1 "cat /proc/net/nf_conntrack | grep ")
  if [ -z "$conn" ]; then
    echo "no active connection"
  else
    echo $conn
  fi
}

#  when the laptop sleeps, usb mode reverts to charge only on this device
usbWifi() {
  port=5555
  ip=$(getIP $S)
  echo "found ip for device $S: $ip"
  connected=$(adb devices | grep $port) 
  if [ -z "$connected" ]; then
    echo "not connected"
    adb tcpip $port
    adb connect $ip
  fi

  connected=$(adb devices | grep $port) 
  if [ -n "$connected" ]; then
    echo "adb over tcpip is connected"
    adb -s $ip:5555 shell svc usb setFunctions mtp
    CMD="/usr/bin/adb -s $ip shell "
  fi
}

log() {
  if [ ! -d /tmp/adbdroid ]; then
    mkdir /tmp/adbdroid
  fi
  tag=$1||'log'
  log="/tmp/adbdroid/$PACKAGE.$NOW.$tag.png"

  $CMD 'stty raw 2>/dev/null; screencap -p' > $log 
  echo $log
}

conn=$(checkTraffic)

usbWifi

if [ "$conn" != "no active connection" ]; then
  ip=$(echo $conn | /home/nathaniel/bin/cperl '/src=(\d+\.\d+\.\d+\.\d+)/ && print $1')

  echo "current connection to knighthood server from $ip, exiting"
  exit 0
fi

echo "no conn, doing stuff"

$DEBUG && log 'prestart'

switchDisplayState on

#force stop
$CMD am force-stop $PACKAGE
#guns up icon
#$CMD input tap 734 554
#
# instead use android monkey utility to start app (what if it was left running)
$CMD monkey -p $PACKAGE -c android.intent.category.LAUNCHER 1
sleep 5
$DEBUG && log 'start' 

cap=$(log 'loading')
echo $cap


#back button
$CMD input keyevent 4
sleep 1
#cancel (do you want to quit)

#this only needed if program wasnt stopped
#$CMD input tap 1400 630
#sleep 2
#
$CMD input tap 940 760
sleep 1
$CMD input keyevent 4
sleep 1
$CMD input tap 940 760


#home
#$CMD input keyevent 3
