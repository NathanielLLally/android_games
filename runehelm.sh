#!/bin/sh
#
#  resolution 2560x1600
#
declare -i limit=0 n=0 once=1 rfreq=3
if [ -n "$1" ]; then limit=$1; fi
while [ \( $(($limit-$n)) -gt 0 \) -o \( $limit -eq 0 \) ]; do
  adb shell input swipe 1229 61 1229 61 59
  sleep 2.5
  adb shell input swipe 1862 335 1862 335 60
  sleep 1.5
  adb shell input swipe 2003 1492 2003 1492 77
  sleep 2.1
  adb shell input swipe 1694 740 1694 740 100
  sleep 29.2
  adb shell input swipe 1228 131 1228 131 77
  sleep 2.5
  adb shell input swipe 1705 737 1705 737 55
  sleep 27.7
  adb shell input swipe 2121 1329 2121 1329 67
  sleep 1.4
  adb shell input swipe 1263 161 1263 161 38
  sleep 1.6
  adb shell input swipe 1955 335 1955 335 53
  sleep 1.5
  adb shell input swipe 1996 1470 1996 1470 90
  sleep 2
  adb shell input swipe 1708 751 1708 751 35
  sleep 25.8
  adb shell input swipe 1205 107 1205 107 59
  sleep 2.6
  adb shell input swipe 1697 768 1697 768 43
  sleep 20.7
  adb shell input swipe 1181 103 1181 103 85
  sleep 1.7
  adb shell input swipe 1694 751 1694 751 59
  sleep 24.9
  adb shell input swipe 2109 1311 2109 1311 21
  sleep 1.7
  adb shell input swipe 1196 112 1196 112 91
  sleep 1.6
  adb shell input swipe 2038 312 2038 312 91
  sleep 1.2
  adb shell input swipe 1968 1464 1968 1464 76
  sleep 2
  adb shell input swipe 1633 714 1670 718 29
  sleep 23.0
  adb shell input swipe 1155 96 1155 96 27
  sleep 1.8
  adb shell input swipe 1703 757 1703 757 61
  sleep 20.3
  adb shell input swipe 1165 84 1165 84 56
  sleep 2.3
  adb shell input swipe 1716 720 1716 720 196
  sleep 32.8
  adb shell input swipe 1176 109 1185 114 69
  sleep 1.6
  adb shell input swipe 1688 750 1688 750 85
  sleep 9.5
  adb shell input swipe 2138 1290 2138 1290 92
  sleep 2.4
  n+=1
  spl=$(bc <<< "scale=3;$SECONDS/$n")
  remain=$(bc <<< "scale=0; (($limit-$n)*$spl)/1")
  eta=$(date "+%r %a" --date="$remain seconds")
  output="$spl s per loop, $n loops"
  if [ $once -gt 0 ]; then
    now=$(date "+%r %a")
    echo "begin $now"
    once=0
  fi
  if [ $(($n%$rfreq)) -eq 0 ]; then
    output+=""
  fi
  if [ $limit -gt 0 ]; then
    rmin=$(bc <<< "scale=0;$remain/60") 
    rh=$(bc <<< "scale=0;$rmin/60")
    output+=" of $(($limit - $n)), eta $eta ($rh:$(($rmin%60)):$(($remain%60)))"
  fi
  echo $output
done
