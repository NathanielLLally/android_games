#!/bin/sh
#
#  resolution 2560x1600
#
declare -i limit=0 n=0 once=1 rfreq=3
if [ -n "$1" ]; then limit=$1; fi
while [ \( $(($limit-$n)) -gt 0 \) -o \( $limit -eq 0 \) ]; do
  adb shell input swipe 1255 80 1255 80 53
  sleep 3.5
  adb shell input swipe 928 1198 928 1198 51
  sleep 2.7
  adb shell input swipe 1710 331 1710 331 67
  sleep 2.2
  adb shell input swipe 1988 1474 1988 1474 99
  sleep 2.7
  adb shell input swipe 1690 733 1690 733 94
  sleep 21.9
  adb shell input swipe 1170 133 1170 133 51
  sleep 1.9
  adb shell input swipe 1690 731 1690 733 141
  sleep 21.5
  adb shell input swipe 1152 125 1152 125 59
  sleep 1.7
  adb shell input swipe 1705 725 1705 725 84
  sleep 26.5
  adb shell input swipe 1141 144 1141 144 85
  sleep 2.4
  adb shell input swipe 1694 738 1694 738 83
  sleep 22.9
  adb shell input swipe 1152 118 1152 118 63
  sleep 1.6
  adb shell input swipe 1705 748 1705 748 115
  sleep 23.8
  adb shell input swipe 1105 122 1105 122 80
  sleep 1.6
  adb shell input swipe 1710 764 1708 759 67
  sleep 23.7
  adb shell input swipe 2103 1342 2103 1342 125
  sleep 1.7
  adb shell input swipe 1198 190 1198 190 133
  sleep 1.5
  adb shell input swipe 1857 324 1857 324 83
  sleep 1.5
  adb shell input swipe 2010 1485 2010 1485 126
  sleep 2.5
  adb shell input swipe 1675 740 1675 740 93
  sleep 24.9
  adb shell input swipe 1128 220 1128 220 88
  sleep 2.0
  adb shell input swipe 1695 740 1695 740 139
  sleep 32.2
  adb shell input swipe 1068 109 1068 109 88
  sleep 2.8
  adb shell input swipe 1705 751 1705 751 118
  sleep 22.5
  adb shell input swipe 1113 133 1113 133 47
  sleep 1.7
  adb shell input swipe 1684 762 1684 762 91
  sleep 21.1
  adb shell input swipe 1139 146 1139 146 71
  sleep 1.6
  adb shell input swipe 1679 724 1679 725 107
  sleep 11.3
  adb shell input swipe 2110 1320 2110 1320 21
  sleep 1.5
  adb shell input swipe 1146 157 1146 157 61
  sleep 2.2
  adb shell input swipe 1971 329 1971 329 67
  sleep 1.6
  adb shell input swipe 1966 1472 1966 1472 75
  sleep 2.2
  adb shell input swipe 1714 774 1714 774 100
  sleep 22.9
  adb shell input swipe 1152 142 1152 142 69
  sleep 1.4
  adb shell input swipe 1690 740 1690 740 84
  sleep 21.6
  adb shell input swipe 1133 137 1133 137 52
  sleep 0.8
  adb shell input swipe 1697 751 1695 751 140
  sleep 24.1
  adb shell input swipe 1129 155 1129 161 93
  sleep 2.2
  adb shell input swipe 1672 777 1672 777 102
  sleep 22.3
  adb shell input swipe 1241 300 1233 309 94
  sleep 2.6
  adb shell input swipe 1677 775 1677 775 91
  sleep 24.2
  adb shell input swipe 1098 227 1098 227 52
  sleep 2.4
  adb shell input swipe 1721 774 1721 774 99
  sleep 24.0
  adb shell input swipe 2114 1309 2112 1309 77
  sleep 2.1
  adb shell input swipe 1204 198 1204 198 131
  sleep 2.8
  adb shell input swipe 2059 322 2059 322 131
  sleep 1.6
  adb shell input swipe 1996 1462 1996 1462 107
  sleep 3.3
  adb shell input swipe 1667 742 1667 742 118
  sleep 24.4
  adb shell input swipe 1152 188 1152 188 84
  sleep 1.8
  adb shell input swipe 1708 742 1708 742 83
  sleep 22.2
  adb shell input swipe 1154 164 1154 164 72
  sleep 1.7
  adb shell input swipe 1675 746 1675 746 91
  sleep 23.3
  adb shell input swipe 1124 127 1124 127 109
  sleep 1.8
  adb shell input swipe 1712 762 1712 762 107
  sleep 21.3
  adb shell input swipe 1146 177 1146 177 53
  sleep 1.8
  adb shell input swipe 1701 779 1701 779 75
  sleep 22.9
  adb shell input swipe 1168 162 1168 162 84
  sleep 1.8
  adb shell input swipe 1699 755 1699 755 147
  sleep 23.3
  adb shell input swipe 1118 168 1118 168 132
  sleep 1.5
  adb shell input swipe 1695 785 1695 785 123
  sleep 25.3
  adb shell input swipe 2114 1309 2112 1309 77
  sleep 2.1
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
