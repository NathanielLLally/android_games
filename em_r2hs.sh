#!/bin/sh
#
#  resolution 2560x1600
furnace="adb shell input swipe 1256 61 1256 61 107"
begin="adb shell input swipe 1993 1472 1993 1472 131"
anvil="adb shell input swipe 1693 737 1693 737 115"

select_item()
{
  item=$1
  shift;
  if [ "$item" == "2hs" ]; then
    adb shell input swipe 294 451 294 451 106
    sleep 1.5
    adb shell input swipe 937 1190 937 1190 61
    sleep 1.5
    adb shell input swipe 1724 340 1724 340 100
    sleep 1.5
  elif [ "$item" == "plus1" ]; then
    adb shell input swipe 1841 324 1841 324 92
    sleep 1.5
  elif [ "$item" == "plus2" ]; then
    adb shell input swipe 1956 325 1956 325 115
    sleep 1.5
  elif [ "$item" == "plus3" ]; then
    adb shell input swipe 2067 318 2067 318 115
    sleep 1.5
  fi
}

reheat()
{
  declare -i n=0
  n=$1
  shift;
  echo "reheat $n"
  for i in $(seq 0 $n); do
    echo $i
    sleep 32
    $($furnace)
    sleep 2
    $($anvil)
  done
}

doitem()
{
  item=$1
  if [ "$item" == "2hs" ];then
    n=3
    r=10
  elif [ "$item" == "plus1" ];then
    n=3
    r=10
  elif [ "$item" == "plus2" ];then
    n=4
    r=6
  elif [ "$item" == "plus3" ];then
    n=4
    r=2
  fi
  $($furnace)
  sleep 2
  select_item $item
  $($begin)
  sleep 2
  $($anvil)
  reheat $n
  sleep $r
  $($begin)
}

declare -i limit=0 n=0 once=1 rfreq=3
if [ -n "$1" ]; then limit=$1; fi
while [ \( $(($limit-$n)) -gt 0 \) -o \( $limit -eq 0 \) ]; do

  doitem 2hs
  doitem plus1
  doitem plus2
  doitem plus3

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
