#!/bin/sh
#
#  resolution 2560x1600
furnace="adb shell input swipe 1256 61 1256 61 107"
begin="adb shell input swipe 1993 1472 1993 1472 131"
anvil="adb shell input swipe 1603 737 1603 737 115"

declare -i limit=28 n=0 once=1 rfreq=3
next="dagger"
if [ -n "$1" ]; then limit=$1; fi
if [ -n "$2" ]; then next=$2; fi

select_item()
{
  item=$1
  shift;
  echo "selecting $item"
  if [ "$item" == "dagger" ]; then
    adb shell input swipe 294 451 294 451 106
    sleep 1.5
    adb shell input swipe 926 855 926 855 99
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
  for i in $(seq 1 $n); do
    echo "$i sleep" 
    sleep 32
    echo "$i heat"
    $($furnace)
    sleep 2
    $($anvil)
  done
}

doitem()
{
  item=$1
  if [ "$item" == "dagger" ];then
    progress=1200
    rn=2
    r=6
  elif [ "$item" == "plus1" ];then
    progress=960
    rn=1
    r=30
  elif [ "$item" == "plus2" ];then
    progress=1200
    rn=2
    r=6
  elif [ "$item" == "plus3" ];then
    progress=1440
    rn=2
    r=19
  fi
  echo "doing item $item"
  $($furnace)
  sleep 2
  select_item $item
  echo "beginning "
  $($begin)
  sleep 2
  $($anvil)
  echo "progress $progress"
  reheat $rn
  echo "sleep $r"
  sleep $r
  $($furnace)
  sleep 1
}

while [ \( $(($limit-$n)) -gt 0 \) -o \( $limit -eq 0 \) ]; do

  if [ "$next" == "dagger" ]; then
     doitem $next 
     next="plus1"
  fi
  if [ "$next" == "plus1" ]; then
     doitem $next 
     next="plus2"
  fi
  if [ "$next" == "plus2" ]; then
     doitem $next 
     next="plus3"
  fi
  if [ "$next" == "plus3" ]; then
     doitem $next 
     next="dagger"
  fi

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
    rmin=$(bc <<< "scale=3;$remain/60") 
    rh=$(bc <<< "scale=3;$rmin/60")
    output+=" of $(($limit - $n)), eta $eta ($rh hours)"
  fi
  echo $output
done
