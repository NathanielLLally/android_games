#!/bin/sh
#
#  resolution 1920x1200
#
#kitchen
bank="adb shell input swipe 700 625 700 625 127"
bank_menu="adb shell input swipe 700 580 700 580 973"
load_last="adb shell input swipe 1000 573 1000 573 215"
#chapel
bank="adb shell input swipe 612 641 612 641 144"
bank_menu="adb shell input swipe 604 663 604 663 972"
load_last="adb shell input swipe 967 660 967 660 246"
preset_13="adb shell input swipe 1680 197 1680 197 197"
preset_14="adb shell input swipe 1751 194 1751 194 175"
preset_15="adb shell input swipe 1489 233 1489 233 197"
action_2="adb shell input swipe 337 824 337 824 175"
action_4="adb shell input swipe 332 972 332 972 145"
action_6="adb shell input swipe 329 1063 329 1063 211"
incense="adb shell input swipe 1577 904 1577 904 232"
make="adb shell input swipe 1617 1062 1617 1062 91"

pre_logs=$preset_13
btn_logs=$action_6
pre_bare=$preset_14
btn_bare=$action_2
pre_ashed=$preset_15
btn_ashed=$action_4


declare -i limit=0 n=0 l=0 once=1 rfreq=3
declare -i unit_input=351
declare -i logs=0 yi=0 iyi=0 li=0
liph=0
eta=""
if [ -n "$1" ]; then logs=$1; fi
if [ -n "$2" ]; then yi=$2; liph=960; fi
if [ -n "$3" ]; then iyi=$3; fi

function stats() {
    ei=$(($logs/2 + $yi + $iyi))

    remain=$(bc <<< "scale=3; $ei/$liph*3600")
    remain=$(bc <<< "scale=0; $remain/1")
    eta=$(date "+%r %a" --date="$remain seconds")
    now=$(date "+%r %a")
    echo "began $then now $now logs $logs bare incense $yi ashed incense $iyi incense $li eta $eta"
}

then=$(date "+%r %a")
stats
while [ $(($logs + $yi + $iyi)) -gt 0 ]; do
  if [ $iyi -gt 0 ]; then
    $($bank)
    sleep 2
    $($pre_ashed)
    sleep 2
    $($btn_ashed)
    sleep 2
    $($make)
    sleep 47
      n=1
      iyi=$iyi-27
      li+=27
      stats
    while  [ $iyi -gt 0 ]; do
      $($bank_menu)
      sleep 3
      $($load_last)
      sleep 2
      $($btn_ashed)
      sleep 2
      $($make)
      sleep 47.5
      n+=1
      iyi=$iyi-27
      li+=27
      stats
    done
    sleep 3
    iyi=0
    spli=$(bc <<< "scale=3;$SECONDS/$li")
    liph=$(bc <<< "scale=3;60/$spli*60")
    ei=$(($logs/2 + $yi))
    remain=$(bc <<< "scale=3; $ei/$liph*3600")
    remain=$(bc <<< "scale=0; $remain/1")
    eta=$(date "+%r %a" --date="$remain seconds")
    echo "s/li $spli li/hr $liph eta $eta"
  fi

  if [ $iyi -lt $((13 * 27)) ]; then
    if [ $yi -lt $((13 * 27)) ]; then
      $($bank)
      sleep 2
      $($pre_logs)
      sleep 2
      $($btn_logs)
      sleep 2
      if [ $once -gt 0 ]; then
        once=0
        $($incense
        sleep 1)
      fi
      $($make)
      sleep 23
      yi+=14
      logs=$logs-28
      stats
      while [ $yi -lt $((13 * 27)) ]; do
        $($bank_menu)
        sleep 2
        $($load_last)
        sleep 2
        $($btn_logs)
        sleep 2
        $($make)
        sleep 23
        yi+=14
        logs=$logs-28
        stats
      done
      sleep 3
    else
      $($bank)
      sleep 2
      $($pre_bare)
      sleep 2
      $($btn_bare)
      sleep 2
      $($make)
      sleep 14
      iyi+=13
      yi=$yi-13
      stats
      while [ $iyi -lt $((13 * 27)) ]; do
        $($bank_menu)
        sleep 2
        $($load_last)
        sleep 2
        $($btn_bare)
        sleep 2
        $($make)
        sleep 14.5
        iyi+=13
        yi=$yi-13
        stats
      done
      sleep 3
    fi
  fi

  #spl=$(bc <<< "scale=3;$SECONDS/$l")
  #remain=$(bc <<< "scale=0; (($limit-$l)*$spl)/1")
  #eta=$(date "+%r %a" --date="$remain seconds")
  #output="$spl s per loop, $l loops"
  #if [ $limit -gt 0 ]; then
  #  rmin=$(bc <<< "scale=3;$remain/60") 
  #  rh=$(bc <<< "scale=3;$rmin/60")
  #  material=$((($limit-$l-1)*$unit_input))
  #  output+="  $(($limit - $l)), eta $eta ($rh hours) $material"
  #fi
  #echo $output
done
now=$(date "+%r %a")
echo "begin $then end $now logs $logs bare incense $yi ashed incense $iyi incense $li"
