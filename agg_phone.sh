#!/bin/sh
#
#  resolution 2400x1080
#
bank_last='adb shell input swipe 888 583 888 583 1058; sleep 0.7; adb shell input swipe 1207 596 1207 596 116'
well='adb shell input swipe 1260 587 1260 587 180'
mix='adb shell input swipe 1891 978 1891 978 100'

source ./phone_vars.sh
preset_pot=$preset_13

declare -i limit=0 n=0 once=1 rfreq=3 spli=30
declare -i unit_input=14 mixpre=1 usewells=1 welltime=0
if [ -n "$1" ]; then limit=$(($1/$unit_input+1)); fi
if [ -n "$2" ]; then welltime=$(($2 * 60)); fi
if [ -n "$3" ]; then usewells=$(($3)); fi
welltimef=$(date "+%r %a" --date="$welltime seconds")
while [ \( $(($limit-$n)) -gt 0 \) -o \( $limit -eq 0 \) ]; do
  if [ $SECONDS -ge $welltime ]; then
    if [ $usewells -ne 1 ]; then
      echo "no wells, exiting"
      exit
    fi
    if [ $n -gt 1 ]; then
      sleep 4
    fi
    $($bank)
    sleep 3
    $($preset_well)
    sleep 3
    $($button_7)
    sleep 2
    $($extend_yes)
    sleep 2
    eval "$enter_amount"
    sleep 1
    #$($extend_yes)
    #sleep 1.5
    mixpre=1
    welltime=$(($SECONDS + $wellfreq - 18 - $spli))
    welltimen=$(($welltime - $SECONDS))
    welltimef=$(date "+%r %a" --date="$welltimen seconds")
    echo "next well at $welltimef"
    wells+=6
  fi

  if [ $mixpre -eq 1 ]; then
        $($bank)
        sleep 2
        $($preset_pot)
        mixpre=0
        sleept=16
  else
    eval "$bank_last"
    sleept=13
  fi
  sleep 1
  $($well)
  sleep 1.25
  $($mix)
  sleep $sleept
  n+=1
  spl=$(bc <<< "scale=3;$SECONDS/$n")
  spli=$(bc <<< "scale=0;($spl/1)+1")
  remain=$(bc <<< "scale=0; (($limit-$n)*$spl)/1")
  eta=$(date "+%r %a" --date="$remain seconds")
  outsph=$(bc <<< "scale=3; ($unit_input*1.15)/$spl * 3600")
  output="$spl s/l, n $n, used $(($n * $unit_input)) primary, $outsph pots/hr"
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
    rh=$(bc <<< "scale=0;$rmin/60")
    rmi=$(bc <<< "scale=0;($rmin%60)/1")
    material=$((($limit-$n-1)*$unit_input))
    output+=" , eta $eta ($rh hours $rmi mins) $material, well $welltimef"
    if [ $rh -eq 0 ]; then
      if [ $rmi -lt 25 ]; then
        usewells=0
      fi
    fi
  fi
  echo $output
done
