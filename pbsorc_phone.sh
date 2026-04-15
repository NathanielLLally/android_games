#!/bin/sh
#
#  resolution 1920x1200
#

bank_last_pre="adb shell input swipe 553 695 553 688 1025; sleep 1; adb shell input swipe 952 716 952 716 100"
bank_last="adb shell input swipe 610 674 610 674 1042; sleep 1.0; adb shell input swipe 992 706 992 706 157"

well="adb shell input swipe 929 623 929 623 200"
mix="adb shell input swipe 1496 1062 1496 1062 200"
bank="adb shell input swipe 614 661 614 661 200"
button_5="adb shell input swipe 469 975 469 975 200"
button_7="adb shell input swipe 466 1089 466 1089 200"
extend_yes="adb shell input swipe 1876 564 1876 564 200"

wellfreq=1800
welltime=0
# 6, enter
enter_amount="adb shell input keyevent 13; sleep 0.2; adb shell input keyevent 66"

source ./phone_vars.sh

preset_pot=$preset_6
preset_pot2=$preset_7


declare -i limit=0 n=0 once=1 rfreq=3 mixpre=1 wells=0 spli=34 welltime=0 usewells=1
declare -i unit_input=14 beginsn=0 pot1=0 pot2=0 batch=140 batchsn=1 firstpot1=0 firstpot2=0
sleept=0
if [ -n "$1" ]; then 
  pot1=$1; 
limit=$(($1/$unit_input+1))
  fi
if [ -n "$2" ]; then welltime=$(($2 * 60)); fi
if [ -n "$3" ]; then usewells=$(($3)); fi
beginsn=$pot1

welltimef=$(date "+%r %a" --date="$welltime seconds")
echo "next well at $welltimef"
while [ \( $(($limit-$n)) -gt 0 \) -o \( $limit -eq 0 \) ]; do
  if [ $SECONDS -ge $welltime  ]; then
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
    welltimes=$(($welltime-$SECONDS))
    welltimef=$(date "+%r %a" --date="$welltimes seconds")
    echo "next well at $welltimef"
    wells+=6
  fi
  if [ \( $pot1 -ge $batch \) -o \( $batchsn -eq 0 \) ];then
    if [ $batchsn -eq 1 ]; then
      mixpre=1
    fi
    batchsn=0
   
    if [ $pot1 -ge 12 ]; then
      if [ $mixpre -eq 1 ]; then
        if [ $firstpot2 -ne 0 ]; then
          sleep 2
        fi

        $($bank)
        sleep 2
        $($preset_pot2)
        mixpre=0
        sleept=12
        firstpot2=1
      else
        eval "$bank_last"
        sleept=10.5
      fi
      sleep 2
      $($well)
      sleep 2
      $($mix)
      echo "sleep $sleept"
      sleep $sleept
      pot1=$pot1-12
      pot2+=13
    else
      batchsn=1
      mixpre=1
    fi
  else

    if [ $mixpre -eq 1 ]; then
        if [ $firstpot1 -ne 0 ]; then
          sleep 1.5
        fi
      $($bank)
      sleep 2
      $($preset_pot)
      mixpre=0
      sleept=14
      firstpot1=1
    else
      eval "$bank_last"
      sleept=12
    fi
    sleep 2
    $($well)
    sleep 2
    $($mix)
    echo "sleep $sleept"
    sleep $sleept
    pot1+=15
  fi


  n+=1
  spl=$(bc <<< "scale=3;$SECONDS/$n")
  spli=$(bc <<< "scale=0;($spl/1)+1")
  remain=$(bc <<< "scale=0; (($limit-$n)*$spl)/1")
  eta=$(date "+%r %a" --date="$remain seconds")
  output="$spl s/l, n $n,"
  outsph=$(bc <<< "scale=3; ($unit_input*1.15)/$spl * 3600")
  #totalin=$(bc <<< "scale=3; $congealed/5*1.111")
  #totalpot1=$(bc <<< "scale=3; (1.15*$totalin-($pot1-$beginsn))")
  #totalpot2=$(bc <<< "scale=3; (1.15*$totalpot1-$pot2)")
  #miasma=$(bc <<< " scale=0; $totalpot2/1.111+1")
  #remain=$(bc <<< "scale=3; (($totalpot1+$totalpot2)/$outsph)*3600")
  #remain=$(bc <<< "scale=0; $remain/1")
  eta=$(date "+%r %a" --date="$remain seconds")
  output+=" $outsph pots/hour, sup rc $pot1 pb sorc $pot2, wells $wells"

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
