#!/bin/sh
#
#  resolution 1920x1200
#

bank_last_pre="adb shell input swipe 553 695 553 688 1025; sleep 1; adb shell input swipe 952 716 952 716 100"
bank_last="adb shell input swipe 610 674 610 674 1042; sleep 1.0; adb shell input swipe 992 706 992 706 157"

well="adb shell input swipe 929 623 929 623 200"
mix="adb shell input swipe 1496 1062 1496 1062 200"
bank="adb shell input swipe 614 661 614 661 200"
preset_1_10="adb shell input swipe 1489 194 1489 194 200"
preset_2_11="adb shell input swipe 1556 198 1556 198 200"
preset_3_12="adb shell input swipe 1607 195 1607 195 200"
preset_well=$preset_1_10
preset_pot=$preset_2_11
preset_pot2=$preset_3_12
button_5="adb shell input swipe 469 975 469 975 200"
button_7="adb shell input swipe 466 1089 466 1089 200"
extend_yes="adb shell input swipe 1876 564 1876 564 200"
wellfreq=1800
welltime=0
# 6, enter
enter_amount="adb shell input keyevent 13; sleep 0.2; adb shell input keyevent 66"


source ./phone_vars.sh

declare -i limit=0 n=0 once=1 rfreq=3 mixpre=1 wells=0 spli=34 welltime=0
declare -i unit_input=27 beginsn=0 sn=0 en=0 batch=270 batchsn=1 sleept=0 congealed=0
if [ -n "$1" ]; then sn=$1; fi
if [ -n "$2" ]; then welltime=$(($2 * 60)); fi
if [ -n "$3" ]; then congealed=$3; fi
beginsn=$sn

welltimef=$(date "+%r %a" --date="$welltime seconds")
echo "next well at $welltimef"
while [ \( $(($limit-$n)) -gt 0 \) -o \( $limit -eq 0 \) ]; do
  if [ $SECONDS -ge $welltime  ]; then
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
    welltimef=$(date "+%r %a" --date="$welltime seconds")
    echo "next well at $welltimef"
    wells+=6
  fi
  if [ \( $sn -ge $batch \) -o \( $batchsn -eq 0 \) ];then
    if [ $batchsn -eq 1 ]; then
      mixpre=1
    fi
    batchsn=0
   
    if [ $sn -ge 27 ]; then
      if [ $mixpre -eq 1 ]; then

        $($bank)
        sleep 2
        $($preset_pot2)
        mixpre=0
        sleept=33
      else
        eval "$bank_last"
        sleept=30
      fi
      sleep 2
      $($well)
      sleep 2.5
      $($mix)
      sleep $sleept
      sn=$sn-27
      en+=31
    else
      batchsn=1
      mixpre=1
    fi
  else

    if [ $mixpre -eq 1 ]; then
      $($bank)
      sleep 2
      $($preset_pot)
      mixpre=0
      sleept=33
    else
      eval "$bank_last"
      sleept=30
    fi
    sleep 2
    $($well)
    sleep 2.5
    $($mix)
    sleep $sleept
    sn+=31
    congealed=$(bc <<< "$congealed-(27*5/1.111)")
  fi


  n+=1
  spl=$(bc <<< "scale=3;$SECONDS/$n")
  spli=$(bc <<< "scale=0;($spl/1)+1")
  remain=$(bc <<< "scale=0; (($limit-$n)*$spl)/1")
  eta=$(date "+%r %a" --date="$remain seconds")
  output="$spl s/l, n $n,"
  if [ $once -gt 0 ]; then
    now=$(date "+%r %a")
    echo "begin $now"
    once=0
  fi
  outsph=$(bc <<< "scale=3; ($unit_input*1.15)/$spl * 3600")
  totalin=$(bc <<< "scale=3; $congealed/5*1.111")
  totalsn=$(bc <<< "scale=3; (1.15*$totalin-($sn-$beginsn))")
  totalen=$(bc <<< "scale=3; (1.15*$totalsn-$en)")
  miasma=$(bc <<< " scale=0; $totalen/1.111+1")
  remain=$(bc <<< "scale=3; (($totalsn+$totalen)/$outsph)*3600")
  remain=$(bc <<< "scale=0; $remain/1")
  eta=$(date "+%r %a" --date="$remain seconds")
  output+=" $outsph pots/hour, sn $sn en $en, wells used $wells ($SECONDS s), total sn $totalsn en $totalen ($miasma gm)" 
  rmin=$(bc <<< "scale=3;$remain/60") 
  rh=$(bc <<< "scale=0;$rmin/60")
  rmi=$(bc <<< "scale=0;$rmin%60")
  output+=", eta $eta ($rh hours $rmi mins), well $welltimef"
  echo $output
done
