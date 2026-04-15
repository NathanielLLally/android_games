#!/bin/sh
#


while [ 1 ]; do

    adb shell input swipe 467 639 467 639 180
    sleep 1
      adb shell input swipe 1062 229 1062 229 140
      sleep 5
done
