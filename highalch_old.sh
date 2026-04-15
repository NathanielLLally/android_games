#!/bin/sh
#

function stop()
{
  echo
  echo "$(($SECONDS/60)) min $(($SECONDS%60)) sec"
  exit 1
}
trap 'stop' SIGINT 

while [ 1 ]; do
adb shell input swipe 905 194 905 194 100
sleep 2
adb shell input swipe 102 297 102 297 100
sleep 2
adb shell input swipe 1841 63 1841 63 100
sleep 2
adb shell input swipe 328 828 328 828 100
sleep 1.5
adb shell input swipe 1562 392 1562 392 100
sleep 1.5
adb shell input swipe 329 829 329 829 100
sleep 1.5
adb shell input swipe 1640 391 1640 391 100
sleep 1.5
adb shell input swipe 327 833 327 833 100
sleep 1.5
adb shell input swipe 1721 398 1721 398 100
sleep 1.5
adb shell input swipe 326 834 326 834 100
sleep 1.5
adb shell input swipe 1467 474 1467 474 100
sleep 1.5
adb shell input swipe 327 832 327 832 100
sleep 1.5
adb shell input swipe 1540 475 1540 475 100
sleep 1.5
adb shell input swipe 327 828 327 828 100
sleep 1.5
adb shell input swipe 1633 478 1633 478 100
sleep 1.5
adb shell input swipe 324 826 324 826 100
sleep 1.5
adb shell input swipe 1715 474 1715 474 100
sleep 1.5
adb shell input swipe 329 833 329 833 100
sleep 1.5
adb shell input swipe 1474 556 1474 556 100
sleep 1.5
adb shell input swipe 333 832 333 832 100
sleep 1.5
adb shell input swipe 1543 565 1543 565 100
sleep 1.5
adb shell input swipe 327 822 327 822 100
sleep 1.5
adb shell input swipe 1641 557 1641 557 100
sleep 1.5
adb shell input swipe 329 829 329 829 100
sleep 1.5
adb shell input swipe 1712 558 1712 558 100
sleep 1.5
adb shell input swipe 335 834 335 834 100
sleep 1.5
adb shell input swipe 1461 645 1461 645 100
sleep 1.5
adb shell input swipe 332 822 332 822 100
sleep 1.5
adb shell input swipe 1557 640 1557 640 100
sleep 1.5
adb shell input swipe 325 824 325 824 100
sleep 1.5
adb shell input swipe 1638 639 1638 639 100
sleep 1.5
adb shell input swipe 326 829 326 829 100
sleep 1.5
adb shell input swipe 1722 643 1722 643 100
sleep 1.5
adb shell input swipe 329 824 329 824 100
sleep 1.5
adb shell input swipe 1460 720 1460 720 100
sleep 1.5
adb shell input swipe 329 826 329 826 100
sleep 1.5
adb shell input swipe 1542 711 1542 711 100
sleep 1.5
adb shell input swipe 337 818 337 818 100
sleep 1.5
adb shell input swipe 1625 726 1625 726 100
sleep 1.5
adb shell input swipe 327 824 327 824 100
sleep 1.5
adb shell input swipe 1719 722 1719 722 100
sleep 1.5
adb shell input swipe 343 828 343 828 100
sleep 1.5
adb shell input swipe 1489 805 1489 805 100
sleep 1.5
adb shell input swipe 335 824 335 824 100
sleep 1.5
adb shell input swipe 1564 801 1564 801 100
sleep 1.5
adb shell input swipe 336 819 336 819 100
sleep 1.5
adb shell input swipe 1633 802 1633 802 100
sleep 1.5
adb shell input swipe 336 817 336 817 100
sleep 1.5
adb shell input swipe 1723 795 1723 795 100
sleep 1.5
adb shell input swipe 333 825 333 825 100
sleep 1.5
adb shell input swipe 1464 888 1464 888 100
sleep 1.5
adb shell input swipe 339 829 339 829 100
sleep 1.5
adb shell input swipe 1550 879 1550 879 100
sleep 1.5
adb shell input swipe 335 830 335 830 100
sleep 1.5
adb shell input swipe 1639 877 1639 877 100
sleep 1.5
adb shell input swipe 329 819 329 819 100
sleep 1.5
adb shell input swipe 1715 886 1715 886 100
sleep 1.5

done


