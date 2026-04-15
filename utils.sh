#!/bin/sh


#declare -A pinsA
getPIN() {
  s=$1
  echo "${pinsA[$s]}"
}

setPIN() {
  s=$1
  if [ -n "$(getDevices | grep $s)" ]; then
    pinsA[$s]="$2 $3 $4 $5 $6 $7 $8 $9"
  fi
}

getDevices() {
  echo $(adb devices | grep device$ | cut -f1)
}

getWlanIP() {
  serial=$1
  if [ -z "$serial" ]; then
	  echo "getWlanIP: error no device "
  fi
  ip=$(adb -s $serial shell "ip addr show wlan0 | grep -e wlan0$ | cut -d\" \" -f 6 | cut -d/ -f 1")
  echo $ip
}

resetWifi() {
  serial=$1
  if [ -z "$serial" ]; then
	  echo "resetWifi: error no device "
  fi
  hasColon=$(echo $serial | grep \:)
  if [ -n "$hasColon" ]; then
	  echo "$serial is over wifi, returning without reset"
	  return
  fi
  echo "disabling wifi"
  status=$(adb -s $serial shell "svc wifi disable")
  echo $status
  echo "enabling wifi"
  status=$(adb -s $serial shell "svc wifi enable")
  sleep 10
  echo $status
}

adbtcpip() {
  serial=$1
  if [ -z "$serial" ]; then
	  echo "adbtcpip: error no device "
  fi
  hasColon=$(echo $serial | grep \:)
  if [ -n "$hasColon" ]; then
	  echo "adbtcpip: device [$serial] is over wifi, returning"
	  return
  fi

  ip=$(getWlanIP $serial)
  status=$(adb -s $serial tcpip 5555)
  echo $status
  sleep 2;
  status=$(adb connect $ip:5555)
  echo $status
  sleep 2;
}

forwardPort() {
  serial=$1
  if [ -z "$serial" ]; then
	  echo "forwardPort: error no device "
  fi
  ip=$(getWlanIP $serial)
  status=$(adb -s $ip:5555 forward tcp:5555 tcp:5555)
  status=$(netstat -lvetpn | grep 127.0.0.1:5555)
  echo $status
}


# Returns the power state of the screen 1 = on, 0 = off
getDisplayState() {
	CMD="adb -s $1 shell dumpsys power | grep mScreenOn= | grep -oE '(true|false)')"
	echo $CMD
	state=$(adb -s $1 shell dumpsys power | grep mScreenOn= | grep -oE '(true|false)')

	# If we didn't get anything it might be a pre-lollipop device
	if [ "$state" = "" ]; then
		state=$(adb -s $1 shell dumpsys power | grep 'Display Power' | grep -oE '(ON|OFF)')
	fi

	if [ "$state" = "ON" ] || [ "$state" = "true" ]; then
		return 1;
	else
		return 0;
	fi
}

#  switchDisplayState on|off
#
switchDisplayState() {

      devices=$(getDevices)
      echo $devices
      if [ -n "$2" ]; then
	      if [ -n "$(echo $devices | grep $2)" ]; then
          devices=$2
        else
          echo "Turning $1 screen on all connected devices..."
        fi
      fi
  if [ "$1" = "on" ]; then
      for device in $devices
        do
          echo -n "Found device: $device ... "

            getDisplayState $device
            state=$?

# If the display is off, turn it on and unlock
            if [ $state -eq 0 ]; then
              echo "display was off, turning on"

# press power on
                adb -s $device shell input keyevent 26


                if [ -n "$(getPIN $device)" ]; then
			echo "entering pin"
			echo $(getPIN $device)
# press menu
                adb -s $device shell input keyevent 82

# enter pin
                adb -s $device shell input keyevent xxx
                adb -s $device shell input keyevent xxx
                adb -s $device shell input keyevent xxx
                adb -s $device shell input keyevent xxx
                fi
# press enter
#                adb -s $device shell input keyevent 66
            else
              echo "display was on, pressing home button to keep alive"
                adb -s $device shell input keyevent 3
                fi

                done

                fi

                if [ "$1" = "off" ]; then

                    for device in $devices
                      do
                        echo -n "Found device: $device ... "

                          getDisplayState $device
                          state=$?

# If the display is on, turn it off
                          if [ $state -eq 1 ]; then
                            echo "display was on, turning off"
                              adb -s $device shell input keyevent 26
                          else
                            echo "display was off"
                              fi

                              done

                              return 1;
                      fi
}
