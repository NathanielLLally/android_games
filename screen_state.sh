#!/bin/sh

# Returns the power state of the screen 1 = on, 0 = off
getDisplayState() {
	state=$(adb -s $1 shell dumpsys power | grep mScreenOn= | grep -oE '(true|false)')

	# If we didn't get anything it might be a pre-lollipop device
	if [ "$state" = "" ]; then
#		state=$(adb -s $1 shell dumpsys power | grep 'Display Power' | grep -oE '(ON|OFF)')
		state=$(adb -s $1 shell dumpsys power | grep 'mWakefulness' | grep -oE '(Awake|Asleep)')
	fi

	if [ "$state" = "Awake" ]; then
		return 1;
	else
		return 0;
	fi
}

if [ $# -eq 0 ]; then
	echo "Usage: $0 [on|off]"
	for device in `adb devices | grep device$ | cut -f1`
	do
		echo -n "Found device: $device ... "
		getDisplayState $device
		state=$?
	done
	exit 1;
fi

if [ "$1" = "on" ]; then
	echo "Turning on screen on all connected devices..."

	for device in `adb devices | grep device$ | cut -f1`
	do
		echo -n "Found device: $device ... "

		getDisplayState $device
		state=$?

		# If the display is off, turn it on and unlock
		if [ $state -eq 0 ]; then
			echo "display was off, turning on"

			# press power on
			adb -s $device shell input keyevent 26
			
			# press menu
			adb -s $device shell input keyevent 82

			# enter pin
			adb -s $device shell input keyevent xxx
			adb -s $device shell input keyevent xxx
			adb -s $device shell input keyevent xxx
			adb -s $device shell input keyevent xxx

			# press enter
			adb -s $device shell input keyevent 66
		else
			echo "display was on, pressing home button to keep alive"
			adb -s $device shell input keyevent 3
		fi
		
	done

	exit 0;
fi

if [ "$1" = "off" ]; then
	echo "Turning off screen on all connected devices..."

	for device in `adb devices | grep device$ | cut -f1`
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

	exit 0;
fi
