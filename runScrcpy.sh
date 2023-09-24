#!/bin/sh
scrcpy -m960 --v4l2-sink=/dev/video0 --show-touches --stay-awake; while [ 1 ]; do echo "starting "; scrcpy -m960 --v4l2-sink=/dev/video0 --no-display --show-touches --stay-awake; sleep 5; done
