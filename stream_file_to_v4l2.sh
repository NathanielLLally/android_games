#!/bin/sh
FILE="$1"

if [ -f "$FILE" ]; then
	ffmpeg -re -stream_loop -1 -i $FILE -f v4l2 /dev/video0
fi
