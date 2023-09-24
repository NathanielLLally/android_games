#!/bin/sh
FILE="$1"
SIZE="$2"
if [ -z "$SIZE" ]; then
  SIZE="960x540"
fi

if [ -n "$FILE" ]; then
  ffmpeg -f v4l2 -framerate 30 -video_size $SIZE -i /dev/video0 $FILE
fi
