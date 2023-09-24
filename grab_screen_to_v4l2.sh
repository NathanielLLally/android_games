#!/bin/sh


ffmpeg -f x11grab -framerate 25 -video_size 1280x720 -i :0.0+0,0 -f v4l2 /dev/video0
