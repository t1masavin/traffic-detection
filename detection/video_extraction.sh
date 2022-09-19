#!/bin/bash
mkdir -p "cutting_video"
ffmpeg -i "video.mp4" -r 4 "cutting_video/out-%03d.jpg"