#!/bin/bash
if [ "$#" -eq  "0" ];
    then 
        my_path='video.mp4'
    else 
        my_path="$1"
fi
mkdir -p "cutting_video"
ffmpeg -i $my_path -r 4 "cutting_video/out-%03d.jpg"
