#!/usr/bin/env bash
folder=$1
ffmpeg -r 10 -f image2 -s 512x512 -i ${folder}/frame%06d.jpg -vcodec libx264 -crf 10 -pix_fmt yuv420p ${folder}/$(basename ${folder}).mp4