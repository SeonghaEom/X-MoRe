#!/bin/bash

data_root='/data/seongha'

arch=RN50
# arch='ViT-B/16'
# bs=64
ctx_init=a_photo_of_a
# retrieve_K=$2
gpu=$1

python ./threshold_with_entropy.py --data=${data_root} --arch ${arch} --gpu ${gpu} --ctx_init ${ctx_init} 
