#!/bin/bash

data_root='/data/seongha'

testsets=$1
gpu=$2
arch=$3
coop_weight=$4

python ./coop_pretrained_zsh.py --data=${data_root} --test_sets ${testsets} -a ${arch} --gpu ${gpu} --load ${coop_weight} 