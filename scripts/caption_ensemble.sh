#!/bin/bash

data_root='/data/seongha'

testsets=$1
gpu=$2
arch=$3
coop_weight=$4

python ./caption_ensemble.py --data=${data_root} --test_sets ${testsets} -a ${arch} --gpu ${gpu}
python ./caption_ensemble.py --data=${data_root} --test_sets ${testsets} -a ${arch} --gpu ${gpu} --load ${coop_weight} 