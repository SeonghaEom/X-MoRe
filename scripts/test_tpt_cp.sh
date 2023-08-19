#!/bin/bash

data_root='/data/seongha'
testsets=$1
# arch=RN50
arch=ViT-B/16
bs=64
ctx_init=a_photo_of_a
retrieve_K=$2
gpu=$3

python ./tpt_classification.py ${data_root} --test_sets ${testsets} \
-a ${arch} -b ${bs} --gpu ${gpu} \
--ours --ctx_init ${ctx_init} --loss entropy --retrieve_K ${retrieve_K}
