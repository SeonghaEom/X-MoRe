#!/bin/bash

data_root='/data/seongha'
cocoop_weight='/home/seongha/CoOp/output/imagenet/CoCoOp/vit_b16_c4_ep10_batch1_ctxv1_16shots/seed0/prompt_learner/model.pth.tar-10'
testsets=$1
# arch=RN50
arch=ViT-B/16
bs=64
gpu=$2

python ./ensemble_with_entropy.py --data=${data_root} --test_sets ${testsets} \
-a ${arch} -b ${bs} --gpu ${gpu} \
--cocoop --load ${cocoop_weight}