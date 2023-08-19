#!/bin/bash
seed=0
dataset=$1
gpu=$2
arch=$3
coop_weight=$4


python ensemble_with_entropy.py --test_sets=${dataset} --data='/data/seongha' --n_ctx=4 --ctx_init='a_photo_of_a' --gpu=${gpu} --arch=${arch} --retrieve_K=2 --seed=${seed} --load ${coop_weight}
python ensemble_with_entropy.py --test_sets=${dataset} --data='/data/seongha' --n_ctx=4 --ctx_init='a_photo_of_a' --gpu=${gpu} --arch=${arch} --retrieve_K=4 --seed=${seed} --load ${coop_weight}
python ensemble_with_entropy.py --test_sets=${dataset} --data='/data/seongha' --n_ctx=4 --ctx_init='a_photo_of_a' --gpu=${gpu} --arch=${arch} --retrieve_K=8 --seed=${seed} --load ${coop_weight}
python ensemble_with_entropy.py --test_sets=${dataset} --data='/data/seongha' --n_ctx=4 --ctx_init='a_photo_of_a' --gpu=${gpu} --arch=${arch} --retrieve_K=16 --seed=${seed} --load ${coop_weight}
python ensemble_with_entropy.py --test_sets=${dataset} --data='/data/seongha' --n_ctx=4 --ctx_init='a_photo_of_a' --gpu=${gpu} --arch=${arch} --retrieve_K=32 --seed=${seed} --load ${coop_weight}
python ensemble_with_entropy.py --test_sets=${dataset} --data='/data/seongha' --n_ctx=4 --ctx_init='a_photo_of_a' --gpu=${gpu} --arch=${arch} --retrieve_K=64 --seed=${seed} --load ${coop_weight}
