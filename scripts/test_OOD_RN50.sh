#['Aircraft', 'Caltech101', 'Cars', 'eurosat', 'Flower102', 'Food101', 'Pets', 'SUN397', 'I', 'A', 'K', 'R', 'V']
gpu=$1
bash scripts/test_dataset_cp.sh I ${gpu} RN50 1
bash scripts/test_dataset_cp.sh A ${gpu} RN50 1
bash scripts/test_dataset_cp.sh K ${gpu} RN50 1
bash scripts/test_dataset_cp.sh V ${gpu} RN50 1
bash scripts/test_dataset_cp.sh R ${gpu} RN50 1

## pretrained
coop_weight='pretrained/coop/rn50_ep50_16shots/nctx4_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-50'
# bash scripts/test_tpt_coop.sh I ${gpu} RN50 ${coop_weight} 
# bash scripts/test_tpt_coop.sh V ${gpu} RN50 ${coop_weight}
# bash scripts/test_tpt_coop.sh A ${gpu} RN50 ${coop_weight}
# bash scripts/test_tpt_coop.sh R ${gpu} RN50 ${coop_weight}
# bash scripts/test_tpt_coop.sh K ${gpu} RN50 ${coop_weight}

# bash scripts/test_coop_zsh.sh I ${gpu} RN50 ${coop_weight} 
# bash scripts/test_coop_zsh.sh V ${gpu} RN50 ${coop_weight}
# bash scripts/test_coop_zsh.sh A ${gpu} RN50 ${coop_weight}
# bash scripts/test_coop_zsh.sh R ${gpu} RN50 ${coop_weight}
# bash scripts/test_coop_zsh.sh K ${gpu} RN50 ${coop_weight}

# #caption ensemble
# bash scripts/caption_ensemble.sh I ${gpu} RN50 ${coop_weight} 
# bash scripts/caption_ensemble.sh V ${gpu} RN50 ${coop_weight}
# bash scripts/caption_ensemble.sh A ${gpu} RN50 ${coop_weight}
# bash scripts/caption_ensemble.sh R ${gpu} RN50 ${coop_weight}
# bash scripts/caption_ensemble.sh K ${gpu} RN50 ${coop_weight}