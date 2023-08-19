#['Aircraft', 'Caltech101', 'Cars', 'eurosat', 'Flower102', 'Food101', 'Pets', 'SUN397', 'I', 'A', 'K', 'R', 'V']
gpu=$1
bash scripts/test_dataset_cp.sh DTD ${gpu} RN50 2
bash scripts/test_dataset_cp.sh UCF101 ${gpu} RN50 2
bash scripts/test_dataset_cp.sh Aircraft ${gpu} RN50 2
bash scripts/test_dataset_cp.sh Caltech101 ${gpu} RN50 2
bash scripts/test_dataset_cp.sh Cars ${gpu} RN50 2
bash scripts/test_dataset_cp.sh eurosat ${gpu} RN50 2
bash scripts/test_dataset_cp.sh Flower102 ${gpu} RN50 2
bash scripts/test_dataset_cp.sh Food101 ${gpu} RN50 2
bash scripts/test_dataset_cp.sh Pets ${gpu} RN50 2
bash scripts/test_dataset_cp.sh SUN397 ${gpu} RN50 2

## pretrained
coop_weight='pretrained/coop/rn50_ep50_16shots/nctx4_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-50'
# bash scripts/test_tpt_coop.sh DTD ${gpu} RN50 ${coop_weight}
# bash scripts/test_tpt_coop.sh UCF101 ${gpu} RN50 ${coop_weight}
# bash scripts/test_tpt_coop.sh Aircraft ${gpu} RN50 ${coop_weight}
# bash scripts/test_tpt_coop.sh Caltech101 ${gpu} RN50 ${coop_weight}
# bash scripts/test_tpt_coop.sh Cars ${gpu} RN50 ${coop_weight}
# bash scripts/test_tpt_coop.sh eurosat ${gpu} RN50 ${coop_weight}
# bash scripts/test_tpt_coop.sh Flower102 ${gpu} RN50 ${coop_weight}
# bash scripts/test_tpt_coop.sh Food101 ${gpu} RN50 ${coop_weight}
# bash scripts/test_tpt_coop.sh Pets ${gpu} RN50 ${coop_weight}
# bash scripts/test_tpt_coop.sh SUN397 ${gpu} RN50 ${coop_weight}

#pretrained coop zsh
# bash scripts/test_coop_zsh.sh DTD ${gpu} RN50 ${coop_weight}
# bash scripts/test_coop_zsh.sh UCF101 ${gpu} RN50 ${coop_weight}
# bash scripts/test_coop_zsh.sh Aircraft ${gpu} RN50 ${coop_weight}
# bash scripts/test_coop_zsh.sh Caltech101 ${gpu} RN50 ${coop_weight}
# bash scripts/test_coop_zsh.sh Cars ${gpu} RN50 ${coop_weight}
# bash scripts/test_coop_zsh.sh eurosat ${gpu} RN50 ${coop_weight}
# bash scripts/test_coop_zsh.sh Flower102 ${gpu} RN50 ${coop_weight}
# bash scripts/test_coop_zsh.sh Food101 ${gpu} RN50 ${coop_weight}
# bash scripts/test_coop_zsh.sh Pets ${gpu} RN50 ${coop_weight}
# bash scripts/test_coop_zsh.sh SUN397 ${gpu} RN50 ${coop_weight}

# caption ensemble
# bash scripts/caption_ensemble.sh DTD ${gpu} RN50 ${coop_weight}
# bash scripts/caption_ensemble.sh UCF101 ${gpu} RN50 ${coop_weight}
# bash scripts/caption_ensemble.sh Aircraft ${gpu} RN50 ${coop_weight}
# bash scripts/caption_ensemble.sh Caltech101 ${gpu} RN50 ${coop_weight}
# bash scripts/caption_ensemble.sh Cars ${gpu} RN50 ${coop_weight}
# bash scripts/caption_ensemble.sh eurosat ${gpu} RN50 ${coop_weight}
# bash scripts/caption_ensemble.sh Flower102 ${gpu} RN50 ${coop_weight}
# bash scripts/caption_ensemble.sh Food101 ${gpu} RN50 ${coop_weight}
# bash scripts/caption_ensemble.sh Pets ${gpu} RN50 ${coop_weight}
# bash scripts/caption_ensemble.sh SUN397 ${gpu} RN50 ${coop_weight}