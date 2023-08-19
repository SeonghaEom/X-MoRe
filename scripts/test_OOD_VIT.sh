#['Aircraft', 'Caltech101', 'Cars', 'eurosat', 'Flower102', 'Food101', 'Pets', 'SUN397', 'I', 'A', 'K', 'R', 'V']
gpu=$1
bash scripts/test_dataset_cp.sh I ${gpu} ViT-B/16 1
bash scripts/test_dataset_cp.sh A ${gpu} ViT-B/16 1
bash scripts/test_dataset_cp.sh K ${gpu} ViT-B/16 1
bash scripts/test_dataset_cp.sh V ${gpu} ViT-B/16 1
bash scripts/test_dataset_cp.sh R ${gpu} ViT-B/16 1

# #pretrained
coop_weight='pretrained/coop/vit_b16_ep50_16shots/nctx4_cscFalse_ctpend/seed2/prompt_learner/model.pth.tar-50'
# bash scripts/test_tpt_coop.sh I ${gpu} ViT-B/16 ${coop_weight}
# bash scripts/test_tpt_coop.sh V ${gpu} ViT-B/16 ${coop_weight}
# bash scripts/test_tpt_coop.sh R ${gpu} ViT-B/16 ${coop_weight}
# bash scripts/test_tpt_coop.sh A ${gpu} ViT-B/16 ${coop_weight}
# bash scripts/test_tpt_coop.sh K ${gpu} ViT-B/16 ${coop_weight}

# bash scripts/ablation.sh I ${gpu}
# bash scripts/ablation.sh V ${gpu}
# bash scripts/ablation.sh R ${gpu}
# bash scripts/ablation.sh A ${gpu}
# bash scripts/ablation.sh K ${gpu}

# bash scripts/test_coop_zsh.sh I ${gpu} ViT-B/16 ${coop_weight}
# bash scripts/test_coop_zsh.sh V ${gpu} ViT-B/16 ${coop_weight}
# bash scripts/test_coop_zsh.sh R ${gpu} ViT-B/16 ${coop_weight}
# bash scripts/test_coop_zsh.sh A ${gpu} ViT-B/16 ${coop_weight}
# bash scripts/test_coop_zsh.sh K ${gpu} ViT-B/16 ${coop_weight}

# #caption ensemble
# bash scripts/caption_ensemble.sh I ${gpu} ViT-B/16 ${coop_weight} 
# bash scripts/caption_ensemble.sh V ${gpu} ViT-B/16 ${coop_weight}
# bash scripts/caption_ensemble.sh A ${gpu} ViT-B/16 ${coop_weight}
# bash scripts/caption_ensemble.sh R ${gpu} ViT-B/16 ${coop_weight}
# bash scripts/caption_ensemble.sh K ${gpu} ViT-B/16 ${coop_weight}
