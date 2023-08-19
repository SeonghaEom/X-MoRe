#['Aircraft', 'Caltech101', 'Cars', 'eurosat', 'Flower102', 'Food101', 'Pets', 'SUN397', 'I', 'A', 'K', 'R', 'V']
gpu=$1
bash scripts/test_dataset_cp.sh DTD ${gpu} ViT-B/16 1
bash scripts/test_dataset_cp.sh UCF101 ${gpu} ViT-B/16 1
bash scripts/test_dataset_cp.sh Aircraft ${gpu} ViT-B/16 2
bash scripts/test_dataset_cp.sh Caltech101 ${gpu} ViT-B/16 2
bash scripts/test_dataset_cp.sh Cars ${gpu} ViT-B/16 2
bash scripts/test_dataset_cp.sh eurosat ${gpu} ViT-B/16 2
bash scripts/test_dataset_cp.sh Flower102 ${gpu} ViT-B/16 2
bash scripts/test_dataset_cp.sh Food101 ${gpu} ViT-B/16 2
bash scripts/test_dataset_cp.sh Pets ${gpu} ViT-B/16 2
bash scripts/test_dataset_cp.sh SUN397 ${gpu} ViT-B/16 2


coop_weight='pretrained/coop/vit_b16_ep50_16shots/nctx4_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-50'
# bash scripts/test_tpt_coop.sh DTD ${gpu} ViT-B/16 ${coop_weight}
# bash scripts/test_tpt_coop.sh UCF101 ${gpu} ViT-B/16 ${coop_weight}
# bash scripts/test_tpt_coop.sh Aircraft ${gpu} ViT-B/16 ${coop_weight}
# bash scripts/test_tpt_coop.sh Caltech101 ${gpu} ViT-B/16 ${coop_weight}
# bash scripts/test_tpt_coop.sh Cars ${gpu} ViT-B/16 ${coop_weight}
# bash scripts/test_tpt_coop.sh eurosat ${gpu} ViT-B/16 ${coop_weight}
# bash scripts/test_tpt_coop.sh Flower102 ${gpu} ViT-B/16 ${coop_weight}
# bash scripts/test_tpt_coop.sh Food101 ${gpu} ViT-B/16 ${coop_weight}
# bash scripts/test_tpt_coop.sh Pets ${gpu} ViT-B/16 ${coop_weight}
# bash scripts/test_tpt_coop.sh SUN397 ${gpu} ViT-B/16 ${coop_weight}

# bash scripts/test_coop_zsh.sh DTD ${gpu} ViT-B/16 ${coop_weight}
# bash scripts/test_coop_zsh.sh UCF101 ${gpu} ViT-B/16 ${coop_weight}
# bash scripts/test_coop_zsh.sh Aircraft ${gpu} ViT-B/16 ${coop_weight}
# bash scripts/test_coop_zsh.sh Caltech101 ${gpu} ViT-B/16 ${coop_weight}
# bash scripts/test_coop_zsh.sh Cars ${gpu} ViT-B/16 ${coop_weight}
# bash scripts/test_coop_zsh.sh eurosat ${gpu} ViT-B/16 ${coop_weight}
# bash scripts/test_coop_zsh.sh Flower102 ${gpu} ViT-B/16 ${coop_weight}
# bash scripts/test_coop_zsh.sh Food101 ${gpu} ViT-B/16 ${coop_weight}
# bash scripts/test_coop_zsh.sh Pets ${gpu} ViT-B/16 ${coop_weight}
# bash scripts/test_coop_zsh.sh SUN397 ${gpu} ViT-B/16 ${coop_weight}

# caption ensemble
# bash scripts/caption_ensemble.sh DTD ${gpu} ViT-B/16 ${coop_weight}
# bash scripts/caption_ensemble.sh UCF101 ${gpu} ViT-B/16 ${coop_weight}
# bash scripts/caption_ensemble.sh Aircraft ${gpu} ViT-B/16 ${coop_weight}
# bash scripts/caption_ensemble.sh Caltech101 ${gpu} ViT-B/16 ${coop_weight}
# bash scripts/caption_ensemble.sh Cars ${gpu} ViT-B/16 ${coop_weight}
# bash scripts/caption_ensemble.sh eurosat ${gpu} ViT-B/16 ${coop_weight}
# bash scripts/caption_ensemble.sh Flower102 ${gpu} ViT-B/16 ${coop_weight}
# bash scripts/caption_ensemble.sh Food101 ${gpu} ViT-B/16 ${coop_weight}
# bash scripts/caption_ensemble.sh Pets ${gpu} ViT-B/16 ${coop_weight}
# bash scripts/caption_ensemble.sh SUN397 ${gpu} ViT-B/16 ${coop_weight}