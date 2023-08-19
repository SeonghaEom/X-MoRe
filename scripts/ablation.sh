

# 1
# python ensemble_with_entropy.py --test_sets='Caltech101' --cap_cache_file='Caltech101_ViT-B16.pkl' --data='/data/seongha' --n_ctx=4 --ctx_init='a_photo_of_a' --gpu=3 --arch=RN50 --retrieve_K=16 --seed=0
# python ensemble_with_entropy.py --test_sets='Caltech101' --cap_cache_file='Caltech101_ViT-B32.pkl' --data='/data/seongha' --n_ctx=4 --ctx_init='a_photo_of_a' --gpu=3 --arch=RN50 --retrieve_K=16 --seed=0


# python ensemble_with_entropy.py --test_sets='Flower102' --cap_cache_file='Flower102_ViT-B16.pkl' --data='/data/seongha' --n_ctx=4 --ctx_init='a_photo_of_a' --gpu=3 --arch=RN50 --retrieve_K=16 --seed=0
# python ensemble_with_entropy.py --test_sets='Flower102' --cap_cache_file='Flower102_ViT-B32.pkl' --data='/data/seongha' --n_ctx=4 --ctx_init='a_photo_of_a' --gpu=3 --arch=RN50 --retrieve_K=16 --seed=0

#2

# python ensemble_with_entropy.py --test_sets='Caltech101' --cap_cache_file='unsorted_Caltech101_ViT-L14.pkl' --data='/data/seongha' --n_ctx=4 --ctx_init='a_photo_of_a' --gpu=3 --arch=RN50 --retrieve_K=2 --seed=0
# python ensemble_with_entropy.py --test_sets='Caltech101' --cap_cache_file='unsorted_Caltech101_ViT-L14.pkl' --data='/data/seongha' --n_ctx=4 --ctx_init='a_photo_of_a' --gpu=3 --arch=RN50 --retrieve_K=4 --seed=0
# python ensemble_with_entropy.py --test_sets='Caltech101' --cap_cache_file='unsorted_Caltech101_ViT-L14.pkl' --data='/data/seongha' --n_ctx=4 --ctx_init='a_photo_of_a' --gpu=3 --arch=RN50 --retrieve_K=8 --seed=0
# python ensemble_with_entropy.py --test_sets='Caltech101' --cap_cache_file='unsorted_Caltech101_ViT-L14.pkl' --data='/data/seongha' --n_ctx=4 --ctx_init='a_photo_of_a' --gpu=3 --arch=RN50 --retrieve_K=16 --seed=0

# python ensemble_with_entropy.py --test_sets='Flower102' --cap_cache_file='unsorted_Flower102_ViT-L14.pkl' --data='/data/seongha' --n_ctx=4 --ctx_init='a_photo_of_a' --gpu=3 --arch=RN50 --retrieve_K=2 --seed=0
# python ensemble_with_entropy.py --test_sets='Flower102' --cap_cache_file='unsorted_Flower102_ViT-L14.pkl' --data='/data/seongha' --n_ctx=4 --ctx_init='a_photo_of_a' --gpu=3 --arch=RN50 --retrieve_K=4 --seed=0
# python ensemble_with_entropy.py --test_sets='Flower102' --cap_cache_file='unsorted_Flower102_ViT-L14.pkl' --data='/data/seongha' --n_ctx=4 --ctx_init='a_photo_of_a' --gpu=3 --arch=RN50 --retrieve_K=8 --seed=0
# python ensemble_with_entropy.py --test_sets='Flower102' --cap_cache_file='unsorted_Flower102_ViT-L14.pkl' --data='/data/seongha' --n_ctx=4 --ctx_init='a_photo_of_a' --gpu=3 --arch=RN50 --retrieve_K=16 --seed=0

#3
data=$1
gpu=$2
python ensemble_with_entropy.py --data='/data/seongha' --n_ctx=4 --ctx_init='a_photo_of_a' --gpu=${gpu} --arch=ViT-B/16 --retrieve_K=16 --seed=0 --test_sets=${data} --load='pretrained/coop/vit_b16_ep50_16shots/nctx4_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-50'