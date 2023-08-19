#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse

import time

from copy import deepcopy

from PIL import Image
import numpy as np
from tqdm import tqdm
from clip import load, tokenize

import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import pandas as pd
import os

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models

import clip
from clip.custom_clip import get_coop
from clip.cocoop import get_cocoop
from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset
from utils.tools import Summary, AverageMeter, ProgressMeter, load_model_weight, set_random_seed, create_logger
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask
from clip_retrieval.clip_client import ClipClient, Modality
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict, OrderedDict
import pickle
from multiprocessing import Manager

client = ClipClient(
    url="http://127.0.0.1:1234/knn-service",
    indice_name='laion_400m',
    modality=Modality.IMAGE,
    num_images=20,
    deduplicate=False,
)
client_backup = ClipClient(
    url="http://127.0.0.1:1234/knn-service",
    indice_name='laion_400m',
    modality=Modality.IMAGE,
    num_images=200,
    deduplicate=False,
)

client_backup2 = ClipClient(
    url="http://127.0.0.1:1234/knn-service",
    indice_name='laion_400m',
    modality=Modality.IMAGE,
    num_images=1000,
    deduplicate=False,
)
## Class to names mapping
fewshot_datasets = ['DTD', 'Flower102', 'Food101', 'Cars', 'SUN397', 
                    'Aircraft', 'Pets', 'Caltech101', 'UCF101', 'eurosat']
test_sets = 'DTD/Flower102/Food101/Cars/SUN397/Aircraft/Pets/Caltech101/UCF101/eurosat'

###############################################################################################################
class CaptionCache(Dataset):
    def __init__(self, shared_dict, length):
        self.shared_dict = shared_dict
        self.length = length

    def __getitem__(self, index, imagepath=None, K = None):
        if index not in self.shared_dict:
            if index == 0: print('Adding {} to shared_dict'.format(index))
            res = client_backup.query(image=imagepath)
            if len(res) < self.length:
                res = client_backup2.query(image=imagepath)
            if isinstance(res, list) and len(res) >= self.length:
                self.shared_dict[index] = [each['caption'] for each in res[:self.length]]
            else:
                self.shared_dict[index] = None
        else:
            if index == 0: print("use cache")
        if self.shared_dict[index]:
            return self.shared_dict[index][:K]
        else: return None

    def __len__(self):
        return self.length

def create_cache(path=None):
    if path and os.path.exists(path):
        with open(path, 'rb') as f:
            unpickler = pickle.Unpickler(f)
            cap_cache_dict = unpickler.load()
        manager = Manager()
        shared_dict = manager.dict(cap_cache_dict)
        print("size of cache ", len(shared_dict.keys()))
    else:
        manager = Manager()
        shared_dict = manager.dict()
    cap_cache = CaptionCache(shared_dict, length=128)
    return cap_cache, len(shared_dict.keys())
################################################################################################################

normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                        std=[0.26862954, 0.26130258, 0.27577711])
fs_test_sets = 'DTD/Flower102/Food101/Cars/SUN397/Aircraft/Pets/Caltech101/UCF101/eurosat'.split('/')
ood_test_Sets = 'V/A/K/R'.split('/')
gpu=3
if gpu==2:
    datasets= fs_test_sets
else:
    datasets = ood_test_Sets
data_path = '/data/seongha'

model = get_coop('ViT-L/14', datasets, gpu, 4, "a_photo_of_a")
model.eval()
for set_id in datasets:
    print("evaluating: {}".format(set_id))
    cap_cache_path = './cap_cache'
    path = os.path.join(cap_cache_path, '{}.pkl'.format(set_id))
    cap_cache, cur_i = create_cache(path)
    for retrieve_K in [128]:
        print("retrieve K: {}".format(retrieve_K))
        Dict = defaultdict(list)
        data_transform = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        batchsize = 1
        if test_sets in fewshot_datasets:
            classnames = eval("{}_classes".format(test_sets.lower()))
        else:
            classnames = imagenet_classes
        # Reset classnames of custom CLIP model
        if len(set_id) > 1: 
            # fine-grained classification datasets
            classnames = eval("{}_classes".format(set_id.lower()))
        else:
            assert set_id in ['A', 'R', 'K', 'V', 'I']
            classnames_all = imagenet_classes
            classnames = []
            if set_id in ['A', 'R', 'V']:
                label_mask = eval("imagenet_{}_mask".format(set_id.lower()))
                if set_id == 'R':
                    for i, m in enumerate(label_mask):
                        if m:
                            classnames.append(classnames_all[i])
                else:
                    classnames = [classnames_all[i] for i in label_mask]
            else:
                classnames = classnames_all

        val_dataset = build_dataset(set_id, data_transform, data_path, mode='test')
        total_length = len(val_dataset)
        print("number of test samples: {}".format(len(val_dataset)))
        val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=1, shuffle=False,
                    num_workers=16, pin_memory=True)
        for i, (c_idx, images, target, imagepath) in tqdm(enumerate(val_loader)): 
            c_idx = c_idx.item()
            retrieved_Caption = cap_cache.__getitem__(c_idx, imagepath[0], retrieve_K)
            if retrieved_Caption==None: continue
            retrieved_Caption = list(set(retrieved_Caption))
            with torch.no_grad():
                image_feature = model.image_encoder(images.cuda(gpu))
                tokens = torch.cat([tokenize(txt, truncate=True) for txt in retrieved_Caption]).cuda(gpu)
                text_feature= model.text_encoder.encode_text(tokens)
                
            score = torch.matmul(image_feature, text_feature.t()) # 512, 512-
            _, indices = torch.sort(score, descending=True)
            indices = indices.squeeze().tolist()
            cap_cache.shared_dict[c_idx] = [retrieved_Caption[i] for i in indices]
            if i == 0: print(cap_cache.shared_dict[c_idx])
        shared_Dict = dict(OrderedDict(sorted(cap_cache.shared_dict.items()))) # sort from key
        with open(os.path.join(cap_cache_path, '{}.pkl'.format(set_id)), "wb") as f:
            pickle.dump(shared_Dict, f)

