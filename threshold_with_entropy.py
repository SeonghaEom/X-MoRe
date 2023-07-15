
import argparse

import time

from copy import deepcopy

from PIL import Image
import numpy as np
from tqdm import tqdm

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

################################################################################################################
class CaptionCache(Dataset):
    def __init__(self, shared_dict, length):
        self.shared_dict = shared_dict
        self.length = length

    def __getitem__(self, index, imagepath, K = None):
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
        if self.shared_dict[index]: return self.shared_dict[index][:K]
        else: return None

    def __len__(self):
        return self.length

def create_cache(path=None):
    if path:
        with open(path, 'rb') as f:
            cap_cache_dict = pickle.load(f)
        manager = Manager()
        shared_dict = manager.dict(cap_cache_dict)
    else:
        manager = Manager()
        shared_dict = manager.dict()
    cap_cache = CaptionCache(shared_dict, length=128)
    return cap_cache
################################################################################################################

def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # avg_logits = logits.mean(0) [1, 1000]
    
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -((avg_logits) * (torch.exp(avg_logits))).sum(dim=-1)

def JeffreyDiv(logit, logit2):
    #logit = image, logit2 = cap
    kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
    class_num = logit.size()[-1]
    uni_ent = avg_entropy(torch.ones(class_num).reshape(1, -1))
    # print("uni_ent ", uni_ent, avg_entropy(logit2))
    alpha = (1 - (avg_entropy(logit2)/uni_ent))
    # print(alpha)
    input = F.log_softmax(logit, dim=-1)
    target = F.softmax(logit2, dim=-1)
    #(1-entropy(p)/entropy(Unif(C)))
    input_ = F.log_softmax(logit2, dim=-1)
    target_ = F.softmax(logit, dim=-1)
    beta = (1 - (avg_entropy(logit)/uni_ent))
    # print(beta, avg_entropy(logit))
    # total = alpha+beta #weight sum 1
    one = alpha * kl_loss(input, target.detach()) #image를 caption에 맞춤
    # print(one)
    two = beta * kl_loss(input_, target_.detach())#caption을 Image에 맞춤
    # print(two)

    # print(alpha, beta)
    # print(one, two)
    # print(alpha/cap_max, beta/img_max)
    return one + two

def accuracy(output, target, topk=(1,), caption=None, logger=None):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        if len(output) == 1:
            pred = output.argmax(dim=-1)
        else:
            pred = torch.mean(output, dim=0)
            pred = pred.argmax(dim=-1, keepdim=True)
        # print("pred ", pred)
        assert pred.shape == target.shape, (pred.shape, target.shape)
        correct = pred.eq(target).sum()
        return correct

def test_time_tuning(model, cap_cache, inputs, optimizer, scaler, imagepath , mean_stat, cap_mean_stat, args, index, save_result):
    # Entropy + Triplet loss function * 0.5
    # Triplet loss function, anchor = retrieved vocab, positive = top5, negative = bottom5
    for j in range(args.tta_steps):
        with torch.cuda.amp.autocast():
            output_img, text_features = model(inputs) # bs, n_cls, (1, 1000), 
            # retrieved_Caption, retrieved_score = return_caption(imagepath, retrieve_K=args.retrieve_K)
            retrieved_Caption = cap_cache.__getitem__(index, imagepath, args.retrieve_K)
            if retrieved_Caption == None:
                return None, None
            # print(retrieved_Caption)
            output_caption = model.caption_ensemble(retrieved_Caption)
            
            ## Entropy
            conf_img = avg_entropy(output_img)
            # print("before ", avg_entropy(output_caption))
            ##### selection#####
            if args.retrieve_K >= 32:
                top = 0.1
                batch_entropy = -(output_caption.softmax(1) * output_caption.log_softmax(1)).sum(1)
                idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
                ##redefine retrieved_Caption, output_caption with selected idx
                retrieved_Caption = [retrieved_Caption[i] for i in idx]
                # print(retrieved_Caption[0])
                output_caption = output_caption[idx]
            conf_cap = avg_entropy(output_caption)
            # print("after ", conf_cap)
            
            #loss, penalty
            # print(JeffreyDiv(output_img, output_caption))
            JeffDiv = JeffreyDiv(output_img, output_caption)
            Entloss = conf_img  + conf_cap

             
            # print(loss)
            optimizer.zero_grad() 
            scaler.scale(Entloss + 1e-2*JeffDiv).backward()
            scaler.step(optimizer)
            scaler.update()

    return retrieved_Caption, output_caption, JeffDiv, Entloss


def test_time_adapt_eval(val_loader, model, cap_cache, model_state, optimizer, optim_state, scaler, save_result=False, set_id='', args=None):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    
    progress = ProgressMeter(
        len(val_loader),
        [batch_time],
        prefix='Test: ',
        logger = None)

    # reset model and switch to evaluate mode
    model.eval()
    with torch.no_grad():
        model.reset()
    end = time.time()
    cnt_empty = 0
    assert save_result != None
    ## to measure accuracy
    total_images = 0
    correct_images = 0
    
    ## image average entropy mean
    mean_stat = torch.tensor([], dtype=torch.float32).cuda(args.gpu, non_blocking=True)
    # cmean_stat = dict()
    
    ## caption average entropy mean
    cap_mean_stat = torch.tensor([], dtype=torch.float32).cuda(args.gpu, non_blocking=True)
    # cap_cmean_stat = 0
    
    # count accuracy of when using caption!
    cnt_cap = 0
    cnt_cap_correct = 0

    for i, (images, target, imagepath) in tqdm(enumerate(val_loader)): 
        assert args.gpu is not None
        if isinstance(images, list):
            for k in range(len(images)):
                images[k] = images[k].cuda(args.gpu, non_blocking=True)
            image = images[0]
            images = torch.cat(images, dim=0)
        else: image = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        ### One time training
        with torch.no_grad():
            model.reset()
        optimizer.load_state_dict(optim_state)
        retrieved_Caption, caption_logit, JeffDiv, Entloss = test_time_tuning(model, cap_cache, image, optimizer, scaler, imagepath[0], mean_stat, cap_mean_stat, args, index=i, save_result = save_result)
        if retrieved_Caption == None:
            cnt_empty +=1 
            continue

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output_img = model.inference(image.cuda(args.gpu, non_blocking=True))
                caption_logit = model.caption_ensemble(retrieved_Caption)

        conf_img = avg_entropy(output_img)
        conf_cap = avg_entropy(caption_logit)
        # print("at eval entropy ", conf_img, conf_cap)
        thres_img = torch.mean(mean_stat).item()
        thres_cap = torch.mean(cap_mean_stat).item()
        gap = thres_img - thres_cap

        # if conf_img > thres_img  and conf_cap < thres_cap and i > 200:
        if conf_img + gap > conf_cap and i > 200:
            correct_ = accuracy(caption_logit, target, topk=(1, 2, 3, 4, 5), caption=None, logger=None).item()
            cnt_cap += 1
            if correct_ : cnt_cap_correct +=1
        else:
            # print(output_img.shape)
            # print("target ", target)
            correct_ = accuracy(output_img , target, topk=(1, 2, 3, 4, 5), caption=None, logger=None).item()
            # print(correct_)
        mean_stat = torch.cat([mean_stat, torch.tensor([conf_img],dtype=torch.float32).cuda(args.gpu)])
        cap_mean_stat = torch.cat([cap_mean_stat, torch.tensor([conf_cap], dtype=torch.float32).cuda(args.gpu)])
        total_images += 1
        correct_images += correct_

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0 or (i+1) == len(val_loader):
            # progress.display(i)
            print("accuracy  ", correct_images/total_images)
            print("image confidence score mean stat", thres_img)
            print("caption confidence score mean stat", thres_cap)
            print("count caption correct {} out of {}".format(cnt_cap_correct, cnt_cap))
            save_result['accuracy'].append(correct_images/total_images)
            save_result['image_conf_mean'].append(torch.mean(mean_stat).item())
            save_result['caption_conf_mean'].append(torch.mean(cap_mean_stat).item())
            save_result['cap_cnt'].append(cnt_cap)
            save_result['cap_corr'].append(cnt_cap_correct)
            save_result['Div'].append(JeffDiv)
            save_result['Ent'].append(Entloss)

            # df = pd.DataFrame(save_result)
            # df = df.reset_index()
            # path = './notebook/inference_with_entropy/{}'.format(arch.replace('/', ''))
            # os.makedirs(path, exist_ok=True)
            # df.to_csv(os.path.join(path, 'inference_with_entropy_{}.csv'.format(set_id)))
        
    print("empty caption count = {}".format(cnt_empty))
    print("Accuracy: {}".format(correct_images/total_images) )
    return save_result

def main():
    args = parser.parse_args()
    set_random_seed(args.seed)

    # This codebase has only been tested under the single GPU setting
    assert args.gpu is not None
    main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu
    set_random_seed(args.seed)
    print("Use GPU: {} for training".format(args.gpu))
    # load model
    if args.test_sets in fewshot_datasets:
        classnames = eval("{}_classes".format(args.test_sets.lower()))
    model = get_coop(args.arch, args.test_sets, args.gpu, args.n_ctx, args.ctx_init)
    model_state = None

    cross_check = set()
    for name, param in model.named_parameters():

        if "prompt_learner" not in name:
            param.requires_grad_(False)
        if param.requires_grad : cross_check.add(name)
    print("tuing parameters ", cross_check)

    print("=> Model created: visual backbone {}".format(args.arch))


    assert gpu is not None
    torch.cuda.set_device(gpu)
    model = model.cuda(gpu)

    trainable_param = model.prompt_learner.parameters()
    optimizer = torch.optim.AdamW(trainable_param, args.lr)
    optim_state = deepcopy(optimizer.state_dict())


    scaler = torch.cuda.amp.GradScaler(init_scale=1000)
    print('=> Using native Torch AMP. Training in mixed precision.')

    cudnn.benchmark = True

    import sys
    from collections import defaultdict
        # norm stats from clip.load()
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                        std=[0.26862954, 0.26130258, 0.27577711])
    
    # iterating through eval datasets
    # if args.gpu == 3:
    #     test_sets='DTD/Food101/Pets'
    # elif args.gpu == 4:
    #     test_sets = 'eurosat/Aircraft'
    # elif args.gpu == 6:
    #     test_sets='Cars/SUN397'
    # elif args.gpu == 7:
    #     test_sets='UCF101'
    # else:
    #     test_sets=args.test_sets
    datasets = args.test_sets.split("/")
    results = {}
    for set_id in datasets:
        print("evaluating: {}".format(set_id))
        if not os.path.exists(os.path.join(args.cap_cache, '{}.pkl'.format(set_id))):
            cap_cache = create_cache()
            save_cache = True
        else:
            # cap_cache = torch.load(os.path.join('./cap_cache', '{}.pt'.format(set_id)))
            cap_cache = create_cache(os.path.join(args.cap_cache, '{}.pkl'.format(set_id)))
            save_cache = False
        print("save cache ", save_cache)
        for retrieve_K in [128, 64, 32, 16, 8, 4, 2]:
            args.retrieve_K = retrieve_K
            print("retrieve K: {}".format(retrieve_K))
            Dict = defaultdict(list)
            data_transform = transforms.Compose([
                transforms.Resize(args.resolution, interpolation=BICUBIC),
                transforms.CenterCrop(args.resolution),
                transforms.ToTensor(),
                normalize,
            ])
            batchsize = 1
            classnames = eval("{}_classes".format(set_id.lower()))
            model.reset_classnames(classnames, args.arch)
            with torch.no_grad():
                model.reset()
            optimizer.load_state_dict(optim_state)
            val_dataset = build_dataset(set_id, data_transform, args.data, mode=args.dataset_mode)
            total_length = len(val_dataset)
            print("number of test samples: {}".format(len(val_dataset)))
            val_loader = torch.utils.data.DataLoader(
                        val_dataset,
                        batch_size=batchsize, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
            results = test_time_adapt_eval(val_loader, model, cap_cache, model_state, optimizer, optim_state, scaler, Dict, set_id, args)
            if save_cache :
                path = args.cap_cache
                os.makedirs(path, exist_ok=True)
                shared_dict = dict(OrderedDict(sorted(cap_cache.shared_dict.items()))) # sort from key
                with open(os.path.join(path, '{}.pkl'.format(set_id)), "wb") as f:
                    pickle.dump(shared_dict, f)
                # torch.save(cap_cache, os.path.join(path, '{}.pt'.format(set_id)))
            
            df = pd.DataFrame(results)
            df = df.reset_index()

            path = './notebook/threshold_with_entropy/{}/{}'.format(args.arch, retrieve_K)
            os.makedirs(path, exist_ok=True)
            df.to_csv(os.path.join(path, 'threshold_with_entropy_{}.csv'.format(set_id)))

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning')
    parser.add_argument('data', metavar='DIR', help='path to dataset root')
    parser.add_argument('--test_sets', type=str, default='A/R/V/K/I', help='test dataset (multiple datasets split by slash)')
    parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
    parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('-p', '--print-freq', default=200, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--gpu', default=3, type=int,
                        help='GPU id to use.')
    parser.add_argument('--tpt', action='store_true', default=False, help='run test-time prompt tuning')
    parser.add_argument('--selection_p', default=0.1, type=float, help='confidence selection percentile')
    parser.add_argument('--tta_steps', default=1, type=int, help='test-time-adapt steps')
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')
    parser.add_argument('--ctx_init', default=None, type=str, help='init tunable prompts')
    parser.add_argument('--cocoop', action='store_true', default=False, help="use cocoop's output as prompt initialization")
    parser.add_argument('--load', default=None, type=str, help='path to a pre-trained coop/cocoop')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cap_cache', type=str, default='./cap_cache')

    main()
