
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
from torch.nn.functional import normalize
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
    num_images=2000,
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
        if index not in self.shared_dict.keys():
            
            print('Adding {} to shared_dict'.format(index))
            res = client_backup.query(image=imagepath)
            if len(res) < self.length:
                res = client_backup2.query(image=imagepath)
            if isinstance(res, list) and len(res) >= self.length:
                self.shared_dict[index] = [each['caption'] for each in res[:self.length]]
            else:
                self.shared_dict[index] = None
        else:
            # print("use cache")
            # print(index)
            pass
        if self.shared_dict[index]: return self.shared_dict[index][:K]
        else:
            return None

    def __len__(self):
        return self.length

def create_cache(path=None):
    if path:
        with open(path, 'rb') as f:
            cap_cache_dict = pickle.load(f)
            Cnt_empty = sum([v==None for k, v in cap_cache_dict.items()])
            print(Cnt_empty)
        manager = Manager()
        shared_dict = manager.dict(cap_cache_dict)
    else:
        manager = Manager()
        shared_dict = manager.dict()
    cap_cache = CaptionCache(shared_dict, length=128)
    return cap_cache
################################################################################################################
def logit_coefficient(logit):
    class_num = logit.shape[-1]
    uni_ent = avg_entropy(torch.ones(class_num).reshape(1, -1))
    ent = avg_entropy(logit)
    alpha = (1 - (ent/uni_ent)) #image coefficient
    return alpha

def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # avg_logits = logits.mean(0) [1, 1000]
    
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -((avg_logits) * (torch.exp(avg_logits))).sum(dim=-1)

def accuracy(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        if len(output) == 1:
            # print("predict image here")
            pred = output.argmax(dim=-1)
        else:
            pred = torch.mean(output, dim=0)
            pred = pred.argmax(dim=-1, keepdim=True)
        # print("pred ", pred)
        assert pred.shape == target.shape, (pred.shape, target.shape)
        correct = pred.eq(target).sum()
        return correct

def test_time_adapt_eval(val_loader, model, cap_cache, optimizer=None, optim_state=None, scaler=None, save_result=False, set_id='', args=None, label_features = None):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    
    progress = ProgressMeter(
        len(val_loader),
        [batch_time],
        prefix='Test: ',
        logger = None)

    # reset model and switch to evaluate mode
    model.eval()
    end = time.time()
    cnt_empty = 0
    assert save_result != None
    ## to measure accuracy
    correct_images = 0
    total_images = 0
    
    ## image average entropy mean
    mean_stat = torch.tensor([], dtype=torch.float32).cuda(args.gpu, non_blocking=True)
    ## caption average entropy mean
    cap_mean_stat = torch.tensor([], dtype=torch.float32).cuda(args.gpu, non_blocking=True)
    
    for i, (c_idx, images, target, imagepath) in tqdm(enumerate(val_loader)): 
        if isinstance(images, list):
            for k in range(len(images)):
                images[k] = images[k].cuda(args.gpu, non_blocking=True)
            image = images[0]
            images = torch.cat(images, dim=0)
        else: image = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        retrieved_Caption = cap_cache.__getitem__(c_idx.item(), imagepath[0], args.retrieve_K)
        if retrieved_Caption == None:
            cnt_empty +=1 
            continue
        else:
            with torch.no_grad():
                output_img = model.inference(image.cuda(args.gpu, non_blocking=True), label_features)
                caption_logit = model.caption_ensemble(retrieved_Caption, label_features)
                if i==0: print("Norm ", torch.norm(output_img), torch.norm(caption_logit[0]))
            # caption_logit = caption_logit[0]
            alpha = logit_coefficient(output_img)
            beta = logit_coefficient(caption_logit)
            mean_stat = torch.cat([mean_stat, torch.tensor([alpha],dtype=torch.float32).cuda(args.gpu)])
            cap_mean_stat = torch.cat([cap_mean_stat, torch.tensor([beta], dtype=torch.float32).cuda(args.gpu)])

            assert mean_stat.requires_grad == False and cap_mean_stat.requires_grad == False
            img_normalized = normalize(mean_stat, p=1.0, dim = 0)[-1]
            cap_normalized = normalize(cap_mean_stat, p=1.0, dim = 0)[-1]
            
            ## normalize logit
            img_logit_norm = torch.nn.functional.softmax(output_img, dim=-1)
            cap_logit_norm = torch.nn.functional.softmax(caption_logit, dim=-1)
            correct_ = accuracy(img_normalized * img_logit_norm + cap_normalized * cap_logit_norm , target).item()

        correct_images += correct_
        total_images +=1
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0 or (i+1) == len(val_loader):
            # progress.display(i)
            print("accuracy {:.4f}".format(correct_images/total_images))
            save_result['accuracy'].append(correct_images/total_images)
            save_result['image_coefficient'].append(torch.mean(mean_stat).item())
            save_result['caption_coefficient'].append(torch.mean(cap_mean_stat).item())
            save_result['empty_caption'].append(cnt_empty)

    print("empty caption count = {}".format(cnt_empty))
    print("Accuracy: {:.4f}".format(correct_images/total_images))
    return save_result

def load_model(args):
    if args.cocoop:
        model = get_cocoop(args.arch, args.test_sets, 'cpu', args.n_ctx)
        assert args.load is not None
        load_model_weight(args.load, model, 'cpu', args) # to load to cuda: device="cuda:{}".format(args.gpu)
        model_state = deepcopy(model.state_dict())
    else:
        model = get_coop(args.arch, args.test_sets, args.gpu, args.n_ctx, args.ctx_init)
        
        if args.load is not None:
            print("Use pre-trained soft prompt (CoOp) as initialization")
            tmp = torch.load(args.load, map_location="cpu")
            pretrained_ctx = tmp['state_dict']['ctx']
            assert pretrained_ctx.size()[0] == args.n_ctx
            with torch.no_grad():
                model.prompt_learner.ctx.copy_(pretrained_ctx)
                model.prompt_learner.ctx_init_state = pretrained_ctx.to(args.gpu)
                assert model.prompt_learner.device.index == args.gpu, model.prompt_learner.device.index
                assert model.prompt_learner.ctx.device.index == args.gpu, model.prompt_learner.ctx.device
                assert model.prompt_learner.ctx_init_state.device.index == args.gpu, model.prompt_learner.ctx_init_state.device
                del pretrained_ctx
                del tmp
            assert model.prompt_learner.device.index == args.gpu, model.prompt_learner.device
        model_state = None
    cross_check = set()
    for name, param in model.named_parameters():
        # if "prompt_learner" not in name:
        param.requires_grad_(False)
        if param.requires_grad : cross_check.add(name)
    print("tuning parameters ", cross_check)

    print("=> Model created: visual backbone {}".format(args.arch))

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    return model

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
    
    print('=> Using native Torch AMP. Training in mixed precision.')

    cudnn.benchmark = True
    from collections import defaultdict
        # norm stats from clip.load()
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                        std=[0.26862954, 0.26130258, 0.27577711])
    
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

        # retrieve_K = 8
        path = './notebook/ensemble_with_entropy/{}/{}/{}'.format(args.arch, args.seed, args.retrieve_K)
        if args.load: path = './notebook/ensemble_with_entropy_pretrained/{}/{}/{}'.format(args.arch, args.seed, args.retrieve_K)
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'ensemble_with_entropy_{}.csv'.format(set_id))
        if os.path.exists(path):
            continue

        print("retrieve K: {}".format(args.retrieve_K))
        Dict = defaultdict(list)
        data_transform = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=BICUBIC),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            normalize,
        ])
        batchsize = 1
        if args.test_sets in fewshot_datasets:
            classnames = eval("{}_classes".format(args.test_sets.lower()))
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
        
        model = load_model(args)
        model.reset_classnames(classnames, args.arch)
        label_features = model.get_text_features() ## keep label text features
        print("label feature shape ", label_features.shape)
        # trainable_param = model.prompt_learner.parameters()
        # optimizer = torch.optim.AdamW(trainable_param, args.lr)
        # optim_state = deepcopy(optimizer.state_dict())
        # optimizer.load_state_dict(optim_state)
        # scaler = torch.cuda.amp.GradScaler()
        
        val_dataset = build_dataset(set_id, data_transform, args.data, mode=args.dataset_mode)
        total_length = len(val_dataset)
        print("number of test samples: {}".format(len(val_dataset)))
        val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batchsize, shuffle=True,
                    num_workers=args.workers, pin_memory=True)
        results = test_time_adapt_eval(val_loader, model, cap_cache, optimizer=None, optim_state=None, scaler=None, save_result=Dict, set_id=set_id, args=args, label_features = label_features)
        if save_cache :
            path = args.cap_cache
            os.makedirs(path, exist_ok=True)
            shared_dict = dict(OrderedDict(sorted(cap_cache.shared_dict.items()))) # sort from key
            with open(os.path.join(path, '{}.pkl'.format(set_id)), "wb") as f:
                pickle.dump(shared_dict, f)
        
        df = pd.DataFrame(results)
        df = df.reset_index()

        path = './notebook/ensemble_with_entropy/{}/{}/{}'.format(args.arch, args.seed, args.retrieve_K)
        if args.load: path = './notebook/ensemble_with_entropy_pretrained/{}/{}/{}'.format(args.arch, args.seed, args.retrieve_K)
        os.makedirs(path, exist_ok=True)
        df.to_csv(os.path.join(path, 'ensemble_with_entropy_{}.csv'.format(set_id)))

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning')
    parser.add_argument('--data', metavar='DIR', help='path to dataset root')
    parser.add_argument('--test_sets', type=str, default='DTD/Flower102/Food101/Cars/SUN397/Aircraft/Pets/Caltech101/UCF101/eurosat', help='test dataset (multiple datasets split by slash)')
    parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT-B/16')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
    parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('-p', '--print-freq', default=1000, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--gpu', default=3, type=int,
                        help='GPU id to use.')
    parser.add_argument('--tpt', action='store_true', default=False, help='run test-time prompt tuning')
    parser.add_argument('--JeffDiv', action='store_true', default=False, help='jeffreydivergence')
    parser.add_argument('--selection_p', default=0.1, type=float, help='confidence selection percentile')
    parser.add_argument('--tta_steps', default=1, type=int, help='test-time-adapt steps')
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')
    parser.add_argument('--ctx_init', default=None, type=str, help='init tunable prompts')
    parser.add_argument('--cocoop', action='store_true', default=False, help="use cocoop's output as prompt initialization")
    parser.add_argument('--load', default=None, type=str, help='path to a pre-trained coop/cocoop')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cap_cache', type=str, default='./cap_cache')
    parser.add_argument('--retrieve_K', type=int, default=32)

    main()
