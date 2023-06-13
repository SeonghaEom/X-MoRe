import os
import time
import random

import numpy as np

import shutil
from enum import Enum

import torch
import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
from collections import Counter

import logging
import datetime

def create_logger(args=None):
    # Get the current date
    current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create a logger object
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Specify the directory path for the log file
    log_directory = f"./logs/{args.arch}/{args.test_sets}/ours_{args.ours}/{args.retrieve_K}"
    # Create the log directory if it doesn't exist
    os.makedirs(log_directory, exist_ok=True)

    log_file = os.path.join(log_directory, f"log_{current_date}.txt")
    file_handler = logging.FileHandler(log_file)

    # Create a formatter and add it to the file handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)
    return logger


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", logger=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger= logger

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
        if self.logger: self.logger.info('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))
        if self.logger: self.logger.info(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,), caption=None, logger=None, args=None):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        if output.shape[0] == 1:#only image prediction
            logit_k, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
        else: # evaluate captions
            bag = []
            # # length = max(5, output.shape[0]-1)
            # cap_pred = output[1:]
            # # cap_pred = torch.mean(cap_pred, 0,  keepdim=True)
            # _, pred = cap_pred.topk(maxk, 1, True, True) #5, 1 #candidate labels
            # pred = pred.reshape(maxk, 1)
            pred = torch.mean(output, 0, keepdim=True)
            _, pred = pred.topk(maxk, 1, True, True)
            pred = pred.reshape(maxk, 1)
            
            # for i, each in enumerate(output):
            #     if i==0: # image
            #         # continue
            #         each = each.unsqueeze(0) #1, 1000
            #         _, image_pred = each.topk(maxk, 1, True, True) #1, 5
            #         image_pred = image_pred.t() #5, 1
            #     else: #caption
            #         each = each.unsqueeze(0) #1, 1000
            #         _, pred = each.topk(maxk, 1, True, True)
            #         pred = pred.t() #5, 1
            #         for elem in pred.tolist(): bag.append(elem[0])
            # # # _, pred = output.topk(maxk, 1, True, True)
            # # # pred = pred.t()
            # c = Counter(bag)

            # pred = c.most_common(maxk)
            
            # print(pred)
            # pred = [each[0] for each in pred]
            # pred = torch.tensor(pred).to(output.device) #5
            # pred = pred.reshape(maxk, 1)
            # print("image prediction" , image_pred)
            print("caption prediction" , pred)
            print("label ", target.view(1, -1).expand_as(pred))
            correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            if k == 1 and correct_k.item() == 0:
                if args.debug:
                    print("-------wrong prediction-------")
                    # logger.info("wrong prediction , logit: ", output)
                    print("target {} & predicted value {}".format( target, pred))
                    print("logit ", logit_k)
                    if logger: logger.info("wrong prediction, target {} & predicted value {}".format(target, pred))
                    print("-------------------------------")
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
        

def load_model_weight(load_path, model, device, args):
    if os.path.isfile(load_path):
        print("=> loading checkpoint '{}'".format(load_path))
        checkpoint = torch.load(load_path, map_location=device)
        state_dict = checkpoint['state_dict']
        # Ignore fixed token vectors
        if "token_prefix" in state_dict:
            del state_dict["token_prefix"]

        if "token_suffix" in state_dict:
            del state_dict["token_suffix"]

        args.start_epoch = checkpoint['epoch']
        try:
            best_acc1 = checkpoint['best_acc1']
        except:
            best_acc1 = torch.tensor(0)
        if device != 'cpu':
            # best_acc1 may be from a checkpoint from a different GPU
            best_acc1 = best_acc1.to(device)
        try:
            model.load_state_dict(state_dict)
        except:
            # TODO: implement this method for the generator class
            model.prompt_generator.load_state_dict(state_dict, strict=False)
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(load_path, checkpoint['epoch']))
        del checkpoint
        torch.cuda.empty_cache()
    else:
        print("=> no checkpoint found at '{}'".format(load_path))


def validate(val_loader, model, criterion, args, output_mask=None):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast():
                output = model(images)
                if output_mask:
                    output = output[:, output_mask]
                loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
        progress.display_summary()

    return top1.avg
