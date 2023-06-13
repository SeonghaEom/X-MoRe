import argparse

import time

from copy import deepcopy

from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms


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
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed, create_logger
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask

from clip_retrieval.clip_client import ClipClient, Modality
# client_model, _, _ = clip.load("ViT-L/14", device="cpu", jit=True)
import spacy
import spacy_fastlang
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('language_detector', last=True)


# def get_image_emb(image):
#     with torch.no_grad():
#         image_emb = client_model.encode_image(image.to("cpu"))
#         image_emb /= image_emb.norm(dim=-1, keepdim=True)
#         # image_emb = image_emb.cpu().detach().numpy().astype("float32")[0]
#         image_emb = image_emb.cpu().detach().numpy()[0]
#         assert image_emb.any()
#         return image_emb
client = ClipClient(
    url="http://127.0.0.1:1234/knn-service",
    indice_name='laion_400m',
    modality=Modality.IMAGE,
    num_images=10,
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

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return logits[idx], idx

def avg_entropy(outputs):
    # epsilon = 1e-10
    assert len(outputs) > 0
    assert torch.any(torch.isnan(outputs)) == False
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, 1000]
    assert torch.any(torch.isnan(logits)) == False
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # avg_logits = logits.mean(0) [1, 1000]
    # print(avg_logits)
    if torch.any(torch.isnan(avg_logits)):
        print("average logits ", outputs.log_softmax(dim=1).mean(0))
    assert torch.any(torch.isnan(avg_logits)) == False
    
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    assert torch.any(torch.isnan(avg_logits)) == False
    return -((avg_logits) * (torch.exp(avg_logits))).sum(dim=-1)

def avg_cos_dist(cand, pseudo):
    # epsilon = 1e-10

    eps = 1e-6
    sum = 0
    for each in pseudo:
        sum += torch.sum(torch.nn.functional.cosine_similarity(cand, each.expand_as(cand), dim=-1, eps=eps))
    return sum

def test_time_tuning(model, inputs, optimizer, scaler, args, logger = None, img_path=None):
    # One time training
    # model is CLIPTestTimeTuning, inputs :images
    
    if args.cocoop:
        image_feature, pgen_ctx = inputs
        pgen_ctx.requires_grad = True
        optimizer = torch.optim.AdamW([pgen_ctx], args.lr)
    
    selected_idx = None
    for j in range(args.tta_steps):
        assert img_path != None
        caption = []
        with torch.cuda.amp.autocast():
            if args.cocoop:
                output = model((image_feature, pgen_ctx))
            else:
                output_img, text_features = model(inputs) 
                # retrieved_txt_prj, _ = model.nn_project(image=inputs, print_hits=False, topk=args.retrieve_K) # K, 512, retrieved text projection
                try:
                    retrieved_txt= [D['caption'] for D in client.query(image=img_path)[:args.retrieve_K]]
                except:
                    print(client.query(image=img_path))
                    retrieved_txt= [D['caption'] for D in client_backup.query(image=img_path)[:args.retrieve_K]]
                if len(retrieved_txt) == args.retrieve_K:
                    retrieved_txt_feat = model.forward_caption(retrieved_txt)
                else:
                    retrieved_txt= [D['caption'] for D in client_backup.query(image=img_path)[:args.retrieve_K]]
                    if len(retrieved_txt) == args.retrieve_K:
                        retrieved_txt_feat = model.forward_caption(retrieved_txt)
                    else:
                        retrieved_txt= [D['caption'] for D in client_backup2.query(image=img_path)[:args.retrieve_K]]
                        assert len(retrieved_txt) == args.retrieve_K, (img_path, len(retrieved_txt))
                        retrieved_txt_feat = model.forward_caption(retrieved_txt)

            if selected_idx is not None:
                output = output[selected_idx]
            else:
                # pass
                # output, selected_idx = select_confident_samples(output, args.selection_p)
                loss = 0
                # ent = avg_entropy(output_img)
                if args.loss in ["both", "cosine"]:
                    # args.retrieve_K = 5 #TODO: 1,2,4,8
                    
                    K = 5
                    top, top_ind = output_img.topk(2*K, 1, True, True)
                    # print(top_ind.shape)
                    top5_ind = top_ind.squeeze()[:K]
                    bot5_ind = top_ind.squeeze()[K:]
                    assert len(top5_ind) == K and len(bot5_ind) == K, (top5_ind, bot5_ind)
                    # bot5, bot5_ind = output_img.topk(K, 1, False, True)
                    print("Top 5 ind ", top5_ind)
                    print("proability top {}: {}".format(K, top.squeeze()[:K]))
                    print("proability bottom {}: {}".format(K, top.squeeze()[K:]))
                    # loss2 = avg_cos_dist(text_features[top5_ind], projections)
                    x1 = torch.cat((text_features[top5_ind].squeeze(), text_features[bot5_ind].squeeze()), dim=0).cuda(args.gpu)
                    label = torch.tensor([1 for _ in range(K)] + [ -1 for _ in range(K)]).cuda(args.gpu)
                    criterion = torch.nn.CosineEmbeddingLoss()
                    loss2=[]
                    for each in retrieved_txt_feat:
                        loss2.append(criterion(x1, each.expand_as(x1), label))
                    print("Cosine embedding loss ", (sum(loss2)/len(loss2)).detach().cpu())
                    loss += sum(loss2)/len(loss2) * 0.5
                if args.loss in ["both", "entropy"]:
                    loss += avg_entropy(retrieved_txt_feat @ text_features.t()) 
                    if args.debug: print("Entropy loss: ", loss.detach().cpu())
                else:
                    NotImplementedError("not implemented for this case")
                    
                # elif loss > 5:
                #     logger.info("entropy {}".format(loss))
                    
                #     assert len(inputs.shape) == 4 and inputs.shape[0] == 1
                #     embedding = get_image_emb(inputs)
                #     assert len(embedding) > 0
                #     q_res = client.query(embedding_input = embedding.tolist())
                #     for each in q_res:
                #         if nlp(each["caption"])._.language == "en":
                #             # print(each)
                #             caption.append(each["caption"])
                    
                #     try:
                #         pseudo = model.forward_caption(caption) # n_cap, 1024
                #         output_cap = pseudo  @ text_features.t()
                        
                #         if len(caption) >= 2:
                #             # print("number of captions ", len(caption))
                #             output_cap, selected_idx = select_confident_samples(output_cap, args.selection_p)
                        
                #         output_cap # 2, 1000
                #         top5, top5_ind = output_cap.topk(5, 1, True, True)
                        
                #         # loss = avg_entropy(output)
                #         loss2 = avg_cos_dist(text_features[top5_ind], pseudo)
                #         logger.info(loss2)
                #         loss += loss2
                #     except:
                #         pass
                
                    # output, selected_idx = select_confident_samples(output_cap, args.selection_p)
                    # output, selected_idx = select_ambiguous_samples(output, args.selection_p)
                    ## get most confident
                    # # print(selected_idx)
                    # val, ind = torch.topk(output[0], 1)
                    # conf_label = ind.item()
                    # # txt_feat = model.get_text_features()
                    # conf_label_emb = txt_feat[conf_label]  ## get most confident label candidate
                    # ## mse loss with caption embedding (top_feat)
                    # mse_loss = torch.nn.MSELoss()
                    # loss2 = mse_loss(conf_label_emb, top_feat)
                    # print("mse loss ", loss2)
                    # loss = avg_entropy(output_img)
            # print("average entropy: ", loss)
        
        optimizer.zero_grad()
        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.step(optimizer)
        scaler.update()
    if args.cocoop:
        return pgen_ctx

    return len(caption)>0


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
    logger = create_logger(args)
    logger.info(args)

    # create model (zero-shot clip model (ViT-L/14@px336) with promptruning)
    if args.test_sets in fewshot_datasets:
        classnames = eval("{}_classes".format(args.test_sets.lower()))
    else:
        classnames = imagenet_classes
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
            # print(pretrained_ctx.device) #0
            assert pretrained_ctx.size()[0] == args.n_ctx
            with torch.no_grad():
                model.prompt_learner.ctx.copy_(pretrained_ctx)
                model.prompt_learner.ctx_init_state = pretrained_ctx.to(args.gpu)
                # print(model.prompt_learner.ctx_init_state)
                assert model.prompt_learner.device.index == args.gpu, model.prompt_learner.device.index
                assert model.prompt_learner.ctx.device.index == args.gpu, model.prompt_learner.ctx.device
                assert model.prompt_learner.ctx_init_state.device.index == args.gpu, model.prompt_learner.ctx_init_state.device
                # model.prompt_learner[0].ctx.copy_(pretrained_ctx)
                # model.prompt_learner[0].ctx_init_state = pretrained_ctx
            # assert torch.cuda.current_device() == args.gpu, torch.cuda.current_device()
                del pretrained_ctx
                del tmp
            # print(pretrained_ctx.device) #0
            assert model.prompt_learner.device.index == args.gpu, model.prompt_learner.device
        model_state = None

    cross_check = set()
    for name, param in model.named_parameters():
        if not args.cocoop:
            if "prompt_learner" not in name:
                param.requires_grad_(False)
        else:
            if "text_encoder" not in name:
                param.requires_grad_(False)
        if param.requires_grad : cross_check.add(name)
        

    # for param_name, param in model.named_parameters():
    #     # Check the current data type
    #     current_dtype = param.dtype
    #     print(f"Parameter {param_name} - Current dtype: {current_dtype}")

    #     # Convert to float32 if not already
    #     if current_dtype != torch.float16:
    #         param.data = param.data.half()
        
    #     # Verify the new data type
    #     new_dtype = param.dtype
    #     print(f"Parameter {param_name} - New dtype: {new_dtype}\n")
    
    print("tuing parameters ", cross_check)
    
    print("=> Model created: visual backbone {}".format(args.arch))
    
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        assert args.gpu is not None
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    # define optimizer
    if args.cocoop:
        optimizer = None
        optim_state = None
    else:
        trainable_param = model.prompt_learner.parameters()
        optimizer = torch.optim.AdamW(trainable_param, args.lr)
        optim_state = deepcopy(optimizer.state_dict())

    # setup automatic mixed-precision (Amp) loss scaling
    scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    print('=> Using native Torch AMP. Training in mixed precision.')

    cudnn.benchmark = True

    # norm stats from clip.load()
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    
    # iterating through eval datasets
    datasets = args.test_sets.split("/")
    results = {}
    for set_id in datasets:

        data_transform = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=BICUBIC),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            normalize,
        ])
        batchsize = 1
        
        logger.info("evaluating: {}".format(set_id))
        print("evaluating: {}".format(set_id))
        # reset the model
        # Reset classnames of custom CLIP model
        if len(set_id) > 1: 
            # fine-grained classification datasets
            classnames = eval("{}_classes".format(set_id.lower()))
        else:
            assert set_id in ['A', 'R', 'K', 'V', 'I'], set_id
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
        if args.cocoop:
            model.prompt_generator.reset_classnames(classnames, args.arch)
            model = model.cpu()
            model_state = model.state_dict()
            model = model.cuda(args.gpu)
        else:
            model.reset_classnames(classnames, args.arch)

        val_dataset = build_dataset(set_id, data_transform, args.data, mode=args.dataset_mode)
        print("number of test samples: {}".format(len(val_dataset)))
        logger.info("number of test samples: {}".format(len(val_dataset)))
        val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batchsize, shuffle=False,
                    num_workers=args.workers, pin_memory=True)
            
        results[set_id] = test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args, logger)
        del val_dataset, val_loader
        try:
            print("=> Acc. on testset [{}]: @1 {}/ @2 {}/ @3 {}/ @4 {}/ @5 {}".format(set_id, results[set_id][0], results[set_id][1], results[set_id][2], results[set_id][3], results[set_id[4], results[set_id][5]]))
        except:
            print("=> Acc. on testset [{}]: {}".format(set_id, results[set_id]))

    print("======== Result Summary ========")
    print("params: nstep	lr	bs")
    print("params: {}	{}	{}".format(args.tta_steps, args.lr, args.batch_size))
    print("\t\t [set_id] \t\t Top-1 acc. \t\t Top-5 acc.")
    for id in results.keys():
        print("{}".format(id), end="	")
    print("\n")
    for id in results.keys():
        print("{:.2f}".format(results[id][0]), end="	")
    print("\n")


def test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args, logger):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top2 = AverageMeter('Acc@2', ':6.2f', Summary.AVERAGE)
    top3 = AverageMeter('Acc@3', ':6.2f', Summary.AVERAGE)
    top4 = AverageMeter('Acc@4', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top2, top3, top4, top5],
        prefix='Test: ',
        logger = logger)

    # reset model and switch to evaluate mode
    model.eval()
    if not args.cocoop: # no need to reset cocoop because it's fixed
        with torch.no_grad():
            model.reset()
    end = time.time()
    
    for i, each in tqdm(enumerate(val_loader)): 
        if args.test_sets in fewshot_datasets:
            image, target, img_path = each
            assert len(img_path) == 1
            
        else:
            image, target = each
        assert args.gpu is not None

        target = target.cuda(args.gpu, non_blocking=True)
        
        ### One time training
        # reset the tunable prompt to its initial state
        if not args.cocoop and not args.caption_ensemble: # no need to reset cocoop because it's fixed
            if args.tta_steps > 0:
                with torch.no_grad():
                    model.reset()
            optimizer.load_state_dict(optim_state)
            
            retrieved_caption = test_time_tuning(model, image.cuda(args.gpu, non_blocking=True), optimizer, scaler, args, logger, img_path[0])
        elif args.caption_ensemble:
            pass
        else:
            NameError("no case for cocoop")

        # The actual inference goes here
        if args.tpt:
            if args.cocoop:
                image_feature = image_feature[0].unsqueeze(0)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                if args.cocoop:
                    output = model((image_feature, pgen_ctx))
                elif args.caption_ensemble:
                    assert img_path
                    img_path = img_path[0]
                    try:
                        retrieved_txt= [D['caption'] for D in client.query(image=img_path)[:args.retrieve_K]]
                    except:
                        print(client.query(image=img_path))
                        retrieved_txt= [D['caption'] for D in client_backup.query(image=img_path)[:args.retrieve_K]]
                    if len(retrieved_txt) == args.retrieve_K:
                        output = model.caption_ensemble(retrieved_txt)
                    else:
                        retrieved_txt= [D['caption'] for D in client_backup.query(image=img_path)[:args.retrieve_K]]
                        if len(retrieved_txt) == args.retrieve_K:
                            output = model.caption_ensemble(retrieved_txt)
                        else:
                            retrieved_txt= [D['caption'] for D in client_backup2.query(image=img_path)[:args.retrieve_K]]
                            assert len(retrieved_txt) == args.retrieve_K, (img_path, len(retrieved_txt))
                            output = model.caption_ensemble(retrieved_txt)
                    
                else: #tpt, coop
                    output, output_cap = model.inference(image.cuda(args.gpu, non_blocking=True), caption= None)
                    # entropy = avg_entropy(output)
                    # print("entropy ", entropy)
                    # if entropy > 1 and output_cap:
                    #     # print(output.shape, output_cap.shape)
                    #     # measure accuracy and record loss
                    #     output = torch.cat([output, output_cap])
        # output = torch.mean(output_cap, dim=0, keepdim=True)
        acc1, acc2, acc3, acc4, acc5 = accuracy(output, target, topk=(1, 2, 3, 4, 5), caption=None, logger=logger, args=args)
        # print(acc1, acc2, acc3, acc4, acc5)
        top1.update(acc1[0], image.size(0))
        top2.update(acc2[0], image.size(0))
        top3.update(acc3[0], image.size(0))
        top4.update(acc4[0], image.size(0))
        top5.update(acc5[0], image.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0:
            progress.display(i)
        

    progress.display_summary()
    return [top1.avg, top2.avg, top3.avg, top4.avg, top5.avg]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning')
    parser.add_argument('data', metavar='DIR', help='path to dataset root')
    parser.add_argument('--test_sets', type=str, default='A/R/V/K/I', help='test dataset (multiple datasets split by slash)')
    parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='RN50')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
    parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('-p', '--print-freq', default=200, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--tpt', action='store_true', default=False, help='run test-time prompt tuning')
    parser.add_argument('--tpt_text', action='store_true', default=False, help='run test-time prompt tuning')
    parser.add_argument('--selection_p', default=0.5, type=float, help='confidence selection percentile')
    parser.add_argument('--tta_steps', default=1, type=int, help='test-time-adapt steps')
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')
    parser.add_argument('--ctx_init', default=None, type=str, help='init tunable prompts')
    parser.add_argument('--cocoop', action='store_true', default=False, help="use cocoop's output as prompt initialization")
    parser.add_argument('--load', default=None, type=str, help='path to a pre-trained coop/cocoop')
    parser.add_argument('--seed', type=int, default=0)
    
    parser.add_argument('--loss', default="both", type=str, help='Either in [both, entropy, cosine]')
    parser.add_argument('--retrieve_K', default=1, type=int, help='1,2,4,8')
    parser.add_argument('--debug', action='store_true', default=False, )
    parser.add_argument('--ours', action='store_true', default=False, help='')
    
    parser.add_argument('--caption_ensemble', action='store_true', default=False, )
    

    main()