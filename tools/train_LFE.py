import os
import numpy as np
import argparse
import time
import random
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 

from tools.models.superresolution.LFE_arch import get_Basic_LFE_RRDBx4, get_Contrastive_LFE_RRDBx4, LFE_Backbone_Merged
from tools.datasets.constant import CONSTANT_policelab_dataroot

# dataset
from tools.datasets.CustomDataset_DIFF_DF import CustomImageDataset
from tools.util import get_model_dir
from tools.test import validate_model


# enable tf32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def parse_args() -> argparse.Namespace:
    
    # basic args
    parser = argparse.ArgumentParser(description='Police Lab Classification model')
    parser.add_argument('--gpus', type=lambda s: [int(item) for item in s.split(',')], required=True, help='gpu ids')
    parser.add_argument('--exp', required=True, help='exp')
    parser.add_argument('--res', type=int, required=True, help='resolution')
    parser.add_argument('--model', type=str, required=True, help='classification model')
    parser.add_argument('--lr', type=float, required=True, help='lr')
    parser.add_argument('--weight_decay', type=float, required=True, help='decay')
    parser.add_argument('--epoch', type=int, required=True, help='epoch')
    parser.add_argument('--batch', type=int, required=True, help='batch')
    parser.add_argument('--log_freq', type=int, required=True, help='log_freq')
    parser.add_argument('--manual_seed', type=int, default=42, help='seed')
    parser.add_argument('--resume_weights', type=str, help='resume_weights')
    
    # sr module args
    parser.add_argument('--lfe_mode', type=str, help='basic | contrastive')
    parser.add_argument('--lfe_unnormalize_mode', type=str, help='imagenet | batchnorm | none')
    parser.add_argument('--lfe_rrdb_bic_ckpt', type=str, help='rrdb_ckpt')
    parser.add_argument('--lfe_rrdb_real_ckpt', type=str, default=None, help='rrdb_ckpt')
    parser.add_argument('--lfe_n_body_blocks', type=int, default=23, help='n_body_blocks')
    
    
    parser.add_argument('--balance_train', type=bool, default=True, help='balance_train')
    
    args = parser.parse_args()
    

    # Assumes basic lfe utilizes only bicubic rrdb
    if args.lfe_mode == 'contrastive':
        assert args.lfe_rrdb_real_ckpt is not None, 'lfe_rrdb_real_ckpt must be provided if lfe_mode is contrastive'



    model_save_dir = get_model_dir(args)
    os.makedirs(model_save_dir,exist_ok=True)
    os.makedirs(os.path.join(model_save_dir,'confidence_graph'),exist_ok=True)
    os.makedirs(os.path.join(os.path.join(model_save_dir,'confidence_graph'),'fake'),exist_ok=True)
    os.makedirs(os.path.join(os.path.join(model_save_dir,'confidence_graph'),'real'),exist_ok=True)
    #need to print args?
    
    
    
    
    return args


def main(args):
    writer = SummaryWriter(get_model_dir(args)) # tensorboard
    
    if args.manual_seed is not None:
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.manual_seed)
        random.seed(args.manual_seed)
    
    #======== Metrics initialization ========
    best_acc = 0
    img_size = 299
    #======== Model init =========
    if args.model == "resnet50":
        from torchvision.models import resnet50
        model = resnet50(weights="IMAGENET1K_V2")
        model.fc = nn.Linear(2048, 1, bias=True)
    elif args.model == "effb4":
        from torchvision.models import efficientnet_b4
        model = efficientnet_b4(weights="IMAGENET1K_V1")
        model.classifier[1] = nn.Linear(1792, 1, bias=True)
    elif args.model == "effb5":
        from torchvision.models import efficientnet_b5
        model = efficientnet_b5(weights="IMAGENET1K_V1")
        model.classifier[1] = nn.Linear(2048, 1, bias=True)
    elif args.model == "effb6":
        from torchvision.models import efficientnet_b6
        model = efficientnet_b6(weights="IMAGENET1K_V1")
        model.classifier[1] = nn.Linear(2304, 1, bias=True)
    elif args.model == "xception":
        from .models.xception_model import Xception
        model = Xception(num_classes=1)
    elif args.model == 'clip':
        from .models.clip_model import CLIPModel
        model = CLIPModel()
        img_size = 224 
    elif args.model == 'cross_efficient_vit':
        from .models.cross_efficient_vit import CrossEfficientViT
        model = CrossEfficientViT()
    else:
        raise NotImplementedError
    
    
    
    # get input dim of model
    model_body_excludefirst = model.features[1:]
    input_dim = 48  # hard coded for now
    
    # connecting model
    if args.lfe_mode == 'basic':
        lfe = get_Basic_LFE_RRDBx4(rrdb_ckpt_path=None, unnormalize_mode='none', n_body_blocks=args.lfe_n_body_blocks, out_dim=input_dim)
    elif args.lfe_mode == 'contrastive':
        lfe = get_Contrastive_LFE_RRDBx4(rrdb_ckpt_path_bic=None, rrdb_ckpt_path_realworld=None, unnormalize_mode='none', n_body_blocks=args.lfe_n_body_blocks, out_dim=input_dim)
    else:
        raise NotImplementedError
    
    model = LFE_Backbone_Merged(lfe, model_body_excludefirst,  model.classifier)
    
    
            
    print(args)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    start_epoch = 0
    model_save_dir = get_model_dir(args)

    #======== Data =======
    
    val_loader = DataLoader(CustomImageDataset(root_dir=f"{CONSTANT_policelab_dataroot}/policelab_gen_dataset_light/test/", img_size= img_size, mode='test'), batch_size=args.batch, shuffle=False)
    # scheduler?

    if args.resume_weights:
        path = os.path.join(model_save_dir, args.resume_weights)
        if os.path.isfile(path):
            print("=> loading weight '{}'".format(path))
            checkpoint = torch.load(path)
            pre_weight = checkpoint['state_dict']
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']
            model.load_state_dict(pre_weight, strict=True)
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded weight '{}'".format(path))
        else:
            raise Exception("=> no weight found at '{}'".format(args.resume_weights))


    
    
    model = nn.DataParallel(model,device_ids=[i for i in range(torch.cuda.device_count())])
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters() ,args.lr, weight_decay=args.weight_decay)
    
    train_loss, train_acc = 0, 0
    #======== Training =======
    for epoch in range(start_epoch, args.epoch):
        train_loader = DataLoader(CustomImageDataset(root_dir=f"{CONSTANT_policelab_dataroot}/policelab_gen_dataset_light/train/", img_size= img_size, balance=args.balance_train), batch_size=args.batch, num_workers=20, shuffle=True)
        train_loss, train_acc = train_model(
            args = args,
            train_loader = train_loader,
            model = model,
            optimizer = optimizer,
            epoch = epoch,
            model_save_dir = model_save_dir
        )


        writer.add_scalar('train loss',train_loss,epoch+1)
        writer.add_scalar('train acc',train_acc,epoch+1)
        if (epoch % (5 if args.balance_train else 1) == 0) and epoch > 1:
            ap, r_acc0, f_acc0, acc0 = validate_model(args,val_loader, model, model_save_dir, epoch, 'test')
            writer.add_scalar('val ap', ap, epoch+1)
            writer.add_scalar('r_acc0', r_acc0, epoch+1)
            writer.add_scalar('f_acc0', f_acc0, epoch+1)
            writer.add_scalar('acc0', acc0, epoch+1)
        if True: #args.TRAIN.save_models:
            filename_model = os.path.join(model_save_dir,'{}.pth'.format(epoch))
            print('Saving checkpoint to: ' + filename_model)

            torch.save(
                {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_acc' : best_acc,
                    'optimizer': optimizer.state_dict(),
                    # 'scheduler': scheduler.state_dict()
                },
                filename_model
            )
            

class Tic():
    """
    Basic Usage:
    tic = Tic()
    print(tic()) # 0.0
    print(tic()) # 0.1
    
    """
    def __init__(self, verbose=True):
        self.tic = time.time()
        self.verbose = verbose

        
    def __call__(self, _str):
        elpased_time = time.time() - self.tic
        self.tic = time.time()
        if self.verbose:
            print(_str, elpased_time)
        return elpased_time
    
    def reset(self):
        self.tic = time.time()


def train_model(
    args: argparse.Namespace,
    train_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    model_save_dir: str
):


    model.train()
    bce_criterion = nn.BCEWithLogitsLoss().cuda()
    total_loss, qry_loss = 0, 0
    total_acc, qry_acc = 0, 0
    numOfQ = 0
    runtime = 0
    s_time = time.time()
    tic = Tic(verbose=False)

    
    for i, (img, label, _) in enumerate(train_loader):
        t0 = time.time()
        
        tic("start")
        img, label = img.cuda().float(), label.cuda().float().unsqueeze(1)
        tic("image load done")
        
        numOfQ += img.shape[0]
        
        optimizer.zero_grad(set_to_none=True)
        tic("zero_grad_done ")
        
        pred = model(img)
        tic("model done")
        
        loss = bce_criterion(pred, label)
        tic("loss done")
        
        loss.backward()
        tic("backward done")
        
        optimizer.step()
        tic("step done")

        with torch.no_grad():
            qry_loss += loss
            total_loss += loss

            probs = F.sigmoid(pred)
            pred = probs > 0.5
            qry_acc += (pred == label).sum().item()
            total_acc += (pred == label).sum().item()
            t1= time.time()
            runtime += (t1-t0)
            
            if i % args.log_freq == 0:
                print('Epoch:[{}][{}/{}]\t'
                      'loss/ACC [{},{:5.3f}]\t'
                      'Runtime:[{}]'
                      .format(epoch, i,len(train_loader),
                              loss, qry_acc/numOfQ,
                              runtime),flush=True)
                qry_loss = 0
                qry_acc = 0
                numOfQ = 0
                runtime = 0
        
        tic("end")
        tic.reset()
        
        
    total_loss = total_loss / len(train_loader)
    total_acc = total_acc / len(train_loader.dataset) 
    print('Epoch {}: Loss / ACC / epoch runtime [{:5.5f},{:5.3f},{}]'.format(
        epoch, total_loss,total_acc, time.time() - s_time))

    return qry_loss, qry_acc

if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpus)
    main(args)