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

# dataset
from .datasets.CustomDataset_DIFF_DF import CustomImageDataset
from .util import get_model_dir
from .test import validate_model


def parse_args() -> argparse.Namespace:
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
    parser.add_argument('--last_layer', type=str, default = 'False', help='resume_weights')

    args = parser.parse_args()

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
    
    

    if False:
        path = os.path.join("weights/ckpt_iter.pth.tar")
        if os.path.isfile(path):
            print("=> loading weight '{}'".format(path))
            checkpoint = torch.load(path)['state_dict']
            post_checkpoint = {}
            for key, value in checkpoint.items():
                if 'module.fc.weight' == key or 'module.fc.bias' == key:
                    continue
                post_checkpoint[key] = value
            model.load_state_dict(post_checkpoint,strict=False)
    # if args.model == 'cross_efficient_vit':
    #     path = 'weights/cross_efficient_vit.pth'
    #     if os.path.isfile(path):
    #         print("=> loading weight '{}'".format(path))
    #         checkpoint = torch.load(path)
    #         model.load_state_dict(checkpoint)
            
    print(args)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    start_epoch = 0
    model_save_dir = get_model_dir(args)

    #======== Data =======
    
    val_loader = DataLoader(CustomImageDataset(root_dir="/workspace/data/policelab_gen_dataset_light/test/", img_size= img_size, mode='test'), batch_size=args.batch, shuffle=False)
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

    #========= train only last =======

    if args.last_layer == 'True':
        print('Train only last fc layer!')
        for param in model.parameters():
            param.requires_grad = False

        for param in model.fc.parameters():
            param.requires_grad = True


    model = nn.DataParallel(model,device_ids=args.gpus)
    model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters() ,args.lr, weight_decay=args.weight_decay)
    #======== Training =======
    for epoch in range(start_epoch, args.epoch):
        train_loader = DataLoader(CustomImageDataset(root_dir="/workspace/data/policelab_gen_dataset_light/train/", img_size= img_size, balance=True), batch_size=args.batch, num_workers=10, shuffle=True)
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
        if epoch % 5 == 0:
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

    for i, (img, label, _) in enumerate(train_loader):
        t0 = time.time()
        img, label = img.cuda().float(), label.cuda().float().unsqueeze(1)
        numOfQ += img.shape[0]
        optimizer.zero_grad()
        pred = model(img)
        
        loss = bce_criterion(pred, label)
        
        loss.backward()
        optimizer.step()

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
            
    total_loss = total_loss / len(train_loader)
    total_acc = total_acc / len(train_loader.dataset) 
    print('Epoch {}: Loss / ACC / epoch runtime [{:5.5f},{:5.3f},{}]'.format(
        epoch, total_loss,total_acc, time.time() - s_time))

    return qry_loss, qry_acc

if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpus)

    main(args)