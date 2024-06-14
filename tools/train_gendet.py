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
    
    # GenDet
    parser.add_argument('--phase', type=str, choices=['teacher', 'gendet', 'classifier'])

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
    elif args.model == 'gendet':
        from .models.clip_model import GenDet
        model = GenDet()
        img_size = 224 
    else:
        raise NotImplementedError
    
    
            
    print(args)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    start_epoch = 0
    model_save_dir = get_model_dir(args)

    #======== Data =======
    
    val_loader = DataLoader(CustomImageDataset(root_dir="/workspace/data/policelab_gen_dataset_light/test/", img_size= img_size, mode='test'), batch_size=args.batch, shuffle=False)
    # scheduler?

    # optimizer = torch.optim.AdamW(model.parameters() ,args.lr, weight_decay=args.weight_decay)
    if args.phase == 'teacher':
        optimizer = torch.optim.AdamW(list(model.teacher.parameters()) + list(model.fc_teacher.parameters()), args.lr, weight_decay=args.weight_decay)
    elif args.phase == 'gendet':
        optimizer = torch.optim.AdamW(list(model.student.parameters()) + list(model.augmenter.parameters()), args.lr, weight_decay=args.weight_decay)
        # optimizer = torch.optim.AdamW(list(model.student.parameters()) + list(model.augmenter.parameters()), 1e-4, weight_decay=args.weight_decay)
    elif args.phase == 'classifier':
        optimizer = torch.optim.AdamW(model.classifier.parameters(), args.lr, weight_decay=args.weight_decay)
        # optimizer = torch.optim.AdamW(model.classifier.parameters(), 1e-4, weight_decay=args.weight_decay)
        pass

    model = nn.DataParallel(model,device_ids=args.gpus)
    model = model.cuda()
    
    if args.resume_weights:
        # path = os.path.join(model_save_dir, args.resume_weights)
        path = os.path.join(args.resume_weights)
        if os.path.isfile(path):
            print("=> loading weight '{}'".format(path))
            checkpoint = torch.load(path)
            pre_weight = checkpoint['state_dict']
            # start_epoch = checkpoint['epoch'] + 1
            # best_acc = checkpoint['best_acc']
            model.load_state_dict(pre_weight, strict=True)
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded weight '{}'".format(path))
        else:
            raise Exception("=> no weight found at '{}'".format(path))

    
    #======== Training =======
    for epoch in range(start_epoch, args.epoch):
        
        if args.phase == 'teacher':
            train_loader = DataLoader(CustomImageDataset(root_dir="/workspace/data/policelab_gen_dataset_light/train/", img_size= img_size, balance=True), batch_size=args.batch, num_workers=10, shuffle=True)
            
            train_loss, train_acc = train_cls_model(
                args = args,
                train_loader = train_loader,
                model = model,
                optimizer = optimizer,
                epoch = epoch,
                model_save_dir = model_save_dir,
                phase = 'teacher',
            )
        elif args.phase == 'gendet':
            train_loader_real = DataLoader(CustomImageDataset(root_dir="/workspace/data/policelab_gen_dataset_light/train/", label_type='real', img_size= img_size, balance=True), batch_size=args.batch, num_workers=10, shuffle=True)
            train_loader_fake = DataLoader(CustomImageDataset(root_dir="/workspace/data/policelab_gen_dataset_light/train/", label_type='fake', img_size= img_size, balance=True), batch_size=args.batch, num_workers=10, shuffle=True)
            train_loader = [train_loader_real, train_loader_fake]
            
            train_loss, train_acc = train_gendet_model(
                args = args,
                train_loader = train_loader,
                model = model,
                optimizer = optimizer,
                epoch = epoch,
                model_save_dir = model_save_dir
            )
        elif args.phase == 'classifier':
            train_loader = DataLoader(CustomImageDataset(root_dir="/workspace/data/policelab_gen_dataset_light/train/", img_size= img_size, balance=True), batch_size=args.batch, num_workers=10, shuffle=True)
            
            train_loss, train_acc = train_cls_model(
                args = args,
                train_loader = train_loader,
                model = model,
                optimizer = optimizer,
                epoch = epoch,
                model_save_dir = model_save_dir,
                phase = 'classifier'
            )
        else:
            NotImplementedError
        
        
        writer.add_scalar('train loss',train_loss,epoch+1)
        writer.add_scalar('train acc',train_acc,epoch+1)
        if ((epoch + 1) % 10 == 0 and args.phase != 'gendet') or epoch == (args.epoch - 1):
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
            

def train_gendet_model(
    args: argparse.Namespace,
    # train_loader: torch.utils.data.DataLoader,
    train_loader: list,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    model_save_dir: str
):

    model.train()

    train_loader_real, train_loader_fake = train_loader[0], train_loader[1]

    # real phase
    runtime = 0
    s_time = time.time()
    
    loss_real = 0.
    for i, (img, label, _) in enumerate(train_loader_real):
        t0 = time.time()
        img, label = img.cuda().float(), label.cuda().float().unsqueeze(1)
        optimizer.zero_grad()
        feat_t, feat_s = model(img, phase='student')
        
        # MSE loss for minimizing discrepancy between outputs of teacher and student
        loss = (feat_t - feat_s).square().mean()
        
        loss.backward()
        optimizer.step()

        loss_real += loss.mean()
        t1= time.time()
        runtime += (t1-t0)
        
        if i % args.log_freq == 0:
            print('Epoch:[{}][{}/{}]\t'
                  'loss/real [{}]\t'
                  'Runtime:[{}]'
                  .format(epoch, i, len(train_loader_real),
                          loss_real / (i + 1), runtime), flush=True)
            runtime = 0


    # fake phase
    runtime = 0
    s_time = time.time()
    
    loss_fake = 0.
    for i, (img, label, _) in enumerate(train_loader_fake):
        t0 = time.time()
        img, label = img.cuda().float(), label.cuda().float().unsqueeze(1)
        optimizer.zero_grad()
        feat_t, feat_s = model(img, phase='student')
        
        # maximizing discrepancy between outputs of teacher and student
        normalize = torch.nn.functional.normalize
        margin = 2.0
        loss = torch.nn.functional.relu(margin - torch.abs(normalize(feat_t, dim=-1) - normalize(feat_s, dim=-1)).square().sum(-1)).mean()
        
        loss.backward()
        optimizer.step()

        loss_fake += loss.mean()

        t1= time.time()
        runtime += (t1-t0)
        
        if i % args.log_freq == 0:
            print('Epoch:[{}][{}/{}]\t'
                  'loss/fake [{}]\t'
                  'Runtime:[{}]'
                  .format(epoch, i,len(train_loader_fake),
                          loss_fake / (i + 1), runtime),flush=True)
            runtime = 0
    

    # augmenter phase
    runtime = 0
    s_time = time.time()
    
    loss_augmenter = 0.
    for i, (img, label, _) in enumerate(train_loader_fake):
        t0 = time.time()
        img, label = img.cuda().float(), label.cuda().float().unsqueeze(1)
        optimizer.zero_grad()
        feat_t, feat_s = model(img, phase='augmenter')
        
        # minimize discrepancy between outputs of teacher and student
        loss = (feat_t - feat_s).square().mean()
        
        loss.backward()
        optimizer.step()

        loss_augmenter += loss
        
        t1= time.time()
        runtime += (t1-t0)
        
        if i % args.log_freq == 0:
            print('Epoch:[{}][{}/{}]\t'
                  'loss/augmenter [{}]\t'
                  'Runtime:[{}]'
                  .format(epoch, i,len(train_loader_fake),
                          loss_augmenter / (i + 1), runtime),flush=True)
            runtime = 0
            
    loss_real = loss_real / len(train_loader_real)
    loss_fake = loss_fake / len(train_loader_fake)
    loss_augmenter = loss_augmenter / len(train_loader_fake)
    
    # print('Epoch {} Phase {}: Loss / ACC / epoch runtime [{:5.5f},{:5.3f},{}]'.format(
    #     epoch, 'real', total_loss,total_acc, time.time() - s_time))
    print('Epoch {}: Loss real / fake / augmenter / epoch runtime [{:5.5f},{:5.5f},{:5.3f},{}]'.format(
        epoch, loss_real, loss_fake, loss_augmenter, time.time() - s_time))

    return loss_fake + loss_real + loss_augmenter, 0



def train_cls_model(
    args: argparse.Namespace,
    # train_loader: torch.utils.data.DataLoader,
    train_loader: list,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    model_save_dir: str,
    phase: str  = 'teacher',
):

    model.train()   
    bce_criterion = nn.BCEWithLogitsLoss().cuda()
    total_loss, qry_loss = 0, 0
    total_acc, qry_acc = 0, 0
    numOfQ = 0
    runtime = 0
    s_time = time.time()

    assert phase in ['teacher', 'classifier']

    for i, (img, label, _) in enumerate(train_loader):
        t0 = time.time()
        img, label = img.cuda().float(), label.cuda().float().unsqueeze(1)
        numOfQ += img.shape[0]
        optimizer.zero_grad()
        pred = model(img, phase=phase)
        
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