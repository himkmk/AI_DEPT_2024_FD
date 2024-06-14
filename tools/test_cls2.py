import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import time
import random
import yaml
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import tqdm
from .datasets.CustomDataset_DIFF_DF import CustomImageDataset
from .util import get_model_dir
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from copy import deepcopy
from tools.datasets.constant import CONSTANT_policelab_dataroot


def parse_args() -> None:
    parser = argparse.ArgumentParser(description='PoliceLab Classification Model')
    parser.add_argument('--gpus', type=lambda s: [int(item) for item in s.split(',')], required=True, help='gpu ids')
    parser.add_argument('--res', type=int, required=True, help='resolution')
    parser.add_argument('--model', type=str, required=True, help='classification model')
    parser.add_argument('--ckpt_used', type=str, required=True, help='best checkpoint')
    parser.add_argument('--exp', type=str, required=True, help='exp')
    parser.add_argument('--manual_seed', type=int, default=42, help='seed')
    parser.add_argument('--batch', type=int, required=True, help='batch')

    args = parser.parse_args()
    # args = DotMap(config)
    # args.gpus = parsed.gpus
    # args.model = parsed.model
    # args.ckpt_used = parsed.ckpt_used
    # args.exp = parsed.exp
    # args.manual_seed = parsed.manual_seed
    
    return args


def main(args):

    if args.manual_seed is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        random.seed(args.manual_seed)
    
    #======== Model init =========
    if args.model == "resnet50":
        from torchvision.models import resnet50
        model = resnet50(weights="IMAGENET1K_V2")
        model.fc = nn.Linear(2048, 1, bias=True)
    elif args.model == "effb4":
        from torchvision.models import efficientnet_b4
        model = efficientnet_b4(weights="IMAGENET1K_V1")
        model.classifier[1] = nn.Linear(1792, 1, bias=True)
    elif args.model == "xception":
        from .models.xception_model import Xception
        model = Xception(num_classes=2)
    elif args.model == 'clip':
        from .models.clip_model import CLIPModel
        model = CLIPModel()
    else:
        raise NotImplementedError
    
    root_model = get_model_dir(args)


    model = nn.DataParallel(model,device_ids=args.gpus)
    model = model.cuda()
    
    if args.ckpt_used is not None:
        filepath = os.path.join("weights/ckpt_iter.pth.tar")
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded model weight '{}'".format(filepath),flush=True)
    else:
        print("=> Not loading anything",flush=True)
    
    test_loader = DataLoader(CustomImageDataset(root_dir=f"{CONSTANT_policelab_dataroot}/policelab_gen_dataset_light/test/", mode='test'), batch_size=args.batch, num_workers=20, shuffle=False)

    # ====== Test  ======
    ap, r_acc0, f_acc0, acc0 =validate_model(
        args= args,
        val_loader=test_loader,
        model=model,
        model_save_dir=os.path.join(root_model,'inference'),
        epoch = 0,
        mode = 'test'
    )

def calculate_acc(y_true, y_pred, thres):
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > thres)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > thres)
    acc = accuracy_score(y_true, y_pred > thres)
    return r_acc, f_acc, acc   

def find_best_threshold(y_true, y_pred):
    "We assume first half is real 0, and the second half is fake 1"

    N = y_true.shape[0]

    if y_pred[0:N//2].max() <= y_pred[N//2:N].min(): # perfectly separable case
        return (y_pred[0:N//2].max() + y_pred[N//2:N].min()) / 2 

    best_acc = 0 
    best_thres = 0 
    for thres in y_pred:
        temp = deepcopy(y_pred)
        temp[temp>=thres] = 1 
        temp[temp<thres] = 0 

        acc = (temp == y_true).sum() / N  
        if acc >= best_acc:
            best_thres = thres
            best_acc = acc 
    
    return best_thres

def validate_model(
    args: argparse.Namespace,
    val_loader: torch.utils.data.DataLoader,
    model : torch.nn.Module,
    model_save_dir : str,
    epoch : int,
    mode : str
):

    print("===> Start testing")

    model.eval()
    runtime = 0
    already_print_class = []
    
    len_per_dict = CustomImageDataset(root_dir=f"{CONSTANT_policelab_dataroot}/policelab_gen_dataset_light/test/", mode='test').len_per_dir()
    sub_dir_key, sub_dir_value = np.array(list(len_per_dict.keys())), np.array(list(len_per_dict.values()))
    acc_per_sub_dir = np.zeros(len(sub_dir_key))
    y_true, y_pred = [], []
    with torch.no_grad():
        for i, (img, label, dir_name) in enumerate(tqdm.tqdm(val_loader)):
            img, label = img.cuda(), label.cuda()
            pred = model(img) # 4 x 2
            pred1 = torch.argmax(pred, dim=1)
            pred = F.softmax(pred, dim=1)[:,1]
            y_pred.extend([pred.cpu()])
            y_true.extend([label.cpu()])

            for dir_, pred_1, label_ in zip(dir_name,pred1,label):
                acc_per_sub_dir[np.where(sub_dir_key == dir_)[0]] += int(pred_1 == label_)
                
        for i, (key, value) in enumerate(zip(sub_dir_key,sub_dir_value)):
            acc_per_sub_dir[i] /= value
            print("len of {} = {}, Acc = {}".format(key, value, acc_per_sub_dir[i]))  
        
        y_true, y_pred = np.array(y_true), np.array(y_pred)

        # ap = average_precision_score(y_true, y_pred)
        r_acc0, f_acc0, acc0 = calculate_acc(y_true, y_pred, 0.5)
        
    print("Epoch:{} r_acc0={}, f_acc0={}, acc0={}".format(epoch, r_acc0, f_acc0, acc0))
    return ap, r_acc0, f_acc0, acc0

if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpus)
    main(args)