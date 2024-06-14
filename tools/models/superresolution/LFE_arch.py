import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import tqdm
import copy
from tools.datasets.custom_augmentation import _unnormalize_imagenetnorm
import tools.models.superresolution.rrdbnet_arch as RRDBNet_arch
from tools.models.superresolution.constant import _CONSTANT_rrdb_realworld_ckpt, _CONSTANT_rrdb_bicubic_ckpt

"""
Code-base for Low-level Feature Extractor (LFE).
currently implemented on RRBD
"""


class Basic_LFE_RRDBx4(nn.Module):
    
    def __init__(self, rrdb_module, unnormalize_mode='imagenet', n_body_blocks=8, out_dim=64):

        super(Basic_LFE_RRDBx4, self).__init__()
        
        
        # modules
        self.conv_first = copy.deepcopy(rrdb_module.conv_first)
        self.body = copy.deepcopy(rrdb_module.body)[:n_body_blocks]
        self.conv_body = copy.deepcopy(rrdb_module.conv_body)
        
        
        
        self.final_conv = nn.Conv2d(in_channels=self.conv_body.weight.shape[0], out_channels=out_dim, kernel_size=3, stride=1, padding=1)
        
        
        # normalizing and unnormalizing
        self.norm_forward = lambda x: _unnormalize_imagenetnorm(x, mode='forward')
        
        if unnormalize_mode == 'imagenet':
            self.norm_backward = lambda x: _unnormalize_imagenetnorm(x, mode='backward')
        elif unnormalize_mode == 'batchnorm':
            out_dim = self.conv_first.weight.shape[0]
            self.norm_backward = nn.BatchNorm2d(out_dim)
        elif unnormalize_mode == 'none' or unnormalize_mode is None:
            self.norm_backward = lambda x: x
        
        # set requires_grad to False for all parameters except for possibly batchnorms
        for param in self.parameters():
            param.requires_grad = False
        for param in self.conv_body.parameters():
            param.requires_grad = True
        for param in self.final_conv.parameters():
            param.requires_grad = True
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.requires_grad = True
            
    
    
    def forward(self, x):
        
        with torch.no_grad():
            x = self.norm_forward(x)  # rrdbnet gets unnormalized input
            
            feat = x
            feat = self.conv_first(feat)
            body_feat = self.body(feat)
            body_feat = self.conv_body(body_feat)
            
        intermediate_feat = self.norm_backward(body_feat)
        intermediate_feat = self.final_conv(intermediate_feat)
        
        return intermediate_feat
    
class Contrastive_LFE_RRDBx4(nn.Module):
    """
    Identical to above, but utilizes contrastive features of bicubic SR modules and real-world SR modules.
    """
    
    
    def __init__(self, rrdb_module_bic, rrdb_module_realworld, n_body_blocks=8, unnormalize_mode='imagenet', out_dim=64):
        
        super(Contrastive_LFE_RRDBx4, self).__init__()
        
        # modules
        self.conv_first_bi = copy.deepcopy(rrdb_module_bic.conv_first)
        self.body_bi = copy.deepcopy(rrdb_module_bic.body)[:n_body_blocks]
        self.conv_body_bi = copy.deepcopy(rrdb_module_bic.conv_body)
        
        self.conv_first_rw = copy.deepcopy(rrdb_module_realworld.conv_first)
        self.body_rw = copy.deepcopy(rrdb_module_realworld.body)[:n_body_blocks]
        self.conv_body_rw = copy.deepcopy(rrdb_module_realworld.conv_body)
        
        self.final_conv = \
            nn.Conv2d(
            self.conv_first_bi.weight.shape[0]+self.conv_first_rw.weight.shape[0], out_dim, 3, 1, 1,
            )
        
        
        # normalizing and unnormalizing
        self.norm_forward = lambda x: _unnormalize_imagenetnorm(x, mode='forward')
        
        if unnormalize_mode == 'imagenet':
            self.norm_backward = lambda x: _unnormalize_imagenetnorm(x, mode='backward')
        elif unnormalize_mode == 'batchnorm':
            out_dim = self.conv_first_bi.weight.shape[0]
            out_dim = self.conv_first_rw.weight.shape[0]  # just in case they are different
            self.norm_backward = nn.BatchNorm2d(out_dim)
            
        elif unnormalize_mode == 'none' or unnormalize_mode is None:
            self.norm_backward = lambda x: x
        
        
        # set requires_grad to False for all parameters except for conv_body/final_conv and possibly batchnorms
        for param in self.parameters():
            param.requires_grad = False
        for param in self.conv_body_rw.parameters():
            param.requires_grad = True
        for param in self.conv_body_bi.parameters():
            param.requires_grad = True
        for param in self.final_conv.parameters():
            param.requires_grad = True
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.requires_grad = True
        
    def forward(self, x):
            
            with torch.no_grad():
                x = self.norm_forward(x)
                
                feat_bi = x
                feat_bi = self.conv_first_bi(feat_bi)
                feat_bi = self.body_bi(feat_bi)
                
                feat_rw = x
                feat_rw = self.conv_first_rw(feat_rw)
                feat_rw = self.body_rw(feat_rw)
            
            body_feat_bi = self.conv_body_bi(feat_bi)
            body_feat_rw = self.conv_body_rw(feat_rw)
            intermediate_feat = torch.cat([body_feat_bi, body_feat_rw], dim=1)
            
            intermediate_feat = self.final_conv(intermediate_feat)
            intermediate_feat = self.norm_backward(intermediate_feat)
            
            return intermediate_feat


def get_Basic_LFE_RRDBx4(rrdb_ckpt_path=None, unnormalize_mode='none', n_body_blocks=8, out_dim=64):
    
    # get default ckpts
    if rrdb_ckpt_path is None:
        rrdb_ckpt_path = _CONSTANT_rrdb_bicubic_ckpt
    
    rrdb_module = RRDBNet_arch.RRDBNet(3, 3)
    try:
        ckpt = torch.load(rrdb_ckpt_path)['params_ema']
    except:
        ckpt = torch.load(rrdb_ckpt_path)['params']
    rrdb_module.load_state_dict(ckpt)
    
    LFE = Basic_LFE_RRDBx4(rrdb_module, unnormalize_mode=unnormalize_mode, n_body_blocks=8, out_dim=out_dim)
    del rrdb_module
    torch.cuda.empty_cache()
    
    return LFE

def get_Contrastive_LFE_RRDBx4(rrdb_ckpt_path_bic=None, rrdb_ckpt_path_realworld=None, unnormalize_mode='none', n_body_blocks=8, out_dim=64):
    
    # get default ckpts
    if  rrdb_ckpt_path_bic is None:        rrdb_ckpt_path_bic = _CONSTANT_rrdb_bicubic_ckpt
    if  rrdb_ckpt_path_realworld is None:  rrdb_ckpt_path_realworld = _CONSTANT_rrdb_realworld_ckpt
    
    
    # load modules
    rrdb_module_bic = RRDBNet_arch.RRDBNet(3, 3)
    try:
        ckpt = torch.load(rrdb_ckpt_path_bic)['params_ema']
    except:
        ckpt = torch.load(rrdb_ckpt_path_bic)['params']
    rrdb_module_bic.load_state_dict(ckpt)
    
    rrdb_module_realworld = RRDBNet_arch.RRDBNet(3, 3)
    try:
        ckpt = torch.load(rrdb_ckpt_path_realworld)['params_ema']
    except:
        ckpt = torch.load(rrdb_ckpt_path_realworld)['params']
    rrdb_module_realworld.load_state_dict(ckpt)
    
    LFE = Contrastive_LFE_RRDBx4(rrdb_module_bic, rrdb_module_realworld, unnormalize_mode=unnormalize_mode, n_body_blocks=8, out_dim=out_dim)
    del rrdb_module_bic
    del rrdb_module_realworld
    torch.cuda.empty_cache()
    
    return LFE

class LFE_Backbone_Merged(nn.Module):
    
    def __init__(self, lfe, backbone, classifier):
        
        super(LFE_Backbone_Merged, self).__init__()
        
        self.lfe = lfe
        self.backbone = backbone
        self.classifier = classifier
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        
        # print(f"inside model, x.shape: {x.shape}, device={x.device}")
        
        x = self.lfe(x)
        x = self.backbone(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        
        return x