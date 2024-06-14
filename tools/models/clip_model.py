import os
# import clip
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Sequential, Module, Linear


# class CLIPModel_linear(nn.Module):
#     def __init__(self, num_class=1):
#         super().__init__()
        
#         self.clip_model, self.preprocess = clip.load("ViT-L/14", device='cpu', jit=False)
#         self.fc = nn.Linear(768, num_class)

#     def forward(self, image_input, return_feature=False):
#         logits_per_image = self.clip_model.encode_image(image_input)
#         logits_per_image = logits_per_image.detach()
#         if return_feature:
#             return logits_per_image
#         return self.fc(logits_per_image)


from .CLIP import clip as clip_custom
from .cross_efficient_vit import Transformer

class CLIPModel(nn.Module):
    def __init__(self, num_class=1):
        super().__init__()
        
        self.clip_model, self.preprocess = clip_custom.load("ViT-L/14", device='cpu', jit=False)
        
        self.proj = nn.Linear(1024, 196)
        self.model = Transformer(dim=196, depth=1, heads=4, dim_head=196 // 4, mlp_dim=196 * 4)
        self.fc = nn.Linear(196, num_class)

    def forward(self, image_input, return_feature=False):
        image_features = self.clip_model.encode_image(image_input)
        spatial_features = image_features[:, 1:].detach()
        
        logits_per_image = self.model(self.proj(spatial_features))[:, 0]
        
        if return_feature:
            return logits_per_image
        return self.fc(logits_per_image)



class GenDet(nn.Module):
    def __init__(self, num_class=1):
        super().__init__()
        
        self.clip_model, self.preprocess = clip_custom.load("ViT-L/14", device='cpu', jit=False)
        
        self.proj = nn.Linear(1024, 196)
        
        self.teacher = Transformer(dim=196, depth=1, heads=4, dim_head=196 // 4, mlp_dim=196 * 4)
        self.fc_teacher = nn.Linear(196, num_class)
        
        self.student = Transformer(dim=196, depth=1, heads=4, dim_head=196 // 4, mlp_dim=196 * 4)
        
        self.augmenter = Transformer(dim=196, depth=1, heads=4, dim_head=196 // 4, mlp_dim=196 * 4)
        
        self.classifier = nn.Linear(196, num_class)
        
    def forward(self, image_input, phase='classifier', return_feature=False):
        
        assert phase in ['teacher', 'student', 'augmenter', 'classifier']
        
        # feature extraction
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
        # spatial_features = image_features[:, 1:].detach()
        spatial_features = image_features[:].detach()
        
        if phase == 'teacher':
            feat = self.proj(spatial_features)
            logits_per_image = self.fc_teacher(self.teacher(feat)[:, 0])
            
            return logits_per_image
            
        elif phase == 'student':
            feat = self.proj(spatial_features).detach()
            
            z_t = self.teacher(feat)[:, 0]
            z_s = self.student(feat)[:, 0]
            
            return z_t, z_s
        
        elif phase == 'augmenter':
            feat = self.proj(spatial_features).detach()
            feat = self.augmenter(feat)
            
            z_t = self.teacher(feat)[:, 0]
            z_s = self.student(feat)[:, 0]
            
            return z_t, z_s
        
        elif phase == 'classifier':
            feat = self.proj(spatial_features).detach()
            
            z_t = self.teacher(feat)[:, 0]
            z_s = self.student(feat)[:, 0]
            
            return self.classifier((z_t - z_s).square())
            
            
            
        # if return_feature:
        #     return logits_per_image
        # return self.fc(logits_per_image)
