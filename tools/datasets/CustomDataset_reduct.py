import os
import torch
import torchvision.transforms as T
# from torchvision.io import read_image
from PIL import Image
import random
import numpy as np
import albumentations as A
import cv2

class CustomImageDataset(torch.utils.data.Dataset):
    """Dataset for few-shot videos, which returns few-shot tasks. """
    def __init__(self, root_dir, resolution, transform=None, mode='train'):
        self.root_dir = root_dir
        self.resolution = resolution
        self.dir_list, self.file_list, self.label_list = [], [], []
        self.mode = mode
        if mode == 'train':            
            for sub_dir in os.listdir(root_dir):
                for sub_sub_dir in os.listdir(os.path.join(root_dir, sub_dir)):
                    if sub_dir == 'real' and sub_sub_dir in ['Celeb-DF-v2_crop', # 5000
                                                             'DFDC_crop',
                                                             'celeba-hq_cropped',
                                                             'DF_FFpp_DeepFakeDetection_original_cropped',
                                                             'DF_FFpp_original_cropped']:  # 1253363
                        _list = os.listdir(os.path.join(root_dir, sub_dir, sub_sub_dir))
                        # if len(_list) > 70000:
                        #     _list = random.sample(_list, 70000)
                        for file in _list:
                            self.dir_list.append(os.path.join(sub_dir, sub_sub_dir))
                            self.file_list.append(os.path.join(root_dir, sub_dir, sub_sub_dir, file))
                            self.label_list.append(0)
                    elif sub_dir == 'fake' and sub_sub_dir in ['Celeb-DF-v2_crop',
                                                                'DFDC_crop',
                                                                'DFMNIST+_crop/blink',
                                                                'DFMNIST+_crop/embarrass',
                                                                'DFMNIST+_crop/left_slope',
                                                                'DFMNIST+_crop/mouth',
                                                                'DFMNIST+_crop/nod',
                                                                'DFMNIST+_crop/right_slope',
                                                                'DFMNIST+_crop/smile',
                                                                'DFMNIST+_crop/surpise',
                                                                'DFMNIST+_crop/up',
                                                                'DFMNIST+_crop/yaw',
                                                                'DF_FFpp_DeepFakeDetection_cropped',
                                                                'DF_FFpp_Deepfakes_cropped',
                                                                'DF_FFpp_Face2Face_cropped',
                                                                'DF_FFpp_FaceShifter_cropped',
                                                                'DF_FFpp_FaceSwap_cropped',
                                                                'DF_FFpp_NeuralTextures_cropped']:
                        for file in os.listdir(os.path.join(root_dir,sub_dir,sub_sub_dir)):
                            self.dir_list.append(os.path.join(sub_dir, sub_sub_dir))
                            self.file_list.append(os.path.join(root_dir, sub_dir, sub_sub_dir, file))
                            self.label_list.append(1)
        else:
            '''
                for test, 
                1. Eg3d_ffhq, stylemap_ffhq, GAN_SC_xl_imagenet
                2. SG2_afhqcat, dog, wild , DF_GAN_afhqcat, dog, wild 
            '''
            for sub_dir in os.listdir(root_dir):
                for sub_sub_dir in os.listdir(os.path.join(root_dir, sub_dir)):
                    if sub_dir == 'fake':
                        for file in os.listdir(os.path.join(root_dir,sub_dir,sub_sub_dir)):
                            self.dir_list.append(os.path.join(sub_dir, sub_sub_dir))
                            self.file_list.append(os.path.join(root_dir, sub_dir, sub_sub_dir, file))
                            self.label_list.append(1)
                    elif sub_dir == 'real':
                        for file in os.listdir(os.path.join(root_dir,sub_dir,sub_sub_dir)):
                            self.dir_list.append(os.path.join(sub_dir, sub_sub_dir))
                            self.file_list.append(os.path.join(root_dir, sub_dir, sub_sub_dir, file))
                            self.label_list.append(0)


        self.albu_pre_train = A.Compose([
            A.OneOf([
                A.ImageCompression(quality_lower=50, quality_upper=95, compression_type=0, p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.GaussNoise(var_limit=(3.0, 10.0), p=1.0),
                A.ToGray(p=1.0),
            ], p=0.5),
            A.Flip(p=0.33),
        ], p=1.0)
        self.albu_pre_val = A.Compose([
            A.PadIfNeeded(min_height=self.resolution, min_width=self.resolution, p=1.0),
            A.CenterCrop(height=self.resolution, width=self.resolution, p=1.0),
        ], p=1.0)
        self.imagenet_norm = T.Compose([
            T.ToPILImage(),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        

    def __len__(self):
        return len(self.file_list)

    def len_per_dir(self):
        unique_dir = list(set(self.dir_list))
        unique_dir.sort()
        len_dict = {}
        for dir in unique_dir:
            len_dict[dir] = self.dir_list.count(dir)

        return len_dict

    def __getitem__(self, index):

        dir = self.dir_list[index]
        image = cv2.imread(self.file_list[index])

        if self.mode =='train':
            image = self.albu_pre_train(image=image)['image']
        else:
            image = self.albu_pre_val(image=image)['image']
        image = self.imagenet_norm(image)
        
        label = self.label_list[index]

        return image, label, dir


if __name__ == "__main__":
    CID = CustomImageDataset('temp')
