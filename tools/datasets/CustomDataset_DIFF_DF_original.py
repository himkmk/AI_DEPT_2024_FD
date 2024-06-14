import os
import torch
import torchvision.transforms as T
# from torchvision.io import read_image
from PIL import Image
import random
import numpy as np
import albumentations as A
import cv2
from .albu import IsotropicResize, WebpAugmentation

class CustomImageDataset(torch.utils.data.Dataset):
    """Dataset for few-shot videos, which returns few-shot tasks. """
    def __init__(self, root_dir, img_size=299, transform=None, mode='train', balance=False):
        self.root_dir = root_dir
        self.dir_list, self.file_list, self.label_list = [], [], []
        self.mode = mode
        self.img_size = img_size
        if mode == 'train':       
            for sub_dir in os.listdir(root_dir):
                for sub_sub_dir in os.listdir(os.path.join(root_dir, sub_dir)):
                    if sub_dir == 'real' and sub_sub_dir in ['Celeb-DF-v2_crop', # 5000
                                                             'DFDC_crop',
                                                             'celeba-hq_cropped',
                                                             'DF_FFpp_DeepFakeDetection_original_cropped',
                                                             'DF_FFpp_original_cropped',
                                                             'ffhq_cropped',
                                                             'DeeperForensics_crop',
                                                             'DIFF_DF_real',
                                                             'DFMNIST+_crop']:
                        _list = list(os.listdir(os.path.join(root_dir, sub_dir, sub_sub_dir)))

                        if balance:
                            max_len = min(2000, len(_list))
                            _list = random.sample(_list, max_len)

                        for file in _list:
                            self.dir_list.append(os.path.join(sub_dir, sub_sub_dir))
                            self.file_list.append(os.path.join(root_dir, sub_dir, sub_sub_dir, file))
                            self.label_list.append(0)

                    elif sub_dir == 'fake' and sub_sub_dir in [ 'Celeb-DF-v2_crop', 
                                                                'DFDC_crop',
                                                                #'DF_FFpp_DeepFakeDetection_cropped',
                                                                #'DF_FFpp_Deepfakes_cropped',
                                                                'DF_FFpp_Face2Face_cropped',
                                                                #'DF_FFpp_FaceShifter_cropped',
                                                                #'DF_FFpp_FaceSwap_cropped',
                                                                'DF_FFpp_NeuralTextures_cropped',
                                                                'DIFF_DF_text2img',
                                                                'DIFF_DF_insight',
                                                                'DIFF_DF_inpainting',
                                                                #'GAN_DF_GAN_ffhq_cropped',
                                                                #'GAN_STYLEMAP_ffhq_cropped',
                                                                #'GAN_eg3d_ffhq_cropped',
                                                                #'GAN_SG2_ffhq_cropped',
                                                                #'GAN_styleswin_CelebAHQ_cropped',
                                                                'CodeFormer_crop29998',
                                                                'CodeFormer_crop29999',
                                                                #'GFPGAN_crop29998', *********
                                                                #'GFPGAN_crop29999', ********
                                                                'NoEnhance_crop29998',
                                                                'NoEnhance_crop29999']:
                        _list = list(os.listdir(os.path.join(root_dir, sub_dir, sub_sub_dir)))

                        if balance:
                            max_len = min(2000, len(_list))
                            _list = random.sample(_list, max_len)


                        for file in _list:
                            self.dir_list.append(os.path.join(sub_dir, sub_sub_dir))
                            self.file_list.append(os.path.join(root_dir, sub_dir, sub_sub_dir, file))
                            self.label_list.append(1)

                    elif sub_dir == 'fake' and sub_sub_dir == 'DFMNIST+_crop':
                        sub_list = [] 
                        for sub_sub_sub_dir in os.listdir(os.path.join(root_dir,sub_dir,sub_sub_dir)):
                            for file in os.listdir(os.path.join(root_dir,sub_dir,sub_sub_dir, sub_sub_sub_dir)):

                                dir_p = os.path.join(sub_dir, sub_sub_dir, sub_sub_sub_dir)
                                file_p = os.path.join(root_dir, sub_dir, sub_sub_dir, sub_sub_sub_dir, file)
                                sub_list.append( [ dir_p, file_p, 1])
                                # self.dir_list.append(os.path.join(sub_dir, sub_sub_dir, sub_sub_sub_dir))
                                # self.file_list.append(os.path.join(root_dir, sub_dir, sub_sub_dir, sub_sub_sub_dir, file))
                                # self.label_list.append(1)
                        if balance:
                             max_len = min(2000, len(sub_list))
                             sub_list = random.sample(sub_list, max_len)
                        for k in sub_list:
                            self.dir_list.append(k[0])
                            self.file_list.append(k[1])
                            self.label_list.append(k[2])

                    elif sub_dir == 'fake' and sub_sub_dir == 'DeeperForensics_crop':
                        sub_list = [] 
                        for sub_sub_sub_dir in os.listdir(os.path.join(root_dir,sub_dir,sub_sub_dir)):
                            for file in os.listdir(os.path.join(root_dir,sub_dir,sub_sub_dir, sub_sub_sub_dir)):
                                dir_p = os.path.join(sub_dir, sub_sub_dir, sub_sub_sub_dir)
                                file_p = os.path.join(root_dir, sub_dir, sub_sub_dir, sub_sub_sub_dir, file)
                                sub_list.append( [ dir_p, file_p, 1])

                                # self.dir_list.append(os.path.join(sub_dir, sub_sub_dir, sub_sub_sub_dir))
                                # self.file_list.append(os.path.join(root_dir, sub_dir, sub_sub_dir, sub_sub_sub_dir, file))
                                # self.label_list.append(1)
                        if balance:
                            max_len = min(2000, len(sub_list))
                            sub_list = random.sample(sub_list, max_len)
                        for k in sub_list:
                            self.dir_list.append(k[0])
                            self.file_list.append(k[1])
                            self.label_list.append(k[2])

        if mode == 'test':
            for sub_dir in os.listdir(root_dir):
                for sub_sub_dir in os.listdir(os.path.join(root_dir, sub_dir)):
                    if sub_dir == 'real' and sub_sub_dir in [
                                                             'Celeb-DF-v2_crop', # 5000
                                                             'DFDC_crop',
                                                             'celeba-hq_cropped',
                                                             'DF_FFpp_DeepFakeDetection_original_cropped',
                                                             'DF_FFpp_original_cropped',
                                                             'DIFF_DF_real',
                                                             'ffhq_cropped',
                                                             'DeeperForensics_crop',
                                                             'DFMNIST+_crop'
                                                              ]:
                        _list = os.listdir(os.path.join(root_dir, sub_dir, sub_sub_dir))
                        for file in _list:
                            self.dir_list.append(os.path.join(sub_dir, sub_sub_dir))
                            self.file_list.append(os.path.join(root_dir, sub_dir, sub_sub_dir, file))
                            self.label_list.append(0)

                    elif sub_dir == 'fake' and sub_sub_dir in [
                                                                #'public_test_images_crop'
                                                                'Celeb-DF-v2_crop',
                                                                'DFDC_crop',
                                                                #'DF_FFpp_DeepFakeDetection_cropped',
                                                                #'DF_FFpp_Deepfakes_cropped',
                                                                'DF_FFpp_Face2Face_cropped',
                                                                #'DF_FFpp_FaceShifter_cropped',
                                                                #'DF_FFpp_FaceSwap_cropped',
                                                                'DF_FFpp_NeuralTextures_cropped',
                                                                'DIFF_DF_text2img',
                                                                'DIFF_DF_insight',
                                                                'DIFF_DF_inpainting',
                                                                # 'GAN_eg3d_ffhq_cropped',
                                                                # 'GAN_SG2_ffhq_cropped',
                                                                # 'GAN_styleswin_CelebAHQ_cropped',
                                                                'CodeFormer_crop29998',
                                                                'CodeFormer_crop29999',
                                                                'GFPGAN_crop29998',
                                                                'GFPGAN_crop29999',
                                                                'NoEnhance_crop29998',
                                                                'NoEnhance_crop29999'
                                                                ]:
                        for file in os.listdir(os.path.join(root_dir,sub_dir,sub_sub_dir)):
                            self.dir_list.append(os.path.join(sub_dir, sub_sub_dir))
                            self.file_list.append(os.path.join(root_dir, sub_dir, sub_sub_dir, file))
                            self.label_list.append(1)

                    elif sub_dir == 'fake' and sub_sub_dir == 'DFMNIST+_crop':
                        for sub_sub_sub_dir in os.listdir(os.path.join(root_dir,sub_dir,sub_sub_dir)):
                            for file in os.listdir(os.path.join(root_dir,sub_dir,sub_sub_dir, sub_sub_sub_dir)):
                                self.dir_list.append(os.path.join(sub_dir, sub_sub_dir, sub_sub_sub_dir))
                                self.file_list.append(os.path.join(root_dir, sub_dir, sub_sub_dir, sub_sub_sub_dir, file))
                                self.label_list.append(1)

                    elif sub_dir == 'fake' and sub_sub_dir == 'DeeperForensics_crop':
                        for sub_sub_sub_dir in os.listdir(os.path.join(root_dir,sub_dir,sub_sub_dir)):
                            for file in os.listdir(os.path.join(root_dir,sub_dir,sub_sub_dir, sub_sub_sub_dir)):
                                self.dir_list.append(os.path.join(sub_dir, sub_sub_dir, sub_sub_sub_dir))
                                self.file_list.append(os.path.join(root_dir, sub_dir, sub_sub_dir, sub_sub_sub_dir, file))
                                self.label_list.append(1)

        self.albu_pre_train_before = A.Compose([
            A.OneOf([
                IsotropicResize(max_side=self.img_size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                IsotropicResize(max_side=self.img_size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                IsotropicResize(max_side=self.img_size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
            ], p=1),
            A.OneOf([
                A.RandomScale(scale_limit=-0.3,interpolation=0, p=1),
                A.RandomScale(scale_limit=-0.3,interpolation=1, p=1),
                A.RandomScale(scale_limit=-0.3,interpolation=2, p=1),
                A.RandomScale(scale_limit=-0.3,interpolation=3, p=1),
                A.RandomScale(scale_limit=-0.3,interpolation=4, p=1),
            ], p=1),      
        ])

        self.albu_pre_train = A.Compose([
            A.ImageCompression(quality_lower=60, quality_upper=100, p=0.2),
            # Add webp augmentation
            WebpAugmentation(p=0.2, qf_range=[30, 90]),
            # 
            A.GaussNoise(p=0.3),
            A.HorizontalFlip(),
            A.OneOf([
                IsotropicResize(max_side=self.img_size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                IsotropicResize(max_side=self.img_size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                IsotropicResize(max_side=self.img_size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
            ], p=1),
            A.PadIfNeeded(min_height=self.img_size, min_width=self.img_size, border_mode=cv2.BORDER_CONSTANT),
            A.OneOf([A.RandomBrightnessContrast(), A.FancyPCA(), A.HueSaturationValue()], p=0.4),
            A.ToGray(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=5, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            A.ImageCompression(quality_lower=60, quality_upper=100, p=0.2)
        ])

        self.albu_pre_val = A.Compose([
            IsotropicResize(max_side=self.img_size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
            A.PadIfNeeded(min_height=self.img_size, min_width=self.img_size, border_mode=cv2.BORDER_CONSTANT),
        ])

        self.imagenet_norm = T.Compose([
            T.ToPILImage(),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.299, 0.225)),
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
            image = self.albu_pre_train_before(image=image)['image']
            image = self.albu_pre_train(image=image)['image']
        else:
            image = self.albu_pre_val(image=image)['image']
        image = self.imagenet_norm(image)
        
        label = self.label_list[index]

        return image, label, dir


if __name__ == "__main__":
    CID = CustomImageDataset('temp')
