import torchvision.transforms as T
import cv2
import numpy as np
from PIL import Image
import torch.nn as nn
import os
import torch
import albumentations as A
import torch.nn.functional as F
from datasets.albu import IsotropicResize

def read_image(numpy_array, w, h, x, y):
        img  = cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR)
        # Get the shape of input image
        real_h,real_w,c = img.shape
        mask = np.zeros((real_h, real_w))
        if x == 0 and y ==0 and w == 0 and h == 0:
            w = real_w
            h = real_h
            

        w = int(float(w))
        h = int(float(h))
        x = int(float(x))
        y = int(float(y))
        
        mask[y:y+h,x:x+w] = 1
        
        x = int(x - w*0.15)
        y = int(y - y*0.15)
        w = int(w*1.3)
        h = int(h*1.3)

        # Crop face based on its bounding box
        y1 = 0 if y < 0 else y
        x1 = 0 if x < 0 else x 
        y2 = real_h if y1 + h > real_h else y + h
        x2 = real_w if x1 + w > real_w else x + w
        crop_img = img[y1:y2,x1:x2,:]
            
        crop_img = cv2.resize(crop_img, (224, 224), interpolation=cv2.INTER_AREA)
        
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        crop_pil_image = Image.fromarray(crop_img)
        
        return crop_pil_image, mask
    

def face_crop_bb_gen(pil_image, detector):
    
    cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    numpy_array = np.array(cv2_image) # pil image numpy 로 변환
    
    resp = detector.detect_faces(numpy_array)
    if type(resp) == tuple: # 이미지에 얼굴 없는 경우 !!!!!!!!!!!!!!!!!!
        return pil_image, 0,0,0,0
    else:
        x1,y1,x2,y2 = resp['face_1']["facial_area"]# 얼굴 detect 결과 저장
    
    crop_pil_image, mask = read_image(np.array(pil_image),  x2-x1 , y2-y1, x1, y1) # 얼굴 자른 후에, 224, 224 resize 이미지 반환
    numpy_array = np.array(crop_pil_image)
    crop_cv2_image = cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR)
    
    return crop_cv2_image, (x1, y1, x2, y2)


# Follow original data preprocesing code: 115.145.135.135:/mnt/hdd0/byeongcheol/yoloface/face_crop.py
def crop_image(image, x1, y1, x2, y2, face_margin=0.3):
    h = y2 - y1 
    w = x2 - x1

    # 얼굴의 주변부위도 포함될 수 있도록 약간 확장
    new_h = h * (1 + face_margin)
    new_w =  w * (1 + face_margin)
    pixel_add_sub_h = int((new_h - h) / 2)
    pixel_add_sub_w = int((new_w - w) / 2)
    
    new_x1 = max(x1 - pixel_add_sub_w, 0)
    new_y1 = max(y1 - pixel_add_sub_h, 0)  
    new_x2 = min(x2 + pixel_add_sub_w, image.shape[1])
    new_y2 = min(y2 + pixel_add_sub_h, image.shape[0])

    bounding_box = [new_x1, new_y1, new_x2, new_y2]
    
    image = image[bounding_box[1]:bounding_box[3],bounding_box[0]:bounding_box[2]]
    
    return image, (x1, y1, x2, y2)



def face_crop_bb_gen_yolo(image, detector):
    
    bboxes, points = detector(image)
    
    x1, y1, x2, y2 = bboxes[0][0] # no batch?
    
    crop_pil_image, mask = crop_image(image, x1, y1, x2, y2)
    crop_cv2_image = np.array(crop_pil_image)

    return crop_cv2_image, (x1, y1, x2, y2)


def ff(model, file_path, detector, img_input_size=224, crop=True):
    
    albu_pre_val = A.Compose([
            IsotropicResize(max_side=100, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
            IsotropicResize(max_side=img_input_size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),

            A.PadIfNeeded(min_height=img_input_size, min_width=img_input_size, border_mode=cv2.BORDER_CONSTANT),
        ], p=1.0)

    transform = T.Compose([
        T.ToPILImage(),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    if crop:
        img = Image.open(file_path)
        
        if type(img) != np.array:
            img = np.array(img)
        
        if img.shape[-1] == 4:
            img = img[:, :, :3] # remove alpha channel
        
        crop_cv2_image, mask = face_crop_bb_gen_yolo(img, detector)
        img = np.array(albu_pre_val(image=crop_cv2_image)['image'])
    else:
        img = cv2.imread(file_path)[:, :, ::-1] # BGR2RGB
        
        if img.shape[-1] == 4:
            img = img[:, :, :3] # remove alpha channel
        
        img = albu_pre_val(image=img)['image']

    img = img[:, :, ::-1]
    cv2.imwrite('temp.png', img)
    
    data = transform(img).unsqueeze(0)
    data = data.cuda()

    model.eval()

    prob = model(data)
    prob = F.sigmoid(prob)
    #mask = mask * float(prob)
    #color_map = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
    #result = cv2.addWeighted(img, 0.5, color_map, 0.5, 0)
    #cv2.imwrite('result.png', result)
    
    return prob, None #result


import sys
if __name__ == "__main__":

    model_name = 'gendet'
    ckpt_path = 'model_ckpt/exp_gendet3_classifier/3.pth'
    # ckpt_path = 'model_ckpt/exp_gendet3_teacher/49.pth'
    
    data_path = sys.argv[1]
    
    # data_path = 'tools/test.png'
    # data_path = '/workspace/data/Yoon_origin_jpg_crop'
    #data_path = '/workspace/data/Yoon_video_frames_hq_crop'
    #data_path = '/workspace/data/policelab_gen_dataset_light/test/fake/GFPGAN_crop29998'

    #======== Model init =========
    if model_name == "resnet50":
        from torchvision.models import resnet50
        model = resnet50(weights="IMAGENET1K_V2")
        model.fc = nn.Linear(2048, 1, bias=True)
    elif model_name == "effb4":
        from torchvision.models import efficientnet_b4
        model = efficientnet_b4(weights="IMAGENET1K_V1")
        model.classifier[1] = nn.Linear(1792, 1, bias=True)
    elif model_name == "effb5":
        from torchvision.models import efficientnet_b5
        model = efficientnet_b5(weights="IMAGENET1K_V1")
        model.classifier[1] = nn.Linear(2048, 1, bias=True)
    elif model_name == "effb6":
        from torchvision.models import efficientnet_b6
        model = efficientnet_b6(weights="IMAGENET1K_V1")
        model.classifier[1] = nn.Linear(2304, 1, bias=True)
    elif model_name == "xception":
        from models.xception_model import Xception
        model = Xception(num_classes=1)
    elif model_name == 'clip':
        from models.clip_model import CLIPModel
        model = CLIPModel()
        img_size = 224 
    elif model_name == 'cross_efficient_vit':
        from models.cross_efficient_vit import CrossEfficientViT
        model = CrossEfficientViT()
    elif model_name == 'gendet':
        from models.clip_model import GenDet
        model = GenDet()
        img_size = 224 
    else:
        raise NotImplementedError

    model = nn.DataParallel(model, device_ids=[0])

    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['state_dict'])


    # Yolo face detector
    from yoloface.face_detector import YoloDetector
    detector = YoloDetector(device="cuda:0")
    

    if os.path.isdir(data_path):
        # if directory is given,
        
        result_dict = {}
        for i in sorted(os.listdir( data_path)):

            path = os.path.join(data_path, i) #/workspace/data/Yoon_origin_jpg_crop/' + str(i) + '.jpg'
            with torch.no_grad():
                prob, result = ff(model, path, detector, img_input_size=img_size)

            print(i, prob)
            result_dict[i] = prob.detach().cpu().item()
        
        with open(f"{data_path}/result.txt", 'w') as f:
            for k in result_dict.keys():
                f.write(f"{k}: \t {result_dict[k]}")
            
    else:
        # if file path is given,
        path = data_path
        prob, result = ff(model, path, detector, img_input_size=img_size)
        
        print(prob)
