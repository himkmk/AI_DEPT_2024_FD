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
            
        crop_img = cv2.resize(crop_img, (224, 224))
        
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        crop_pil_image = Image.fromarray(crop_img)
        
        return crop_pil_image, mask
    

def face_crop_bb_gen(pil_image):
    from retinaface import RetinaFace
    
    cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    numpy_array = np.array(cv2_image) # pil image numpy 로 변환
    
    resp = RetinaFace.detect_faces(numpy_array) # 얼굴 detect
    if type(resp) == tuple: # 이미지에 얼굴 없는 경우 !!!!!!!!!!!!!!!!!!
        return pil_image, 0,0,0,0
    else:
        x1,y1,x2,y2 = resp['face_1']["facial_area"]# 얼굴 detect 결과 저장
    
    crop_pil_image = read_image(np.array(pil_image),  x2-x1 , y2-y1, x1, y1) # 얼굴 자른 후에, 224, 224 resize 이미지 반환
    numpy_array = np.array(crop_pil_image)
    crop_cv2_image = cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR)
    
    return crop_cv2_image, (x1, y1, x2, y2)


def ff(model, file_path):
    
    #img = Image.open(file_path)
    #crop_cv2_image, mask = face_crop_bb_gen(img)
    

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
    #crop_cv2_image = albu_pre_val(image=crop_cv2_image)
    img = cv2.imread(file_path)
    img = albu_pre_val(image=img)['image']

    #cv2.imwrite( os.path.basename(path) + '_test.png', img)
    data = transform(img).unsqueeze(0)
    data = data.cuda()

    model.eval()
    
    assert os.path.isfile(ckpt_path), ckpt_path
    
    prob = model(data)
    prob = F.sigmoid(prob)
    #mask = mask * float(prob)
    #color_map = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
    #result = cv2.addWeighted(img, 0.5, color_map, 0.5, 0)
    #cv2.imwrite('result.png', result)
    
    return prob, None#result



model_name ='efficientnet_b5'
ckpt_path = '../model_ckpt/exp_4/49.pth'
data_path = '/workspace/data/Yoon_origin_jpg_crop'
#data_path = '/workspace/data/Yoon_video_frames_hq_crop'
#data_path = '/workspace/data/policelab_gen_dataset_light/test/fake/GFPGAN_crop29998'

if model_name == 'efficientnet_b5':
    from torchvision.models import efficientnet_b5
    model = efficientnet_b5(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(2048, 1, bias=True)
    img_input_size = 299

elif model_name == 'clip':
    from models.clip_model import CLIPModel
    model = CLIPModel()
    img_input_size = 224

model = nn.DataParallel(model, device_ids=[0])

checkpoint = torch.load(ckpt_path)
model.load_state_dict(checkpoint['state_dict'])

for i in sorted(os.listdir( data_path)):

    path = os.path.join(data_path, i) #/workspace/data/Yoon_origin_jpg_crop/' + str(i) + '.jpg'
    prob, result = ff(model, path)

    print(i)
    print(prob)
    print()



#/mnt/hdd0/byeongcheol/