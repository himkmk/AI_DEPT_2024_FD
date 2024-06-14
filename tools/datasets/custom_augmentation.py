from io import BytesIO
from PIL import Image
import numpy as np
import torch

def WebpCompression(image, quality=50):
    image = Image.fromarray(image)
    outputIoStream = BytesIO()
    image.save(outputIoStream, format='WEBP', quality=quality)
    outputIoStream.seek(0)
    return np.array(Image.open(outputIoStream))

def JPEGCompression(image, qf=75):
    outputIoStream = BytesIO()
    image.save(outputIoStream, "JPEG", quality=qf, optimice=True)
    outputIoStream.seek(0)
    return Image.open(outputIoStream)


def _unnormalize_imagenetnorm(image, mode='forward'):
    """
    # 2024 05 03 mklee
    
    In order to add some modules that does not have normalization.
    Since default dataset (CustomDataset) normalizes the image, we need to unnormalize it / and re-normalize it, before and after the additional module.
    
    Given bchw torch.tensor, and an imagenet-normalization T.Normalize((0.485, 0.456, 0.406), (0.229, 0.299, 0.225)),
    
    if mode=='forward':
        unnormalize this, in cuda. Assume images are already in cuda.
    if mode=='backward':
        normalize this, in cuda. Assume images are already in cuda.
    
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
    
    if mode.lower()=='forward':        return image * std + mean
    elif mode.lower()=='backward':     return (image - mean) / std
    else:                              raise ValueError("mode should be either 'forward' or 'backward'")

    





if __name__ == "__main__":
    import cv2
    
    fn = "./result.jpg"
    
    img = cv2.imread(fn)
    
    for qf in [0.1, 1, 5, 10, 30, 50, 70, 90]:
        img_compressed = WebpCompression(img, quality=qf)
        cv2.imwrite(f"webp_example_{qf:.1f}.png", img_compressed)