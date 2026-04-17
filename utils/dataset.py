import pandas as pd
import numpy as np
from PIL import Image
import os

import pandas as pd
import numpy as np
import torch

import cv2

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .octip_segmentation_2d.octip.retina_localizer import RetinaLocalizer, RetinaLocalizationDataset
from .octip_segmentation_2d.octip.pre_processor import PreProcessor
from .data_processing import octip_preprocess_input

# Fonction pour transformer une étiquette en vecteur one-hot encodé
def one_hot_encode(label, num_classes=4):
    one_hot = torch.zeros(num_classes)
    one_hot[label] = 1
    return one_hot

class MARIO_DS_T1(Dataset):
    """This data loader return b-scans uniformly selected based on teh stored indexes."""

    def __init__(self, df,
                 mode = "test",
                 image_size=(512,200), #200x512
                 gray_scale=False, #False [c=3] / True [c=1]
                 root_dir='',
                 processing_octip = True,
                 ):

        self.samples = df
        self.image_size = image_size
        self.gray_scale = gray_scale
        self.mode = mode

        self.targets = self.samples.label.tolist()
        self.root_dir = root_dir
        
        self.processing_octip = processing_octip
        
        #oct preprocessing
        if self.processing_octip : 
            
            model_directory = "./utils/octip_models"

            self.localizer1 = RetinaLocalizer('FPN','efficientnetb6',(384, 384),model_directory = model_directory)
            self.localizer2 = RetinaLocalizer('FPN', 'efficientnetb7', (320, 320),model_directory = model_directory)

            self.preprocessor = PreProcessor(200, min_height = 100, normalize_intensities = True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):


        sample = self.samples.iloc[idx] 

        sample_path_t0 = self.root_dir + sample['image_at_ti']
        sample_path_t1 = self.root_dir + sample['image_at_ti+1']
        
        _label = self.targets[idx]
        #_label = one_hot_encode(_label)

        _case = sample['case']
        
        # OCTIP 
        if self.processing_octip :
            
            # segmenting the retina of two B-scans with the first model 
            _, seg1_imgs = self.localizer1(RetinaLocalizationDataset([sample_path_t0, sample_path_t1], 6, self.localizer1)) #shape (1,384,384,1) 
            # segmenting the retina of two B-scans with the second model
            _, seg2_imgs = self.localizer2(RetinaLocalizationDataset([sample_path_t0, sample_path_t1], 6, self.localizer2)) #shape (1,320,320,1)
            
            seg1_imgs = (seg1_imgs.squeeze()*255).astype(np.uint8)
            seg2_imgs = (seg2_imgs.squeeze()*255).astype(np.uint8)
            
#             img_t0 = self.preprocessor(sample_path_t0, [seg1_imgs[0], seg2_imgs[0]], output_directory='')
#             img_t1 = self.preprocessor(sample_path_t1, [seg1_imgs[1], seg2_imgs[1]], output_directory='')

            try:
                img_t0 = self.preprocessor(sample_path_t0, [seg1_imgs[0], seg2_imgs[0]], output_directory='')
            except Exception as e:
                print(f"An error occurred while processing img_t0: {e}")
                img_t0 = cv2.imread(sample_path_t0 , 0) # Retourner l'image originale en cas d'erreur

            try:
                img_t1 = self.preprocessor(sample_path_t1, [seg1_imgs[1], seg2_imgs[1]], output_directory='')
            except Exception as e:
                print(f"An error occurred while processing img_t1: {e}")
                img_t1 = cv2.imread(sample_path_t1 , 0) # Retourner l'image originale en cas d'erreur
            
        else :

            img_t0 = cv2.imread(sample_path_t0 , 0) #shape = [200,x]
            img_t1 = cv2.imread(sample_path_t1 , 0) #shape = [200,x]
            
        # Resize 
        img_t0 = cv2.resize(img_t0, self.image_size) # opencv resize => image_size (W, H) 
        img_t1 = cv2.resize(img_t1, self.image_size) # opencv resize => image_size (W, H) 

        #  Data Augmentation
        if self.mode == "train":  
            im_aug = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2),
                transforms.RandomPerspective(),
                transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.5)),
            ])
            img_t0 = im_aug(Image.fromarray(img_t0))
            img_t1 = im_aug(Image.fromarray(img_t1))


        
        img_t0 = np.array(img_t0) # Convert to np.array
        img_t1 = np.array(img_t1) # Convert to np.array

        img_t0 = np.array(cv2.normalize(img_t0, None, 0, 255, cv2.NORM_MINMAX), dtype=np.uint8)
        img_t1 = np.array(cv2.normalize(img_t1, None, 0, 255, cv2.NORM_MINMAX), dtype=np.uint8)

        img_t0 = np.expand_dims(img_t0, axis=-1)  # expand dimension for gray scale
        img_t1 = np.expand_dims(img_t1, axis=-1)  # expand dimension for gray scale

        if not self.gray_scale:  # instead of reading image as rgb, we repeat the image for three times
            img_t0 = np.repeat(img_t0, 3, axis=-1)
            img_t1 = np.repeat(img_t1, 3, axis=-1)

        #convert to tensor
        img_t0 = transforms.ToTensor()(img_t0)
        img_t1 = transforms.ToTensor()(img_t1)

        #normalization
        #img_t0 = transforms.Normalize(mean=[0.5], std=[0.5])(img_t0) 
        #img_t1 = transforms.Normalize(mean=[0.5], std=[0.5])(img_t1) 

        return img_t0, img_t1, _label, _case
    

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

class MARIO_DS_T2_V2(Dataset):

    def __init__(self, df,
                 mode = "test",
                 image_size=(512,200), #200x512
                 gray_scale=False, #False [c=3] / True [c=1]
                 root_dir='',
                 processing_octip = True,
                 mae_model = None
                 ):

        self.samples = df
        self.image_size = image_size
        self.gray_scale = gray_scale
        self.mode = mode

        self.targets = self.samples.label.tolist()
        self.root_dir = root_dir
        
        self.mae_model = mae_model
        
        self.processing_octip = processing_octip
        
        # OCTIP
        if self.processing_octip : 
            
            model_directory = "./utils/octip_models"

            self.localizer1 = RetinaLocalizer('FPN','efficientnetb6',(384, 384),model_directory = model_directory)
            self.localizer2 = RetinaLocalizer('FPN', 'efficientnetb7', (320, 320),model_directory = model_directory)

            self.preprocessor = PreProcessor(200, min_height = 100, normalize_intensities = True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):


        sample = self.samples.iloc[idx] 

        sample_path_t0 = self.root_dir + sample['image']
#         sample_path_t1 = "/home/pzhang/Data/database/MARIO_OCTIP_CHALLENGE/" + sample['image']
            
        _label = self.targets[idx]
        #_label = one_hot_encode(_label)

        _case = sample['case']

        if self.processing_octip :
            
            # segmenting the retina of two B-scans with the first model 
            _, seg1_imgs = self.localizer1(RetinaLocalizationDataset([sample_path_t0], 6, self.localizer1)) #shape (1,384,384,1) 
            # segmenting the retina of two B-scans with the second model
            _, seg2_imgs = self.localizer2(RetinaLocalizationDataset([sample_path_t0], 6, self.localizer2)) #shape (1,320,320,1)
            
            seg1_imgs = (seg1_imgs.squeeze()*255).astype(np.uint8)
            seg2_imgs = (seg2_imgs.squeeze()*255).astype(np.uint8)
            
#             img_t0 = self.preprocessor(sample_path_t0, [seg1_imgs[0], seg2_imgs[0]], output_directory='')
            
            try:
            
                img_t0 = self.preprocessor(sample_path_t0, [seg1_imgs, seg2_imgs], output_directory='')
                
            except Exception as e:
                
                print(f"An error occurred while processing img_t0: {e}")
                img_t0 = cv2.imread(sample_path_t0 , 0) # Retourner l'image originale en cas d'erreur
               
            
            
        #image t+1 generation with octip and patch progression model
        img_t1 = Image.fromarray(img_t0)  # img_original.size : (768, 200)

        img_t1 = img_t1.resize((224, 224))
        img_t1 = np.array(img_t1) / 255. # img.shape : (224, 224, 3) 

        # # # instead of reading image as rgb, we repeat the image for three times
        img_t1 = np.expand_dims(img_t1, axis=-1)  # expand dimension for gray scale
        img_t1 = np.repeat(img_t1, 3, axis=-1)

        img_t1 = img_t1 - imagenet_mean
        img_t1 = img_t1 / imagenet_std

        x = torch.tensor(img_t1)
        # make it a batch-like
        x = x.unsqueeze(dim=0)
        x = torch.einsum('nhwc->nchw', x)

        loss, y, mask = self.mae_model(x.float(), x.float(), mask_ratio=0.25)

        # Patchify x 
        x = self.mae_model.patchify(x)
        x = x.float()
        # Expand mask to match feature dimensions [N, L, D]
        # mask = mask.unsqueeze(-1).repeat(1, 1, x.shape[2])
        x[mask == 0] = y

        # Unpatchify x and x2 back to image format
        x = self.mae_model.unpatchify(x)
        x = torch.einsum('nchw->nhwc', x).detach().cpu()  # NCHW to NHWC

        img_t1 = x.squeeze(0).numpy() #3 224 224

        # Reverse processing 

        # Un-normalize & *255
        img_t1 = (img_t1 * imagenet_std + imagenet_mean) * 255 
        # Convert the PyTorch tensor to NumPy

        # [224, 224, 3]  # Select the second channel (middle)
        img_t1 = img_t1[:, :, 1] 
        
        #resize 
        img_t0 = cv2.resize(img_t0, self.image_size) # opencv resize => image_size (W, H) 
        img_t1 = cv2.resize(img_t1, self.image_size) # opencv resize => image_size (W, H) 
       
        img_t0 = np.array(img_t0) # convert to np.array
        img_t1 = np.array(img_t1) # convert to np.array

        #normalize 
        img_t0 = np.array(cv2.normalize(img_t0, None, 0, 255, cv2.NORM_MINMAX), dtype=np.uint8)
        img_t1 = np.array(cv2.normalize(img_t1, None, 0, 255, cv2.NORM_MINMAX), dtype=np.uint8)

#         print("2 -------", img_t0)
        
        
        img_t0 = np.expand_dims(img_t0, axis=-1)  # expand dimension for gray scale
        img_t1 = np.expand_dims(img_t1, axis=-1)  # expand dimension for gray scale

        if not self.gray_scale:  # instead of reading image as rgb, we repeat the image for three times
            img_t0 = np.repeat(img_t0, 3, axis=-1)
            img_t1 = np.repeat(img_t1, 3, axis=-1)

        #convert to tensor
        img_t0 = transforms.ToTensor()(img_t0)
        img_t1 = transforms.ToTensor()(img_t1)

#         #normalization
#         img_t0 = transforms.Normalize(mean=[0.5], std=[0.5])(img_t0) 
#         img_t1 = transforms.Normalize(mean=[0.5], std=[0.5])(img_t1) 

        return img_t0, img_t1, _label, _case
