import numpy as np
import torch
from torchvision import transforms

from utils.octip_segmentation_2d.octip.retina_localizer import RetinaLocalizer, RetinaLocalizationDataset
from utils.octip_segmentation_2d.octip.pre_processor import PreProcessor

def octip_preprocess_input(localizer1, localizer2, preprocessor, image_path):
    # octip preprocessing 
    
    # segmenting the retina of one B-scan with the first model 

    _, seg1 = localizer1(RetinaLocalizationDataset([image_path], 1, localizer1)) #shape (1,384,384,1) 
    seg1 = (seg1.squeeze()*255).astype(np.uint8)

    # segmenting the retina of one B-scan with the second model
    _, seg2 = localizer2(RetinaLocalizationDataset([image_path], 1, localizer2)) #shape (1,320,320,1)
    seg2 = (seg2.squeeze()*255).astype(np.uint8)
    
    preprocessed_image = preprocessor(img_path, [seg1, seg2], output_directory="" ) #one each time 
    
    return preprocessed_image
