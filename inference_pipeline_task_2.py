import torch
import numpy as np
import os
import pandas as pd
from sklearn.metrics import f1_score, cohen_kappa_score, matthews_corrcoef
from utils.dataset import MARIO_DS_T2_V2
from utils.scoring import specificity, compute_metrics, plot_confusion_matrix
from tqdm import tqdm

from models.model import MarioModelT1

from torch.utils.data import DataLoader 
import torchvision.transforms as T

from utils.mae_model import PatchProgressionAutoencoderViT
from utils.mae_utils import *

import operator
import functools

import cv2

class InferenceTask2:
    
    def __init__(self, model_paths, model_names, model_params, test_number, model_weights=None, *args, **kwargs):
        """
        Initializes the inference class with model paths and weights.

        Args:
            model_paths (list): List of paths to the model files.
            model_weights (list, optional): List of weights for each model. Defaults to equal weights.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = [self.load_model(model_name,model_path,model_params) for model_name,model_path in zip(model_names,model_paths)]
        print(f"Using device: {self.device}")
        self.i = test_number
        if model_weights is None:
            self.model_weights = [1.0 / len(model_paths)] * len(model_paths)
        else:
            self.model_weights = model_weights

    def load_model(self, model_name, model_path,model_params, *args, **kwargs):
        """
        Loads a model from a given path and it's class name.

        Args:
            model_name (str): name of the model class.
            model_path (str): Path to the model file.
            

        Returns:
            torch.nn.Module: Loaded model.
        """
        model = eval(model_name)(model_params['backbone'], model_params['pretrained'], model_params['num_classes'], model_params['model_type'] )
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        
        return model

    def simple_inference(self, data_loader):
        """
        Performs inference on the data using the loaded model.

        Args:
            data_loader (DataLoader): DataLoader for the input data.

        Returns:
            list: True labels, predicted labels, and case IDs.
        """
        
        ## The proposed example only use the pair of OCT slice, but you are free to update if your pipeline involve
        ## localizer and the clinical, udapte accordingly 
        
        y_prob = []
        y_true = []
        y_pred = []
        cases = []
        
        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(data_loader)):
                
                imgs_t0, imgs_t1, labels, case_ids = data
                imgs_t0 = imgs_t0.to(self.device).float()
                imgs_t1 = imgs_t1.to(self.device).float()
                
                output = model(imgs_t0, imgs_t1)
                               
                prediction = output.argmax(dim=1).item()
                
                y_pred.append(prediction)
                y_true.append(labels)
                
                cases.append(case_ids)
                
        return y_true, y_pred, cases

    def scoring(self, y_true, y_pred):
        """
        DO NOT EDIT THIS CODE

        Calculates F1 score, Matthews Correlation Coefficient, and Specificity for a classification task.

        Args:
            y_true (list): True labels.
            y_pred (list): Predicted labels.

        Returns:
            dict: Dictionary containing F1 score, Matthews Correlation Coefficient, Specificity, and Quadratic-weighted Kappa metrics.
        """
        return {
            "F1_score": f1_score(y_true, y_pred, average="micro"),
            "Rk-correlation": matthews_corrcoef(y_true, y_pred),
            "Specificity": specificity(y_true, y_pred),
            "Quadratic-weighted_Kappa": cohen_kappa_score(y_true, y_pred, weights="quadratic")
        }

    def simple_ensemble_inference(self, data_loader):
        """
        Performs inference using model ensembling and test time augmentation.

        Args:
            data_loader (DataLoader): DataLoader for the input data.

        Returns:
            list: True labels, predicted labels, and case IDs.
        """
        
        y_prob = []
        y_true = []
        y_pred = []
        cases = []
        
        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(data_loader)):
                
                imgs_t0, imgs_t1, labels, case_ids = data
                imgs_t0 = imgs_t0.to(self.device).float()
                imgs_t1 = imgs_t1.to(self.device).float()
                
                outputs = []
                
                for model in self.models:
                    
                    output = model(imgs_t0, imgs_t1)
                    outputs.append(output)
                
                averaged_output = torch.mean(torch.stack(outputs), dim=0)
                
                averaged_output_task_2 = averaged_output[:, :3] #remove class 3
                
                prediction = list(averaged_output_task_2.argmax(dim=1).cpu().detach().numpy())
                
                y_pred.append(prediction)
                y_true.append(labels.tolist())
                
                cases.append(case_ids.tolist())
    
        return y_true, y_pred, cases  

    def run(self, data_loader, use_ensemble = True):
        """
        Runs the inference and saves results.

        Args:
            data_loader (DataLoader): DataLoader for the input data.
            use_tta (bool): Whether to use test time augmentation.
            n_augmentations (int): Number of augmentations to apply for TTA.

        Returns:
            dict: Dictionary containing various scores.
        """
        
        
        ## You can test as much inference pipeline you which
        # in your local machine. You will have to select
        # two shot to for the final submission. 
        # The inference should always return a list of batch containing label,prediction,cases 
        # The method run should always return the scores
        
        if use_ensemble:
            y_true, y_pred, cases = self.simple_ensemble_inference(data_loader)

        else:
            y_true, y_pred, cases = self.simple_inference(data_loader)
            
            
        # DO NOT EDIT THIS PART
    
            
        y_true = functools.reduce(operator.iconcat, y_true, [])
        y_pred = functools.reduce(operator.iconcat, y_pred, [])
        cases = functools.reduce(operator.iconcat, cases, [])          

#         output_file = f"output/results_task2_team_{os.environ['Team_name']}_method_{self.i}.csv"
        
        output_file = f"output/results_task2_team_df41_method_{self.i}.csv"
        
        df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'cases': cases})
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        self.i +=1
        return self.scoring(y_true, y_pred)

# Main execution
# print(f"Starting the inference for the team: {os.environ['Team_name']}")

label_type = "multiclass"
image_size=(512,200) #W,H
gray_scale = False

# Load csv

df = pd.read_csv('./csv/df_task2_val_challenge.csv') 

df = df[:100]

# Load model PPMAE()

model_mae = PatchProgressionAutoencoderViT()
chkpt_dir = "./utils/ppmae_models/ppmae_vit_large_v2.pth"

checkpoint = torch.load(chkpt_dir, map_location='cpu')

msg = model_mae.load_state_dict(checkpoint, strict=False)
print(msg)

# Load dataset

# docker 
dataset = MARIO_DS_T2_V2(df, mode="test", image_size=(512,200) ,gray_scale=gray_scale, root_dir= "./data/", processing_octip=True, mae_model=model_mae)

# local 
# dataset = MARIO_DS_T2_V2(df, mode="test", image_size=(512,200) ,gray_scale=gray_scale, root_dir= "/home/pzhang/Data/database/MARIO_CHALLENGE/data_2/data_task2/val/", processing_octip=True, mae_model=model_mae)

data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


### [TEST 1] : model_paths from task 1 v1
model_paths = ['./models/task1_models/v1/task1_best_model_f0.pth', './models/task1_models/v1/task1_best_model_f1.pth', './models/task1_models/v1/task1_best_model_f2.pth', './models/task1_models/v1/task1_best_model_f3.pth']  # multiple models 

model_names = ["MarioModelT1", "MarioModelT1", "MarioModelT1", "MarioModelT1"]

model_params = {
    "backbone": "resnet50",
    "pretrained": False,
    "num_classes": 4,
    "model_type": "features_fusion"
}


num_test = 1 
inference_task2 = InferenceTask2(model_paths, model_names, model_params, num_test)

scores_1 = inference_task2.run(data_loader, use_ensemble = True)
print(f" Obtained scores for inference method 1: F1_score: {scores_1['F1_score']}, Rk-correlation: {scores_1['Rk-correlation']}, Specificity: {scores_1['Specificity']}, Quadratic-weighted_Kappa: {scores_1['Quadratic-weighted_Kappa']}")

### [TEST 2] : model_paths from task 1 v2
# model_paths
model_paths = ['./models/task1_models/v2/task1_best_model_f0_ens_2.pth', './models/task1_models/v2/task1_best_model_f1_ens_2.pth', './models/task1_models/v2/task1_best_model_f2_ens_2.pth', './models/task1_models/v2/task1_best_model_f3_ens_2.pth']  # multiple models 

num_test = 2
inference_task2 = InferenceTask2(model_paths, model_names, model_params, num_test)

scores_2 = inference_task2.run(data_loader, use_ensemble = True)
print(f" Obtained scores for inference method 2: F1_score: {scores_2['F1_score']}, Rk-correlation: {scores_2['Rk-correlation']}, Specificity: {scores_2['Specificity']}, Quadratic-weighted_Kappa: {scores_2['Quadratic-weighted_Kappa']}")


# model_paths
model_paths = ['./models/task2_models/best_model_finetune_f0.pth', './models/task2_models/best_model_finetune_f1.pth', './models/task2_models/best_model_finetune_f2.pth', './models/task2_models/best_model_finetune_f3.pth']  # multiple models 

num_test = 3
inference_task2 = InferenceTask2(model_paths, model_names, model_params, num_test)

scores_3 = inference_task2.run(data_loader, use_ensemble = True)
print(f" Obtained scores for inference method 3: F1_score: {scores_3['F1_score']}, Rk-correlation: {scores_3['Rk-correlation']}, Specificity: {scores_3['Specificity']}, Quadratic-weighted_Kappa: {scores_3['Quadratic-weighted_Kappa']}")

###################################### Print all scores #############################################
print(f" Obtained scores for inference method 1: F1_score: {scores_1['F1_score']}, Rk-correlation: {scores_1['Rk-correlation']}, Specificity: {scores_1['Specificity']}, Quadratic-weighted_Kappa: {scores_1['Quadratic-weighted_Kappa']}")
print(f" Obtained scores for inference method 2: F1_score: {scores_2['F1_score']}, Rk-correlation: {scores_2['Rk-correlation']}, Specificity: {scores_2['Specificity']}, Quadratic-weighted_Kappa: {scores_2['Quadratic-weighted_Kappa']}")
print(f" Obtained scores for inference method 3: F1_score: {scores_3['F1_score']}, Rk-correlation: {scores_3['Rk-correlation']}, Specificity: {scores_3['Specificity']}, Quadratic-weighted_Kappa: {scores_3['Quadratic-weighted_Kappa']}")
