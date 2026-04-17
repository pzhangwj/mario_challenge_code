import functools
import operator
import os

import pandas as pd
import torch
import yaml
from sklearn.metrics import f1_score, matthews_corrcoef
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.model import MarioModelT1
from utils.dataset import MARIO_DS_T1
from utils.scoring import specificity


class InferenceTask1:
    def __init__(self, model_paths, model_names, model_params, num_test, output_dir, model_weights=None, *args, **kwargs):

        """
        Initializes the inference class with model paths and weights.

        Args:
            model_paths (list): List of paths to the model files.
            model_weights (list, optional): List of weights for each model. Defaults to equal weights.
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_paths = model_paths
        self.model_names = model_names
        self.model_params = model_params
        self.output_dir = output_dir
        self.i = num_test

        self._check_model_paths()
        self.models = [
            self.load_model(model_name, model_path, model_params)
            for model_name, model_path in zip(model_names, model_paths)
        ]
        print(f"Using device: {self.device}")

        if model_weights is None:
            self.model_weights = [1.0 / len(model_paths)] * len(model_paths)
        else:
            self.model_weights = model_weights

    def _check_model_paths(self):
        missing_paths = [path for path in self.model_paths if not os.path.exists(path)]
        if missing_paths:
            raise FileNotFoundError("Missing model checkpoint(s):\n" + "\n".join(missing_paths))

    def load_model(self, model_name, model_path, model_params, *args, **kwargs):

        """
        Loads a model from a given path and it's class name.

        Args:
            model_name (str): name of the model class.
            model_path (str): Path to the model file.
            

        Returns:
            torch.nn.Module: Loaded model.
        """

        model = eval(model_name)(
            model_params["backbone"],
            model_params["pretrained"],
            model_params["num_classes"],
            model_params["model_type"],
        )
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

        y_true, y_pred, cases = [], [], []

        with torch.no_grad():
            for data in tqdm(data_loader):
                imgs_t0, imgs_t1, labels, case_ids = data
                imgs_t0 = imgs_t0.to(self.device).float()
                imgs_t1 = imgs_t1.to(self.device).float()

                output = self.models[0](imgs_t0, imgs_t1)
                prediction = output.argmax(dim=1).item()

                y_pred.append([prediction])
                y_true.append(labels.tolist())
                cases.append(case_ids.tolist())

        return y_true, y_pred, cases

    def scoring(self, y_true, y_pred):

        """
        DO NOT EDIT THIS CODE
        Calculates various scoring metrics.

        Args:
            y_true (list): True labels.
            y_pred (list): Predicted labels.

        Returns:
            dict: Dictionary containing various scores.
        """

        return {
            "F1_score": f1_score(y_true, y_pred, average="micro"),
            "Rk-correlation": matthews_corrcoef(y_true, y_pred),
            "Specificity": specificity(y_true, y_pred),
        }

    def simple_ensemble_inference(self, data_loader):

        """
        Performs inference using model ensembling and test time augmentation.

        Args:
            data_loader (DataLoader): DataLoader for the input data.

        Returns:
            list: True labels, predicted labels, and case IDs.
        """

        y_true, y_pred, cases = [], [], []

        with torch.no_grad():
            for data in tqdm(data_loader):
                imgs_t0, imgs_t1, labels, case_ids = data
                imgs_t0 = imgs_t0.to(self.device).float()
                imgs_t1 = imgs_t1.to(self.device).float()

                outputs = []
                for model in self.models:
                    output = model(imgs_t0, imgs_t1)
                    outputs.append(output)

                averaged_output = torch.mean(torch.stack(outputs), dim=0)
                prediction = list(averaged_output.argmax(dim=1).cpu().detach().numpy())

                y_pred.append(prediction)
                y_true.append(labels.tolist())
                cases.append(case_ids.tolist())

        return y_true, y_pred, cases

    def run(self, data_loader, use_ensemble=True):

        """
        Runs the inference and saves results.

        Args:
            data_loader (DataLoader): DataLoader for the input data.
            use_tta (bool): Whether to use test time augmentation.
            n_augmentations (int): Number of augmentations to apply for TTA.

        Returns:
            dict: Dictionary containing various scores.
        """
        
        if use_ensemble:
            y_true, y_pred, cases = self.simple_ensemble_inference(data_loader)
        else:
            y_true, y_pred, cases = self.simple_inference(data_loader)

        y_true = functools.reduce(operator.iconcat, y_true, [])
        y_pred = functools.reduce(operator.iconcat, y_pred, [])
        cases = functools.reduce(operator.iconcat, cases, [])

        os.makedirs(self.output_dir, exist_ok=True)
        output_file = os.path.join(
            self.output_dir, f"results_task1_team_df41_method_{self.i}.csv"
        )
        df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "cases": cases})
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        self.i += 1
        return self.scoring(y_true, y_pred)


def load_config(config_path="config_task1.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()

    if not os.path.exists(cfg["csv_path"]):
        raise FileNotFoundError(
            f"CSV file not found: {cfg['csv_path']}. Please provide the MARIO metadata file."
        )

    if not os.path.exists(cfg["data_root"]):
        raise FileNotFoundError(
            f"Data directory not found: {cfg['data_root']}. Please provide the MARIO dataset locally."
        )

    df = pd.read_csv(cfg["csv_path"])
    max_samples = cfg.get("max_samples")
    if max_samples is not None:
        df = df[:max_samples]

    dataset = MARIO_DS_T1(
        df,
        mode="test",
        image_size=tuple(cfg["image_size"]),
        gray_scale=cfg["gray_scale"],
        root_dir=cfg["data_root"],
        processing_octip=True,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
    )

    inference_task1_v1 = InferenceTask1(
        cfg["model_paths_v1"],
        cfg["model_names"],
        cfg["model_params"],
        num_test=1,
        output_dir=cfg["output_dir"],
    )
    scores_1 = inference_task1_v1.run(data_loader, use_ensemble=True)
    print(
        f"[Task1 - V1] F1_score: {scores_1['F1_score']}, "
        f"Rk-correlation: {scores_1['Rk-correlation']}, "
        f"Specificity: {scores_1['Specificity']}"
    )

    inference_task1_v2 = InferenceTask1(
        cfg["model_paths_v2"],
        cfg["model_names"],
        cfg["model_params"],
        num_test=2,
        output_dir=cfg["output_dir"],
    )
    scores_2 = inference_task1_v2.run(data_loader, use_ensemble=True)
    print(
        f"[Task1 - V2] F1_score: {scores_2['F1_score']}, "
        f"Rk-correlation: {scores_2['Rk-correlation']}, "
        f"Specificity: {scores_2['Specificity']}"
    )


if __name__ == "__main__":
    main()
