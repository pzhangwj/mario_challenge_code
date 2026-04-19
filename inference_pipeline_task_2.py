import functools
import operator
import os

import pandas as pd
import torch
import yaml
from sklearn.metrics import cohen_kappa_score, f1_score, matthews_corrcoef
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.model import MarioModelT1
from utils.dataset import MARIO_DS_T2
from utils.mae_model import PatchProgressionAutoencoderViT
from utils.scoring import specificity


class InferenceTask2:
    def __init__(self, model_paths, model_names, model_params, test_number, output_dir, model_weights=None, *args, **kwargs):
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
        self.i = test_number

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

    def load_model(self, model_name, model_path, model_params):
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
            "Quadratic-weighted_Kappa": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
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
                averaged_output_task_2 = averaged_output[:, :3]
                prediction = list(averaged_output_task_2.argmax(dim=1).cpu().detach().numpy())

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
            self.output_dir, f"results_task2_team_df41_method_{self.i}.csv"
        )
        df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "cases": cases})
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        self.i += 1
        return self.scoring(y_true, y_pred)


def load_config(config_path="config_task2.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_mae_model(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"MAE checkpoint not found: {checkpoint_path}"
        )

    model_mae = PatchProgressionAutoencoderViT()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    msg = model_mae.load_state_dict(checkpoint, strict=False)
    print(msg)
    return model_mae


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

    model_mae = load_mae_model(cfg["mae_checkpoint"])

    dataset = MARIO_DS_T2(
        df,
        mode="test",
        image_size=tuple(cfg["image_size"]),
        gray_scale=cfg["gray_scale"],
        root_dir=cfg["data_root"],
        processing_octip=True,
        mae_model=model_mae,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
    )

    inference_task2_v1 = InferenceTask2(
        cfg["model_paths_v1"],
        cfg["model_names"],
        cfg["model_params"],
        test_number=1,
        output_dir=cfg["output_dir"],
    )
    scores_1 = inference_task2_v1.run(data_loader, use_ensemble=True)
    print(
        f"[Task2 - V1] F1_score: {scores_1['F1_score']}, "
        f"Rk-correlation: {scores_1['Rk-correlation']}, "
        f"Specificity: {scores_1['Specificity']}, "
        f"Quadratic-weighted_Kappa: {scores_1['Quadratic-weighted_Kappa']}"
    )

    inference_task2_v2 = InferenceTask2(
        cfg["model_paths_v2"],
        cfg["model_names"],
        cfg["model_params"],
        test_number=2,
        output_dir=cfg["output_dir"],
    )
    scores_2 = inference_task2_v2.run(data_loader, use_ensemble=True)
    print(
        f"[Task2 - V2] F1_score: {scores_2['F1_score']}, "
        f"Rk-correlation: {scores_2['Rk-correlation']}, "
        f"Specificity: {scores_2['Specificity']}, "
        f"Quadratic-weighted_Kappa: {scores_2['Quadratic-weighted_Kappa']}"
    )

    inference_task2_v3 = InferenceTask2(
        cfg["model_paths_v3"],
        cfg["model_names"],
        cfg["model_params"],
        test_number=3,
        output_dir=cfg["output_dir"],
    )
    scores_3 = inference_task2_v3.run(data_loader, use_ensemble=True)
    print(
        f"[Task2 - V3] F1_score: {scores_3['F1_score']}, "
        f"Rk-correlation: {scores_3['Rk-correlation']}, "
        f"Specificity: {scores_3['Specificity']}, "
        f"Quadratic-weighted_Kappa: {scores_3['Quadratic-weighted_Kappa']}"
    )


if __name__ == "__main__":
    main()
