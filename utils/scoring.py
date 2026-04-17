# Function to calculate Specificity metric
import numpy as np
import math
import os

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, cohen_kappa_score
import seaborn as sns

import matplotlib.pylab as plt
import torchvision.transforms as T

# Fonction pour calculer les métriques
def compute_metrics(y_true, y_pred):
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    return accuracy, precision, recall, f1, cm, kappa

# Nouvelle fonction pour dessiner et enregistrer la matrice de confusion
def plot_confusion_matrix(cm, class_names, filename='confusion_matrix.png'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(filename)
    plt.close()


def specificity(class_ground_truth, class_prediction):
    """
    Calculates the Specificity metric for a binary classification problem.

    Args:
        class_ground_truth (array-like): Array of true class labels.
        class_prediction (array-like): Array of predicted class labels.

    Returns:
        float: The average Specificity score.
    """

    eps = 0.000001  # Add a small value to avoid division by zero
    cnf_matrix = confusion_matrix(class_ground_truth, class_prediction)

    # Calculate True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    # Cast all values to float to avoid type errors
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    # Calculate Specificity for each class and average them
    spe = TN / (TN + FP + eps)
    spe = spe.mean()

    return spe