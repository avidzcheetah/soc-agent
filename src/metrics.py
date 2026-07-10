# Responsibility: Accuracy, Macro F1, MCC, confusion matrix
import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def compute_metrics(labels, preds):
    """
    Calculate Evaluation metrics.
    """
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "precision": precision,
        "recall": recall
    }

def generate_classification_report(labels, preds, active_classes, output_path):
    """
    Generate JSON classification report file.
    """
    pass

def plot_confusion_matrices(labels, preds, results_dir, action_space_path="data/action_space.csv"):
    """
    Plot raw and normalized confusion matrix heatmaps using matplotlib/seaborn.
    """
    pass
