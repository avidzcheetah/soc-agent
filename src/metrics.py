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

def generate_classification_report(labels, preds, output_path):
    """
    Generate JSON and TXT classification report file.
    """
    report_dict = classification_report(labels, preds, output_dict=True, zero_division=0)
    report_txt = classification_report(labels, preds, zero_division=0)
    
    with open(output_path.replace('.json', '.txt'), 'w') as f:
        f.write(report_txt)
        
    with open(output_path, 'w') as f:
        json.dump(report_dict, f, indent=4)
        
    return report_dict

def plot_confusion_matrices(labels, preds, results_dir):
    """
    Plot raw and normalized confusion matrix heatmaps using matplotlib/seaborn.
    """
    from sklearn.metrics import confusion_matrix
    import os
    
    # Raw Confusion Matrix
    cm_raw = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Raw)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(results_dir, 'confusion_matrix_raw.png'))
    plt.close()
    
    # Normalized Confusion Matrix
    cm_norm = confusion_matrix(labels, preds, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues')
    plt.title('Confusion Matrix (Normalized)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(results_dir, 'confusion_matrix_normalized.png'))
    plt.close()

