# Responsibility: Accuracy, Macro F1, MCC, confusion matrix

def compute_metrics(labels, preds, active_classes):
    """
    Calculate F1 metrics and correlation coefficients, filtering out inactive classes.
    """
    pass

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
