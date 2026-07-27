import os
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np

# Create output directories
out_dir = "results/plots"
pdf_out = "results/Phase1_Experiment_Results_Report.pdf"
os.makedirs(out_dir, exist_ok=True)

# Set matplotlib style
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.titlesize': 14,
    'figure.dpi': 300
})

# Load available epoch data
exp_data = {}

if os.path.exists("experiments/EXP_20260710_001/metadata/metrics.json"):
    with open("experiments/EXP_20260710_001/metadata/metrics.json") as f:
        exp_data["EXP_001 (Baseline)"] = json.load(f)

if os.path.exists("experiments/EXP_20260712_001/metadata/metrics.json"):
    with open("experiments/EXP_20260712_001/metadata/metrics.json") as f:
        exp_data["EXP_002A (Weighted CE)"] = json.load(f)

if os.path.exists("experiments/EXP_20260713_001/metadata/metrics.json"):
    with open("experiments/EXP_20260713_001/metadata/metrics.json") as f:
        exp_data["EXP_002D (Targeted Synth)"] = json.load(f)

# EXP_002G (FP32)
exp_data["EXP_002G (FP32 Ablation - Selected Final)"] = [
    {"epoch": 1, "train": {"loss": 1.4192, "accuracy": 0.6934}, "val": {"loss": 0.8962, "accuracy": 0.8947}},
    {"epoch": 2, "train": {"loss": 0.6807, "accuracy": 0.9715}, "val": {"loss": 0.8263, "accuracy": 0.9285}},
    {"epoch": 3, "train": {"loss": 0.6445, "accuracy": 0.9825}, "val": {"loss": 0.8055, "accuracy": 0.9344}},
    {"epoch": 4, "train": {"loss": 0.6272, "accuracy": 0.9891}, "val": {"loss": 0.8042, "accuracy": 0.9383}},
    {"epoch": 5, "train": {"loss": 0.6199, "accuracy": 0.9920}, "val": {"loss": 0.8104, "accuracy": 0.9396}},
    {"epoch": 6, "train": {"loss": 0.6141, "accuracy": 0.9944}, "val": {"loss": 0.7942, "accuracy": 0.9441}},
    {"epoch": 7, "train": {"loss": 0.6174, "accuracy": 0.9930}, "val": {"loss": 0.7958, "accuracy": 0.9454}},
    {"epoch": 8, "train": {"loss": 0.6045, "accuracy": 0.9969}, "val": {"loss": 0.7890, "accuracy": 0.9448}},
    {"epoch": 9, "train": {"loss": 0.6041, "accuracy": 0.9972}, "val": {"loss": 0.7958, "accuracy": 0.9441}},
    {"epoch": 10, "train": {"loss": 0.6038, "accuracy": 0.9972}, "val": {"loss": 0.7936, "accuracy": 0.9461}}
]

with PdfPages(pdf_out) as pdf:
    # --- Page 1: Individual Dual Y-Axis Graphs (Accuracy & Val Loss vs Epoch in SAME graph) ---
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle("Per-Experiment Dual Y-Axis Analysis: Accuracy & Validation Loss vs Epoch", fontsize=14, fontweight='bold')
    
    ax_flat = axes.flatten()
    
    for i, (name, epochs) in enumerate(exp_data.items()):
        ax1 = ax_flat[i]
        x = [e["epoch"] for e in epochs]
        train_acc = [e["train"]["accuracy"] for e in epochs]
        val_acc = [e["val"]["accuracy"] for e in epochs]
        val_loss = [e["val"]["loss"] for e in epochs]
        
        # Left Y-Axis: Accuracy
        color1 = 'tab:blue'
        color1_val = 'tab:cyan'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy', color=color1, fontweight='bold')
        l1 = ax1.plot(x, train_acc, color=color1, marker='o', label='Train Accuracy', linewidth=2)
        l2 = ax1.plot(x, val_acc, color=color1_val, marker='s', linestyle='--', label='Val Accuracy', linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_ylim(0.2 if "002A" in name else 0.5, 1.02)
        ax1.grid(True, linestyle=":", alpha=0.5)
        
        # Right Y-Axis: Validation Loss
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Validation Loss', color=color2, fontweight='bold')
        l3 = ax2.plot(x, val_loss, color=color2, marker='^', linestyle='-', label='Val Loss', linewidth=2)
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # Combine legends
        lines = l1 + l2 + l3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='center right' if "002A" not in name else 'upper right', fontsize=8)
        
        ax1.set_title(name, fontsize=11, fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf.savefig(fig)
    plt.close()

    # --- Page 2: High-Resolution Individual 1x1 Graphs for Each Experiment ---
    for name, epochs in exp_data.items():
        fig, ax1 = plt.subplots(figsize=(9, 5.5))
        
        x = [e["epoch"] for e in epochs]
        train_acc = [e["train"]["accuracy"] for e in epochs]
        val_acc = [e["val"]["accuracy"] for e in epochs]
        val_loss = [e["val"]["loss"] for e in epochs]
        train_loss = [e["train"]["loss"] for e in epochs]
        
        color_acc = '#1f77b4'
        color_val_acc = '#17becf'
        color_loss = '#d62728'
        color_train_loss = '#ff7f0e'
        
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Accuracy', color=color_acc, fontsize=12, fontweight='bold')
        l1 = ax1.plot(x, train_acc, color=color_acc, marker='o', label='Training Accuracy', linewidth=2.5, markersize=7)
        l2 = ax1.plot(x, val_acc, color=color_val_acc, marker='s', linestyle='--', label='Validation Accuracy', linewidth=2.5, markersize=7)
        ax1.tick_params(axis='y', labelcolor=color_acc, labelsize=11)
        ax1.grid(True, linestyle="--", alpha=0.5)
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('Loss', color=color_loss, fontsize=12, fontweight='bold')
        l3 = ax2.plot(x, val_loss, color=color_loss, marker='^', linestyle='-', label='Validation Loss', linewidth=2.5, markersize=7)
        l4 = ax2.plot(x, train_loss, color=color_train_loss, marker='d', linestyle=':', label='Training Loss', linewidth=2, markersize=6)
        ax2.tick_params(axis='y', labelcolor=color_loss, labelsize=11)
        
        lines = l1 + l2 + l3 + l4
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='center right', frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
        
        plt.title(f"Detailed Dual Y-Axis Curve: {name}\n(Accuracy & Loss Trajectories vs Epoch)", fontsize=13, fontweight='bold')
        plt.tight_layout()
        
        # Save individual PNG asset too
        safe_name = name.split(" ")[0].lower()
        plt.savefig(f"{out_dir}/{safe_name}_dual_axis.png")
        
        pdf.savefig(fig)
        plt.close()

    # --- Page 3: Summary Comparison Charts & Table ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    summary_data = [
        {"Exp": "EXP_001", "Name": "Baseline", "Macro F1": 0.6248, "Weighted F1": 0.8975, "MCC": 0.8807, "Top-2 Acc": 0.9030},
        {"Exp": "EXP_002A", "Name": "Weighted CE", "Macro F1": 0.5741, "Weighted F1": 0.8066, "MCC": 0.7706, "Top-2 Acc": 0.9012},
        {"Exp": "EXP_002B", "Name": "Label Smooth", "Macro F1": 0.6191, "Weighted F1": 0.9181, "MCC": 0.9069, "Top-2 Acc": 0.9500},
        {"Exp": "EXP_002C", "Name": "CISA Tags", "Macro F1": 0.7057, "Weighted F1": 0.9416, "MCC": 0.9332, "Top-2 Acc": 0.9680},
        {"Exp": "EXP_002D", "Name": "Targeted Synth", "Macro F1": 0.7296, "Weighted F1": 0.9438, "MCC": 0.9341, "Top-2 Acc": 0.9682},
        {"Exp": "EXP_002G", "Name": "FP32 Ablation (Final)", "Macro F1": 0.7121, "Weighted F1": 0.9454, "MCC": 0.9364, "Top-2 Acc": 0.9695}
    ]
    df_sum = pd.DataFrame(summary_data)
    
    # Top plot: Bar chart of Weighted F1 & Macro F1
    x_indices = np.arange(len(df_sum))
    width = 0.35
    
    axes[0].bar(x_indices - width/2, df_sum["Macro F1"], width, label="Macro F1", color="#1f77b4", edgecolor="black")
    axes[0].bar(x_indices + width/2, df_sum["Weighted F1"], width, label="Weighted F1", color="#2ca02c", edgecolor="black")
    axes[0].set_xticks(x_indices)
    axes[0].set_xticklabels(df_sum["Exp"] + "\n(" + df_sum["Name"] + ")", fontsize=9)
    axes[0].set_ylabel("Score", fontsize=11, fontweight='bold')
    axes[0].set_title("Macro F1 vs Weighted F1 Across Key Phase 1 Experiments", fontsize=12, fontweight='bold')
    axes[0].set_ylim(0.5, 1.0)
    axes[0].legend(loc="lower right")
    axes[0].grid(axis='y', linestyle="--", alpha=0.5)
    
    # Bottom plot: Summary Table
    axes[1].axis('tight')
    axes[1].axis('off')
    table_data = [
        ["Experiment", "Key Technique", "Macro F1", "Weighted F1", "MCC", "Top-2 Acc", "Verdict"],
        ["EXP_001", "Baseline CE + Sampler", "0.6248", "0.8975", "0.8807", "0.9030", "Baseline Reference"],
        ["EXP_002A", "Weighted CE Loss", "0.5741", "0.8066", "0.7706", "0.9012", "Double-weighting collapse"],
        ["EXP_002B", "Label Smoothing (0.1)", "0.6191", "0.9181", "0.9069", "0.9500", "Stabilized training"],
        ["EXP_002C", "CISA Phase Tags", "0.7057", "0.9416", "0.9332", "0.9680", "Disambiguated semantics"],
        ["EXP_002D", "Targeted Synthetic", "0.7296", "0.9438", "0.9341", "0.9682", "Peak Macro F1"],
        ["EXP_002G", "FP32 Precision Ablation", "0.7121", "0.9454", "0.9364", "0.9695", "SELECTED FINAL MODEL"]
    ]
    table = axes[1].table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    # Style header row
    for j in range(len(table_data[0])):
        table[(0, j)].set_facecolor('#1f77b4')
        table[(0, j)].get_text().set_color('white')
        table[(0, j)].get_text().set_weight('bold')
    
    # Highlight final selected model row
    for j in range(len(table_data[0])):
        table[(6, j)].set_facecolor('#e8f5e9')
        table[(6, j)].get_text().set_weight('bold')

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()

print(f"PDF report successfully generated at {pdf_out}")
