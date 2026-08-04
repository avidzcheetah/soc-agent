import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set publication style
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300
})

out_dir = "results/plots"
os.makedirs(out_dir, exist_ok=True)

# 1. Load available epoch-level data from JSON & logs
exp_data = {}

# EXP_001 (EXP_20260710_001)
if os.path.exists("experiments/EXP_20260710_001/metadata/metrics.json"):
    with open("experiments/EXP_20260710_001/metadata/metrics.json") as f:
        exp_data["EXP_001 (Baseline)"] = json.load(f)

# EXP_002A (EXP_20260712_001)
if os.path.exists("experiments/EXP_20260712_001/metadata/metrics.json"):
    with open("experiments/EXP_20260712_001/metadata/metrics.json") as f:
        exp_data["EXP_002A (Weighted CE)"] = json.load(f)

# EXP_002D (EXP_20260713_001)
if os.path.exists("experiments/EXP_20260713_001/metadata/metrics.json"):
    with open("experiments/EXP_20260713_001/metadata/metrics.json") as f:
        exp_data["EXP_002D (Targeted Synth)"] = json.load(f)

# EXP_002G (EXP_20260727_001 - FP32)
exp_data["EXP_002G (FP32 Ablation - Final)"] = [
    {"epoch": 1, "train": {"loss": 1.4192, "accuracy": 0.6934, "macro_f1": 0.5291}, "val": {"loss": 0.8962, "accuracy": 0.8947, "macro_f1": 0.6795, "top2_accuracy": 0.9454}},
    {"epoch": 2, "train": {"loss": 0.6807, "accuracy": 0.9715, "macro_f1": 0.9474}, "val": {"loss": 0.8263, "accuracy": 0.9285, "macro_f1": 0.7551, "top2_accuracy": 0.9643}},
    {"epoch": 3, "train": {"loss": 0.6445, "accuracy": 0.9825, "macro_f1": 0.9670}, "val": {"loss": 0.8055, "accuracy": 0.9344, "macro_f1": 0.7188, "top2_accuracy": 0.9688}},
    {"epoch": 4, "train": {"loss": 0.6272, "accuracy": 0.9891, "macro_f1": 0.9777}, "val": {"loss": 0.8042, "accuracy": 0.9383, "macro_f1": 0.7356, "top2_accuracy": 0.9682}},
    {"epoch": 5, "train": {"loss": 0.6199, "accuracy": 0.9920, "macro_f1": 0.9850}, "val": {"loss": 0.8104, "accuracy": 0.9396, "macro_f1": 0.7918, "top2_accuracy": 0.9688}},
    {"epoch": 6, "train": {"loss": 0.6141, "accuracy": 0.9944, "macro_f1": 0.9893}, "val": {"loss": 0.7942, "accuracy": 0.9441, "macro_f1": 0.7639, "top2_accuracy": 0.9701}},
    {"epoch": 7, "train": {"loss": 0.6174, "accuracy": 0.9930, "macro_f1": 0.9865}, "val": {"loss": 0.7958, "accuracy": 0.9454, "macro_f1": 0.7475, "top2_accuracy": 0.9695}},
    {"epoch": 8, "train": {"loss": 0.6045, "accuracy": 0.9969, "macro_f1": 0.9940}, "val": {"loss": 0.7890, "accuracy": 0.9448, "macro_f1": 0.7375, "top2_accuracy": 0.9675}},
    {"epoch": 9, "train": {"loss": 0.6041, "accuracy": 0.9972, "macro_f1": 0.9946}, "val": {"loss": 0.7958, "accuracy": 0.9441, "macro_f1": 0.7916, "top2_accuracy": 0.9721}},
    {"epoch": 10, "train": {"loss": 0.6038, "accuracy": 0.9972, "macro_f1": 0.9945}, "val": {"loss": 0.7936, "accuracy": 0.9461, "macro_f1": 0.8025, "top2_accuracy": 0.9701}}
]

colors = {
    "EXP_001 (Baseline)": "#7f7f7f",
    "EXP_002A (Weighted CE)": "#d62728",
    "EXP_002D (Targeted Synth)": "#1f77b4",
    "EXP_002G (FP32 Ablation - Final)": "#2ca02c"
}

markers = {
    "EXP_001 (Baseline)": "o",
    "EXP_002A (Weighted CE)": "s",
    "EXP_002D (Targeted Synth)": "^",
    "EXP_002G (FP32 Ablation - Final)": "D"
}

# --- Per-Experiment Dual-Axis Plots (Accuracy & Loss on same scale) ---
for name, epochs in exp_data.items():
    fig, ax1 = plt.subplots(figsize=(9, 5.5))

    x = [e["epoch"] for e in epochs]
    train_acc = [e["train"]["accuracy"] for e in epochs]
    val_acc = [e["val"]["accuracy"] for e in epochs]
    train_loss = [e["train"]["loss"] for e in epochs]
    val_loss = [e["val"]["loss"] for e in epochs]

    # Compute shared y-limits so both axes use the same scale
    all_vals = train_acc + val_acc + train_loss + val_loss
    ymin = 0.0
    ymax = max(1.02, max(all_vals) * 1.05)

    # Left Y-Axis: Accuracy
    color_train_acc = '#1f77b4'
    color_val_acc = '#17becf'
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', color=color_train_acc, fontsize=12, fontweight='bold')
    l1 = ax1.plot(x, train_acc, color=color_train_acc, marker='o', label='Training Accuracy', linewidth=2.5, markersize=7)
    l2 = ax1.plot(x, val_acc, color=color_val_acc, marker='s', linestyle='--', label='Validation Accuracy', linewidth=2.5, markersize=7)
    ax1.tick_params(axis='y', labelcolor=color_train_acc, labelsize=11)
    ax1.set_ylim(ymin, ymax)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Right Y-Axis: Loss (same scale)
    ax2 = ax1.twinx()
    color_val_loss = '#d62728'
    color_train_loss = '#ff7f0e'
    ax2.set_ylabel('Loss', color=color_val_loss, fontsize=12, fontweight='bold')
    l3 = ax2.plot(x, val_loss, color=color_val_loss, marker='^', linestyle='-', label='Validation Loss', linewidth=2.5, markersize=7)
    l4 = ax2.plot(x, train_loss, color=color_train_loss, marker='d', linestyle=':', label='Training Loss', linewidth=2, markersize=6)
    ax2.tick_params(axis='y', labelcolor=color_val_loss, labelsize=11)
    ax2.set_ylim(ymin, ymax)

    # Combined legend
    lines = l1 + l2 + l3 + l4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', frameon=True, facecolor='white', framealpha=0.9, fontsize=10)

    plt.title(f"Dual Y-Axis: {name}\n(Accuracy & Loss vs Epoch — Same Scale)", fontsize=12, fontweight='bold')
    plt.tight_layout()

    # Save as individual PNG
    short_name = name.split(" ")[0].lower().replace("(", "").replace(")", "")
    plt.savefig(f"{out_dir}/{short_name}_dual_axis.png")
    plt.close()

# --- Plot 5: Training Accuracy vs Validation Loss (Supervisor Special Request) ---
plt.figure(figsize=(8, 5))
for name, epochs in exp_data.items():
    x = [e["train"]["accuracy"] for e in epochs]
    y = [e["val"]["loss"] for e in epochs]
    plt.plot(x, y, label=name, color=colors[name], marker=markers[name], linewidth=2, markersize=6)
    for i, e in enumerate(epochs):
        plt.annotate(f"E{e['epoch']}", (x[i], y[i]), textcoords="offset points", xytext=(5,-5), fontsize=8)
plt.title("Training Accuracy vs Validation Loss (Convergence & Overfitting Trajectory)")
plt.xlabel("Training Accuracy")
plt.ylabel("Validation Loss")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(f"{out_dir}/train_acc_vs_val_loss.png")
plt.close()

# --- Global Metrics Across All Experiments ---
summary_data = [
    {"Exp": "EXP_001", "Name": "Baseline", "Macro F1": 0.6248, "Weighted F1": 0.8975, "MCC": 0.8807, "Top-2 Acc": 0.9030},
    {"Exp": "EXP_002A", "Name": "Weighted CE", "Macro F1": 0.5741, "Weighted F1": 0.8066, "MCC": 0.7706, "Top-2 Acc": 0.9012},
    {"Exp": "EXP_002B", "Name": "Label Smooth", "Macro F1": 0.6191, "Weighted F1": 0.9181, "MCC": 0.9069, "Top-2 Acc": 0.9500},
    {"Exp": "EXP_002B_Ext", "Name": "LS 20 Ep", "Macro F1": 0.6143, "Weighted F1": 0.9158, "MCC": 0.9031, "Top-2 Acc": 0.9480},
    {"Exp": "EXP_002C", "Name": "CISA Tags", "Macro F1": 0.7057, "Weighted F1": 0.9416, "MCC": 0.9332, "Top-2 Acc": 0.9680},
    {"Exp": "EXP_002D", "Name": "Targeted Synth", "Macro F1": 0.7296, "Weighted F1": 0.9438, "MCC": 0.9341, "Top-2 Acc": 0.9682},
    {"Exp": "EXP_002E", "Name": "Gemini Synth", "Macro F1": 0.6950, "Weighted F1": 0.9410, "MCC": 0.9310, "Top-2 Acc": 0.9660},
    {"Exp": "EXP_002F", "Name": "Claude Synth", "Macro F1": 0.7142, "Weighted F1": 0.9430, "MCC": 0.9348, "Top-2 Acc": 0.9680},
    {"Exp": "EXP_002G", "Name": "FP32 Ablation (Final)", "Macro F1": 0.7121, "Weighted F1": 0.9454, "MCC": 0.9364, "Top-2 Acc": 0.9695}
]
df_sum = pd.DataFrame(summary_data)

# --- Plot 6: Macro F1 Across Experiments ---
plt.figure(figsize=(10, 5))
bars = plt.bar(df_sum["Exp"] + "\n(" + df_sum["Name"] + ")", df_sum["Macro F1"], color="#1f77b4", edgecolor="black", width=0.6)
bars[5].set_color("#17becf") # EXP_002D
bars[8].set_color("#2ca02c") # EXP_002G final
plt.title("Macro F1 Across All Phase 1 Experiments")
plt.ylabel("Macro F1 Score")
plt.ylim(0.5, 0.8)
plt.grid(axis='y', linestyle="--", alpha=0.6)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f"{yval:.4f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(f"{out_dir}/macro_f1_across_experiments.png")
plt.close()

# --- Plot 7: Weighted F1 Across Experiments ---
plt.figure(figsize=(10, 5))
bars = plt.bar(df_sum["Exp"] + "\n(" + df_sum["Name"] + ")", df_sum["Weighted F1"], color="#2b5c8f", edgecolor="black", width=0.6)
bars[8].set_color("#2ca02c") # Final chosen
plt.title("Weighted F1 Across All Phase 1 Experiments (Deployment Performance)")
plt.ylabel("Weighted F1 Score")
plt.ylim(0.75, 0.98)
plt.grid(axis='y', linestyle="--", alpha=0.6)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.003, f"{yval:.4f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(f"{out_dir}/weighted_f1_across_experiments.png")
plt.close()

# --- Plot 8: MCC Across Experiments ---
plt.figure(figsize=(10, 5))
bars = plt.bar(df_sum["Exp"] + "\n(" + df_sum["Name"] + ")", df_sum["MCC"], color="#6a3d9a", edgecolor="black", width=0.6)
bars[8].set_color("#2ca02c") # Final chosen
plt.title("Matthews Correlation Coefficient (MCC) Across All Phase 1 Experiments")
plt.ylabel("MCC Score")
plt.ylim(0.72, 0.97)
plt.grid(axis='y', linestyle="--", alpha=0.6)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.003, f"{yval:.4f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(f"{out_dir}/mcc_across_experiments.png")
plt.close()

print("All plots generated successfully in results/plots/")
