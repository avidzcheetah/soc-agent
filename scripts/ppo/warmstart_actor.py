#!/usr/bin/env python3
"""
Experiment 4A: Behavioral Cloning Warm-Start for PPO Actor

Distills the SecBERT classifier's learned decision boundaries into the PPO Actor
MLP using knowledge distillation (KL-divergence). This gives PPO a strong
initialization that already handles minority classes, instead of starting from
random Xavier weights.

Usage:
    python scripts/ppo/warmstart_actor.py

Output:
    models/ppo/warmstart_actor.pt  — Pre-trained Actor state_dict
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from src.secbert.dataset import SecBERTDataset
from src.ppo.networks import ActorNetwork

# ── Configuration ────────────────────────────────────────────────
SEED = 42
SECBERT_PATH = "models/secbert_finetuned"
TRAIN_DATA_PATH = "data/processed/train.csv"
VAL_DATA_PATH = "data/processed/val.csv"
ACTION_SPACE_PATH = "data/action_space.csv"
OUTPUT_PATH = "models/ppo/warmstart_actor.pt"

# Knowledge Distillation hyperparameters
TEMPERATURE = 2.0        # Softens teacher probabilities for richer gradient signal
KD_EPOCHS = 50           # Number of distillation training epochs
KD_LR = 1e-3             # Learning rate for Actor MLP during distillation
KD_BATCH_SIZE = 128      # Batch size for distillation training
HARD_LABEL_WEIGHT = 0.3  # Weight for hard label cross-entropy (auxiliary)
SOFT_LABEL_WEIGHT = 0.7  # Weight for KL-divergence with teacher soft labels


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("Experiment 4A: Behavioral Cloning Warm-Start")
    print("=" * 70)
    print(f"  Device:         {device}")
    print(f"  SecBERT Model:  {SECBERT_PATH}")
    print(f"  Train Data:     {TRAIN_DATA_PATH}")
    print(f"  Output:         {OUTPUT_PATH}")
    print(f"  Temperature:    {TEMPERATURE}")
    print(f"  Epochs:         {KD_EPOCHS}")
    print()

    # ── Step 1: Load the full SecBERT classifier (teacher) ───────
    print("[1/5] Loading SecBERT classifier (teacher model)...")
    import pandas as pd
    action_space_df = pd.read_csv(ACTION_SPACE_PATH)
    PHASE_MAP = dict(zip(action_space_df["Index"], action_space_df["CISA Phase"]))

    teacher = AutoModelForSequenceClassification.from_pretrained(SECBERT_PATH)
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(SECBERT_PATH)
    print(f"  Teacher loaded: {teacher.config.architectures}")
    print(f"  Classifier head: {teacher.classifier.weight.shape}")  # [20, 768]

    # ── Step 2: Pre-compute CLS embeddings and teacher logits ────
    print("\n[2/5] Pre-computing CLS embeddings and teacher logits for all training data...")
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    # Prepend CISA phase tags (same as during Phase 1 training)
    train_df_prep = train_df.copy()
    train_df_prep["text"] = train_df_prep.apply(
        lambda row: f"[{PHASE_MAP[row['action_label']]}] {row['text']}", axis=1
    )

    texts = train_df_prep["text"].tolist()
    labels = train_df_prep["action_label"].tolist()

    # Create dataset and dataloader
    train_dataset = SecBERTDataset(texts, labels, tokenizer, max_length=512)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    data_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, collate_fn=collator)

    # Extract the base encoder from the teacher (without classifier head)
    encoder = getattr(teacher, teacher.base_model_prefix)
    encoder.eval()

    all_embeddings = []
    all_teacher_logits = []
    all_labels = []

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            print(f"\r  Batch {i+1}/{len(data_loader)}", end="", flush=True)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"]

            # Get CLS embedding (pooler_output matches what the classifier expects)
            outputs = encoder(input_ids, attention_mask=attention_mask)
            cls_embeddings = outputs.pooler_output  # [batch, 768]

            # Get teacher's full classifier logits directly from the embedding
            # (bypasses a second redundant forward pass through the 110M param BERT)
            teacher_logits = teacher.classifier(cls_embeddings)  # [batch, 20]

            all_embeddings.append(cls_embeddings.cpu())
            all_teacher_logits.append(teacher_logits.cpu())
            all_labels.append(batch_labels)

    all_embeddings = torch.cat(all_embeddings, dim=0)        # [12312, 768]
    all_teacher_logits = torch.cat(all_teacher_logits, dim=0)  # [12312, 20]
    all_labels = torch.cat(all_labels, dim=0)                  # [12312]

    print(f"\n  Embeddings shape:     {all_embeddings.shape}")
    print(f"  Teacher logits shape: {all_teacher_logits.shape}")
    print(f"  Labels shape:         {all_labels.shape}")

    # Verify teacher accuracy on training data
    teacher_preds = torch.argmax(all_teacher_logits, dim=-1)
    teacher_acc = (teacher_preds == all_labels).float().mean().item()
    print(f"  Teacher train accuracy: {teacher_acc:.4%}")

    # ── Step 3: Also pre-compute validation embeddings for monitoring ──
    print("\n[3/5] Pre-computing validation embeddings...")
    val_df = pd.read_csv(VAL_DATA_PATH)
    val_df_prep = val_df.copy()
    val_df_prep["text"] = val_df_prep.apply(
        lambda row: f"[{PHASE_MAP[row['action_label']]}] {row['text']}", axis=1
    )

    val_dataset = SecBERTDataset(
        val_df_prep["text"].tolist(), val_df_prep["action_label"].tolist(),
        tokenizer, max_length=512
    )
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collator)

    val_embeddings = []
    val_labels_list = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = encoder(input_ids, attention_mask=attention_mask)
            val_embeddings.append(outputs.pooler_output.cpu())
            val_labels_list.append(batch["labels"])

    val_embeddings = torch.cat(val_embeddings, dim=0)
    val_labels_tensor = torch.cat(val_labels_list, dim=0)
    print(f"  Val embeddings shape: {val_embeddings.shape}")

    # ── Step 4: Knowledge Distillation Training ──────────────────
    print(f"\n[4/5] Training Actor MLP via Knowledge Distillation...")
    print(f"  Architecture: 768 → 512 → 256 → 20")
    print(f"  Temperature:  {TEMPERATURE}")
    print(f"  Soft weight:  {SOFT_LABEL_WEIGHT}, Hard weight: {HARD_LABEL_WEIGHT}")
    print(f"  Epochs:       {KD_EPOCHS}")
    print(f"  LR:           {KD_LR}")
    print()

    # Initialize student Actor with random weights
    student = ActorNetwork(state_dim=768, action_dim=20).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=KD_LR)

    # Create distillation dataset
    kd_dataset = TensorDataset(
        all_embeddings, all_teacher_logits, all_labels
    )
    kd_loader = DataLoader(kd_dataset, batch_size=KD_BATCH_SIZE, shuffle=True)

    best_val_acc = 0.0
    best_state_dict = None

    for epoch in range(1, KD_EPOCHS + 1):
        student.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for emb_batch, teacher_logit_batch, label_batch in kd_loader:
            emb_batch = emb_batch.to(device)
            teacher_logit_batch = teacher_logit_batch.to(device)
            label_batch = label_batch.to(device)

            # Student forward pass
            student_logits = student(emb_batch)

            # KL-Divergence loss (soft labels from teacher)
            # KL(teacher || student) at temperature T
            soft_teacher = F.log_softmax(teacher_logit_batch / TEMPERATURE, dim=-1)
            soft_student = F.log_softmax(student_logits / TEMPERATURE, dim=-1)
            kl_loss = F.kl_div(
                soft_student, soft_teacher.exp(),
                reduction="batchmean"
            ) * (TEMPERATURE ** 2)

            # Hard label cross-entropy loss (auxiliary)
            hard_loss = F.cross_entropy(student_logits, label_batch)

            # Combined loss
            loss = SOFT_LABEL_WEIGHT * kl_loss + HARD_LABEL_WEIGHT * hard_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * emb_batch.size(0)
            preds = torch.argmax(student_logits, dim=-1)
            epoch_correct += (preds == label_batch).sum().item()
            epoch_total += emb_batch.size(0)

        train_acc = epoch_correct / epoch_total
        avg_loss = epoch_loss / epoch_total

        # Validation evaluation
        student.eval()
        with torch.no_grad():
            val_emb_device = val_embeddings.to(device)
            val_logits = student(val_emb_device)
            val_preds = torch.argmax(val_logits, dim=-1).cpu()
            val_acc = (val_preds == val_labels_tensor).float().mean().item()

            # Compute Macro F1 on validation
            from sklearn.metrics import f1_score
            val_macro_f1 = f1_score(
                val_labels_tensor.numpy(), val_preds.numpy(),
                average="macro", zero_division=0, labels=list(range(20))
            )

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{KD_EPOCHS} | "
                f"Loss: {avg_loss:.4f} | "
                f"Train Acc: {train_acc:.4%} | "
                f"Val Acc: {val_acc:.4%} | "
                f"Val Macro F1: {val_macro_f1:.4f}"
            )

        # Save best based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_macro_f1
            best_state_dict = {k: v.clone().cpu() for k, v in student.state_dict().items()}

    print(f"\n  Best Val Acc: {best_val_acc:.4%}, Val Macro F1: {best_val_f1:.4f}")

    # ── Step 5: Save warm-started Actor weights ──────────────────
    print(f"\n[5/5] Saving warm-started Actor to {OUTPUT_PATH}...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    torch.save({
        "actor_state_dict": best_state_dict,
        "warmstart_val_acc": best_val_acc,
        "warmstart_val_macro_f1": best_val_f1,
        "temperature": TEMPERATURE,
        "epochs": KD_EPOCHS,
        "method": "behavioral_cloning_kd",
    }, OUTPUT_PATH)

    print(f"  Saved successfully.")
    print()
    print("=" * 70)
    print("Warm-Start Complete")
    print(f"  Actor initialized at Val Acc: {best_val_acc:.4%}")
    print(f"  Actor initialized at Val Macro F1: {best_val_f1:.4f}")
    print(f"  Next step: python scripts/ppo/train.py --warmstart_path {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
