import os
import random
import numpy as np
import pandas as pd
import torch
import sys

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# Ensure the src folder is in the Python path
sys.path.append(os.getcwd())

# Import SecBERT specific components
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from src.secbert.dataset import SecBERTDataset

# Import PPO components
from src.ppo.networks import ActorNetwork

def main():
    print("=" * 50)
    print("Diagnostic Baseline Evaluation on VALIDATION Set")
    print("=" * 50)
    
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    TEST_PATH = "data/processed/val.csv"  # Changed to val.csv
    SECBERT_PATH = "models/secbert_finetuned"
    PPO_PATH = "models/ppo/best_ppo_policy.pt"
    ACTION_SPACE_PATH = "data/action_space.csv"
    
    # Verify Paths
    for p in [TEST_PATH, SECBERT_PATH, PPO_PATH, ACTION_SPACE_PATH]:
        if not os.path.exists(p):
            print(f"Error: Could not find {p}")
            return

    # Load Data
    test_df = pd.read_csv(TEST_PATH)
    print(f"\nValidation samples: {len(test_df)}")

    # Load Action Names
    action_space_df = pd.read_csv(ACTION_SPACE_PATH)
    ACTION_NAMES = dict(zip(action_space_df['Index'], action_space_df['Action Name']))
    PHASE_MAP = dict(zip(action_space_df['Index'], action_space_df['CISA Phase']))

    # Preprocess text
    def prepend_phase(df):
        df_copy = df.copy()
        df_copy['text'] = df_copy.apply(lambda row: f"[{PHASE_MAP[row['action_label']]}] {row['text']}", axis=1)
        return df_copy

    test_df_prep = prepend_phase(test_df)
    test_texts = test_df_prep['text'].tolist()
    test_labels = test_df_prep['action_label'].tolist()

    # Tokenizer & Dataloader
    tokenizer = AutoTokenizer.from_pretrained(SECBERT_PATH)
    test_dataset = SecBERTDataset(test_texts, test_labels, tokenizer, max_length=512)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collator)

    # SECBERT EVALUATION
    print("\n--- Evaluating SecBERT Baseline on Val ---")
    secbert_classifier = AutoModelForSequenceClassification.from_pretrained(SECBERT_PATH)
    secbert_classifier.to(device)
    secbert_classifier.eval()

    y_true_secbert = []
    y_pred_secbert = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            print(f"\rSecBERT Batch {i+1}/{len(test_loader)}", end="")
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = secbert_classifier(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)

            y_pred_secbert.extend(preds.cpu().numpy())
            y_true_secbert.extend(labels.cpu().numpy())
            
    print("\nSecBERT Evaluation Done.")
    secbert_accuracy = accuracy_score(y_true_secbert, y_pred_secbert)
    secbert_macro_f1 = f1_score(y_true_secbert, y_pred_secbert, average="macro", zero_division=0)
    secbert_weighted_f1 = f1_score(y_true_secbert, y_pred_secbert, average="weighted", zero_division=0)
    secbert_mcc = matthews_corrcoef(y_true_secbert, y_pred_secbert)

    # PPO EVALUATION
    print("\n--- Evaluating SecBERT + PPO on Val ---")
    encoder_only = getattr(secbert_classifier, secbert_classifier.base_model_prefix)
    encoder_only.eval()

    state_dim = 768
    action_dim = 20
    actor = ActorNetwork(state_dim, action_dim).to(device)

    checkpoint = torch.load(PPO_PATH, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'actor_state_dict' in checkpoint:
        actor.load_state_dict(checkpoint['actor_state_dict'])
    else:
        actor.load_state_dict(checkpoint)
        
    actor.eval()

    y_true_ppo = []
    y_pred_ppo = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            print(f"\rPPO Batch {i+1}/{len(test_loader)}", end="")
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = encoder_only(input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state[:, 0, :]

            action_logits = actor(embeddings)
            preds = torch.argmax(action_logits, dim=-1)

            y_pred_ppo.extend(preds.cpu().numpy())
            y_true_ppo.extend(labels.cpu().numpy())

    print("\nPPO Evaluation Done.")
    ppo_accuracy = accuracy_score(y_true_ppo, y_pred_ppo)
    ppo_macro_f1 = f1_score(y_true_ppo, y_pred_ppo, average="macro", zero_division=0)
    ppo_weighted_f1 = f1_score(y_true_ppo, y_pred_ppo, average="weighted", zero_division=0)
    ppo_mcc = matthews_corrcoef(y_true_ppo, y_pred_ppo)

    # METRICS COMPARISON
    print("\n" + "=" * 50)
    print("FINAL VALIDATION COMPARISON RESULTS")
    print("=" * 50)
    
    print(f"{'Metric':<15} | {'SecBERT':<15} | {'SecBERT + PPO':<15} | {'Diff'}")
    print("-" * 60)
    print(f"{'Accuracy':<15} | {secbert_accuracy:<15.4f} | {ppo_accuracy:<15.4f} | {ppo_accuracy - secbert_accuracy:+.4f}")
    print(f"{'Macro F1':<15} | {secbert_macro_f1:<15.4f} | {ppo_macro_f1:<15.4f} | {ppo_macro_f1 - secbert_macro_f1:+.4f}")
    print(f"{'Weighted F1':<15} | {secbert_weighted_f1:<15.4f} | {ppo_weighted_f1:<15.4f} | {ppo_weighted_f1 - secbert_weighted_f1:+.4f}")
    print(f"{'MCC':<15} | {secbert_mcc:<15.4f} | {ppo_mcc:<15.4f} | {ppo_mcc - secbert_mcc:+.4f}")
    
    # PER-CLASS COMPARISON
    print("\n--- Per-Class F1 Score Comparison ---")
    secbert_report = classification_report(
        y_true_secbert, y_pred_secbert,
        labels=list(ACTION_NAMES.keys()), target_names=list(ACTION_NAMES.values()),
        output_dict=True, zero_division=0
    )

    ppo_report = classification_report(
        y_true_ppo, y_pred_ppo,
        labels=list(ACTION_NAMES.keys()), target_names=list(ACTION_NAMES.values()),
        output_dict=True, zero_division=0
    )

    print(f"{'Action':<35} | {'Support':<7} | {'SecBERT F1':<12} | {'PPO F1':<12} | {'Diff'}")
    print("-" * 85)
    for action_name in ACTION_NAMES.values():
        s_f1 = secbert_report[action_name]['f1-score']
        p_f1 = ppo_report[action_name]['f1-score']
        diff = p_f1 - s_f1
        support = secbert_report[action_name]['support']
        print(f"{action_name:<35} | {support:<7} | {s_f1:<12.4f} | {p_f1:<12.4f} | {diff:+.4f}")

    print("\nGenerating validation confusion matrices...")
    os.makedirs("docs/evaluation_results", exist_ok=True)
    
    # Plot SecBERT CM
    cm_sec = confusion_matrix(y_true_secbert, y_pred_secbert, labels=list(ACTION_NAMES.keys()))
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_sec, display_labels=list(ACTION_NAMES.values()))
    disp.plot(cmap="Blues", ax=ax, xticks_rotation='vertical')
    plt.title("Validation Confusion Matrix: Fine-tuned SecBERT")
    plt.tight_layout()
    plt.savefig("docs/evaluation_results/val_cm_secbert.png")
    plt.close()
    
    # Plot PPO CM
    cm_ppo = confusion_matrix(y_true_ppo, y_pred_ppo, labels=list(ACTION_NAMES.keys()))
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_ppo, display_labels=list(ACTION_NAMES.values()))
    disp.plot(cmap="Blues", ax=ax, xticks_rotation='vertical')
    plt.title("Validation Confusion Matrix: SecBERT + PPO")
    plt.tight_layout()
    plt.savefig("docs/evaluation_results/val_cm_ppo.png")
    plt.close()
    
    print("Validation confusion matrices saved to docs/evaluation_results/")

if __name__ == "__main__":
    main()
