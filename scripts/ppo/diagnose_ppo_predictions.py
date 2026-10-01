import os
import random
import numpy as np
import pandas as pd
import torch
import sys
from collections import Counter

# Ensure the src folder is in the Python path
sys.path.append(os.getcwd())
torch.set_num_threads(4)

from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from src.secbert.dataset import SecBERTDataset
from src.ppo.networks import ActorNetwork

def main():
    print("=" * 70, flush=True)
    print("PPO PREDICTION DISTRIBUTION & CONFUSION ANALYSIS (VALIDATION SET)", flush=True)
    print("=" * 70, flush=True)

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    VAL_PATH = "data/processed/val.csv"
    SECBERT_PATH = "models/secbert_finetuned"
    PPO_PATH = "models/ppo/best_ppo_policy.pt"
    ACTION_SPACE_PATH = "data/action_space.csv"

    for p in [VAL_PATH, SECBERT_PATH, PPO_PATH, ACTION_SPACE_PATH]:
        if not os.path.exists(p):
            print(f"Error: Missing {p}", flush=True)
            return

    val_df = pd.read_csv(VAL_PATH)
    action_space_df = pd.read_csv(ACTION_SPACE_PATH)
    ACTION_NAMES = dict(zip(action_space_df['Index'], action_space_df['Action Name']))
    PHASE_MAP = dict(zip(action_space_df['Index'], action_space_df['CISA Phase']))

    # Preprocess text with phase prepended (consistent with evaluate_validation.py)
    val_df_prep = val_df.copy()
    val_df_prep['text'] = val_df_prep.apply(lambda r: f"[{PHASE_MAP[r['action_label']]}] {r['text']}", axis=1)
    texts = val_df_prep['text'].tolist()
    labels = val_df_prep['action_label'].tolist()

    tokenizer = AutoTokenizer.from_pretrained(SECBERT_PATH)
    dataset = SecBERTDataset(texts, labels, tokenizer, max_length=512)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collator)

    # 1. SecBERT Inference
    print("\n[*] Running SecBERT baseline inference...", flush=True)
    secbert_model = AutoModelForSequenceClassification.from_pretrained(SECBERT_PATH).to(device)
    secbert_model.eval()

    y_true = []
    y_pred_secbert = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            print(f"\rSecBERT Batch {i+1}/{len(loader)}", end="", flush=True)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            lbls = batch["labels"].to(device)

            outputs = secbert_model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)

            y_pred_secbert.extend(preds.cpu().numpy().tolist())
            y_true.extend(lbls.cpu().numpy().tolist())

    print("\nSecBERT done.", flush=True)

    # 2. PPO Inference
    print("[*] Running PPO policy inference...", flush=True)
    encoder = getattr(secbert_model, secbert_model.base_model_prefix)
    encoder.eval()

    actor = ActorNetwork(state_dim=768, action_dim=20).to(device)
    checkpoint = torch.load(PPO_PATH, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'actor_state_dict' in checkpoint:
        actor.load_state_dict(checkpoint['actor_state_dict'])
    else:
        actor.load_state_dict(checkpoint)
    actor.eval()

    y_pred_ppo = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            print(f"\rPPO Batch {i+1}/{len(loader)}", end="", flush=True)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = encoder(input_ids, attention_mask=attention_mask)
            cls_emb = outputs.last_hidden_state[:, 0, :]
            logits = actor(cls_emb)
            preds = torch.argmax(logits, dim=-1)

            y_pred_ppo.extend(preds.cpu().numpy().tolist())

    print("\nPPO done.", flush=True)

    y_true = np.array(y_true)
    y_pred_secbert = np.array(y_pred_secbert)
    y_pred_ppo = np.array(y_pred_ppo)

    # Global Distribution Comparison
    print("\n" + "=" * 70)
    print("OVERALL ACTION DISTRIBUTION: TRUE vs SECBERT vs PPO")
    print("=" * 70)
    print(f"{'Action Name':<28} | {'True':<6} | {'SecBERT Pred':<13} | {'PPO Pred':<10} | {'PPO Shift'}")
    print("-" * 75)

    true_counts = Counter(y_true)
    sec_counts = Counter(y_pred_secbert)
    ppo_counts = Counter(y_pred_ppo)

    for idx in range(20):
        name = ACTION_NAMES.get(idx, f"action_{idx}")
        t_c = true_counts.get(idx, 0)
        s_c = sec_counts.get(idx, 0)
        p_c = ppo_counts.get(idx, 0)
        diff = p_c - t_c
        diff_str = f"+{diff}" if diff > 0 else str(diff)
        print(f"{name:<28} | {t_c:<6} | {s_c:<13} | {p_c:<10} | {diff_str}")

    # Specific Target Action Breakdown: where do misclassifications go?
    targets = [
        ("enable_deep_logging", 1),
        ("block_dest_ip", 4),
        ("escalate_to_human", 19),
        ("block_port", 6),
        ("restore_registry", 14),
        ("snapshot_forensics", 17)
    ]

    print("\n" + "=" * 70)
    print("DETAILED MISCLASSIFICATION MAPPING FOR TARGET MINORITY ACTIONS")
    print("=" * 70)

    for action_name, action_idx in targets:
        mask = (y_true == action_idx)
        count = int(np.sum(mask))
        if count == 0:
            print(f"\n--- {action_name} (idx {action_idx}): 0 samples in validation ---")
            continue

        ppo_preds_for_class = y_pred_ppo[mask]
        sec_preds_for_class = y_pred_secbert[mask]

        correct_ppo = int(np.sum(ppo_preds_for_class == action_idx))
        correct_sec = int(np.sum(sec_preds_for_class == action_idx))

        print(f"\nTarget: {action_name} (Index {action_idx}) | Total Ground Truth: {count}")
        print(f"  SecBERT correct : {correct_sec}/{count} ({correct_sec/count*100:.1f}%)")
        print(f"  PPO correct     : {correct_ppo}/{count} ({correct_ppo/count*100:.1f}%)")
        print("  PPO Predictions Breakdown:")
        
        pred_dist = Counter(ppo_preds_for_class)
        for pred_idx, pred_cnt in pred_dist.most_common():
            pred_name = ACTION_NAMES.get(pred_idx, f"action_{pred_idx}")
            pred_phase = PHASE_MAP.get(pred_idx, "Unknown")
            true_phase = PHASE_MAP.get(action_idx, "Unknown")
            same_phase = "(SAME PHASE)" if pred_phase == true_phase else f"(DIFF PHASE: {pred_phase})"
            status = "[CORRECT]" if pred_idx == action_idx else f"[MISCLASSIFIED -> {pred_name}]"
            print(f"    - {pred_name} (idx {pred_idx}): {pred_cnt}/{count} ({pred_cnt/count*100:.1f}%) {status} {same_phase}")

    print("\n" + "=" * 70)
    print("CONFUSION SUMMARY FOR ALL MISCLASSIFIED SAMPLES BY PPO")
    print("=" * 70)
    wrong_mask = (y_true != y_pred_ppo)
    total_wrong = int(np.sum(wrong_mask))
    print(f"Total PPO validation errors: {total_wrong} / {len(y_true)} ({total_wrong/len(y_true)*100:.2f}%)")
    
    # What are the most common predicted actions when PPO makes an error?
    error_pred_counts = Counter(y_pred_ppo[wrong_mask])
    print("\nWhen PPO makes an ERROR, what does it falsely predict most often?")
    for pred_idx, cnt in error_pred_counts.most_common(8):
        pred_name = ACTION_NAMES.get(pred_idx, f"action_{pred_idx}")
        print(f"  - {pred_name:<25}: {cnt:>3} times ({cnt/total_wrong*100:.1f}% of all errors)")

if __name__ == "__main__":
    main()
