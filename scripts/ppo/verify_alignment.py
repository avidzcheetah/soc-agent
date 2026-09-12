#!/usr/bin/env python3
"""
Experiment 4A - Subtask 2.1: Embedding Alignment Verification

This script verifies that the 768-dimensional CLS embedding produced by the
`SecBERTStateEncoder` (used as the state representation for PPO) is mathematically
identical to the 768-dimensional embedding fed into the `BertForSequenceClassification`
classifier head during Phase 1 training.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.encoder import SecBERTStateEncoder

def main():
    model_path = "models/secbert_finetuned"
    device = torch.device("cpu")

    # 1. Load components
    teacher = AutoModelForSequenceClassification.from_pretrained(model_path)
    teacher.eval()
    
    ppo_encoder = SecBERTStateEncoder(checkpoint_path=model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 2. Get 10 representative samples from validation data
    val_data_path = "data/processed/val.csv"
    action_space_path = "data/action_space.csv"
    
    val_df = pd.read_csv(val_data_path)
    action_space_df = pd.read_csv(action_space_path)
    phase_map = dict(zip(action_space_df["Index"], action_space_df["CISA Phase"]))

    # Sample 10 incidents from different classes using robust drop_duplicates
    val_df_sampled = val_df.drop_duplicates(subset=['action_label']).head(10)
    
    test_texts = [
        f"[{phase_map[int(row['action_label'])]}] {row['text']}" 
        for _, row in val_df_sampled.iterrows()
    ]

    max_diffs = []
    mean_diffs = []
    cosine_sims = []
    allclose_results = []

    for text in test_texts:
        # --- Teacher Pathway ---
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = teacher.bert(**inputs)
            # The exact tensor passed to `teacher.classifier` is the pooler_output
            teacher_embedding = outputs.pooler_output.squeeze(0).numpy()

        # --- PPO Pathway ---
        ppo_embedding = ppo_encoder.encode_incident(text).numpy()

        # --- Comparison ---
        diff = np.abs(teacher_embedding - ppo_embedding)
        max_diffs.append(np.max(diff))
        mean_diffs.append(np.mean(diff))
        
        # Cosine similarity
        cos_sim = np.dot(teacher_embedding, ppo_embedding) / (np.linalg.norm(teacher_embedding) * np.linalg.norm(ppo_embedding))
        cosine_sims.append(cos_sim)
        
        # Allclose
        allclose_results.append(np.allclose(teacher_embedding, ppo_embedding, atol=1e-5))

    # Calculate aggregates
    overall_max_diff = max(max_diffs)
    overall_mean_diff = np.mean(mean_diffs)
    overall_cos_sim = np.mean(cosine_sims)
    overall_allclose = all(allclose_results)

    # Print the exact requested report format
    print("\nEmbedding Alignment Verification")
    print("================================")
    print(f"\nCheckpoint: {model_path}")
    print(f"Hidden size: 768")
    print(f"\nSamples tested: {len(test_texts)}")
    print(f"\nMaximum absolute difference : {overall_max_diff:.8e}")
    print(f"Mean absolute difference    : {overall_mean_diff:.8e}")
    print(f"Cosine similarity           : {overall_cos_sim:.6f}")
    print(f"Allclose                    : {'PASS' if overall_allclose else 'FAIL'}")

    print()
    if overall_allclose:
        print("[PASS] SecBERTStateEncoder produces")
        print("       the exact classifier-input representation.")
    else:
        print("[FAIL] Mismatch detected. Do not proceed to distillation.")

if __name__ == "__main__":
    main()
