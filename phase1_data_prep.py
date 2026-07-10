import pandas as pd
import numpy as np
import os
import random
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: torch not found. Skipping torch seed initialization.")
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from scipy.sparse import triu
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import time

def set_seeds(seed=42):
    print(f"[Step 11] Setting all random seeds to {seed}...")
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def phase1():
    print("=== PHASE 1: BEFORE SECBERT TRAINING ===")
    
    # Set seeds early for reproducible tf-idf breaking ties etc.
    set_seeds(42)

    # 1. Verify Dataset Schema & 8. Merge
    print("\n[Step 1 & 8] Loading and Merging Datasets...")
    cissm_path = 'data/processed/cissm_processed.csv'
    rcatt_path = 'data/processed/rcatt_processed.csv'
    
    df_cissm = pd.read_csv(cissm_path)[['text', 'action_label']]
    df_cissm['source_dataset'] = 'CISSM'
    
    df_rcatt = pd.read_csv(rcatt_path)[['text', 'action_label']]
    df_rcatt['source_dataset'] = 'rcATT'
    
    df = pd.concat([df_cissm, df_rcatt], ignore_index=True)
    initial_count = len(df)
    print(f"Merged {initial_count} records.")

    # 2. Validate Labels
    print("\n[Step 2] Validating Labels...")
    action_space_path = 'data/action_space.csv'
    df_action_space = pd.read_csv(action_space_path)
    valid_labels = set(df_action_space['Index'].astype(int))
    unique_labels = set(df['action_label'])
    
    invalid_labels = unique_labels - valid_labels
    if invalid_labels:
        raise ValueError(f"CRITICAL FAILURE: Invalid action_labels found: {invalid_labels}")
    if not all(0 <= label <= 19 for label in unique_labels):
        raise ValueError(f"CRITICAL FAILURE: action_labels out of bounds (0-19)")
    print("Label Validation Passed.")

    # 3. Clean the Text
    print("\n[Step 3] Cleaning Text (Missing, Whitespace, Garbage)...")
    df.dropna(subset=['text', 'action_label'], inplace=True)
    df['text'] = df['text'].astype(str).str.strip()
    garbage_mask = df['text'].str.lower().isin(['', 'null', 'nan', 'n/a', 'unknown'])
    length_mask = df['text'].str.len() < 15
    df = df[~(garbage_mask | length_mask)]
    df['action_label'] = df['action_label'].astype(int)
    print(f"Dropped {initial_count - len(df)} garbage/short records.")

    # 4. Remove Duplicates
    print("\n[Step 4] Removing Duplicates (Exact & Near)...")
    df.drop_duplicates(subset=['text'], inplace=True)
    exact_dedup_count = len(df)
    print(f"Dropped exact duplicates. Remaining: {exact_dedup_count}")
    
    print("Running TF-IDF Cosine Similarity for Near-Duplicates (Threshold >= 0.90)...")
    df.reset_index(drop=True, inplace=True)
    vectorizer = TfidfVectorizer(max_features=10000, analyzer='word', ngram_range=(1, 3))
    tfidf_matrix = vectorizer.fit_transform(df['text'])
    similarity_matrix = tfidf_matrix * tfidf_matrix.T
    upper_tri = triu(similarity_matrix, k=1)
    row_indices, col_indices = upper_tri.nonzero()
    high_sim_mask = upper_tri.data >= 0.90
    cols_to_drop = col_indices[high_sim_mask]
    indices_to_drop = set(cols_to_drop)
    df.drop(index=list(indices_to_drop), inplace=True)
    final_count = len(df)
    print(f"Dropped {exact_dedup_count - final_count} near-duplicates. Pristine count: {final_count}")

    # 7. Analyze Token Lengths
    print("\n[Step 7] Analyzing Token Lengths with SecBERT Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('jackaduma/SecBERT')
    token_lengths = df['text'].apply(lambda x: len(tokenizer.encode(str(x), add_special_tokens=True)))
    
    avg_len = token_lengths.mean()
    p95_len = np.percentile(token_lengths, 95)
    max_len = token_lengths.max()
    print(f"Average Token Length: {avg_len:.2f}")
    print(f"95th Percentile: {p95_len}")
    print(f"Maximum Token Length: {max_len}")
    
    if p95_len <= 512:
        print("-> MAX_LEN = 512 is appropriate and safe.")
    else:
        print("-> WARNING: 95th percentile exceeds 512 tokens. Heavy truncation will occur.")

    # 5. Analyze Class Distribution
    print("\n[Step 5] Analyzing Class Distribution...")
    class_counts = df['action_label'].value_counts().sort_index()
    class_percentages = df['action_label'].value_counts(normalize=True).sort_index() * 100
    print("Class Counts:")
    print(class_counts)
    print("\nClass Percentages (%):")
    print(class_percentages)
    
    plt.figure(figsize=(10,6))
    class_counts.plot(kind='bar')
    plt.title('Action Class Distribution')
    plt.xlabel('Action Label')
    plt.ylabel('Count')
    plt.savefig('results/class_distribution.png')
    print("Saved histogram to results/class_distribution.png")

    # 6. Compute Class Weights
    print("\n[Step 6] Computing Class Weights...")
    classes = np.unique(df['action_label'])
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=df['action_label'])
    class_weights_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    
    # Inject 0.0 weight for entirely missing classes (e.g. 2 and 18) to ensure 20 logits are supported
    for i in range(20):
        if i not in class_weights_dict:
            class_weights_dict[i] = 0.0
            print(f"-> Class {i} is completely missing! Assigned weight 0.0")
            
    print(f"Computed weights: {class_weights_dict}")
    with open('data/class_weights.json', 'w') as f:
        json.dump(class_weights_dict, f, indent=4)
    print("Saved class weights to data/class_weights.json")

    # 9. Shuffle & 10. Split the data
    print("\n[Step 9 & 10] Shuffling and Splitting (80/10/10)...")
    # First split 80% train, 20% temp
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    for train_idx, temp_idx in splitter.split(df, df['action_label']):
        train_df = df.iloc[train_idx]
        temp_df = df.iloc[temp_idx]
        
    # Then split the 20% temp into 10% val, 10% test (which is 50% of the temp)
    splitter_temp = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    for val_idx, test_idx in splitter_temp.split(temp_df, temp_df['action_label']):
        val_df = temp_df.iloc[val_idx]
        test_df = temp_df.iloc[test_idx]
        
    print(f"Training Set: {len(train_df)} records")
    print(f"Validation Set: {len(val_df)} records")
    print(f"Test Set: {len(test_df)} records")
    
    train_df.to_csv('data/processed/train.csv', index=False)
    val_df.to_csv('data/processed/val.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)
    print("Saved train.csv, val.csv, and test.csv to data/processed/")
    
    print("\n=== PHASE 1 COMPLETE ===")

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    phase1()
