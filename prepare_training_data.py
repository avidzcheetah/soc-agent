import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import triu
import time

def process_datasets():
    print("=== SOC Agent Data Preparation Pipeline ===")
    
    # 1. Load your existing processed files
    cissm_path = 'data/processed/cissm_processed.csv'
    rcatt_path = 'data/processed/rcatt_processed.csv'
    
    if not os.path.exists(cissm_path) or not os.path.exists(rcatt_path):
        raise FileNotFoundError("Processed datasets not found.")
        
    df_cissm = pd.read_csv(cissm_path)
    df_rcatt = pd.read_csv(rcatt_path)
    
    # 2. Add the source_dataset tag
    df_cissm['source_dataset'] = 'CISSM'
    df_rcatt['source_dataset'] = 'rcATT'
    
    # Extract only the needed columns based on the protocol
    df_cissm = df_cissm[['text', 'action_label', 'source_dataset']]
    df_rcatt = df_rcatt[['text', 'action_label', 'source_dataset']]
    
    # 3. Combine them into one dataframe
    df = pd.concat([df_cissm, df_rcatt], ignore_index=True)
    initial_count = len(df)
    print(f"Initial combined dataset size: {initial_count} records")
    
    # 4. Apply the strict Missing Value & Garbage policy
    print("Applying strict garbage filtering...")
    
    # Drop NaNs
    df.dropna(subset=['text', 'action_label'], inplace=True)
    
    # Convert to string to avoid issues
    df['text'] = df['text'].astype(str)
    
    # Strip whitespace
    df['text'] = df['text'].str.strip()
    
    # Define garbage masks
    garbage_mask = df['text'].str.lower().isin(['', 'null', 'nan', 'n/a', 'unknown'])
    length_mask = df['text'].str.len() < 15
    
    # Apply filtering
    df = df[~(garbage_mask | length_mask)]
    df['action_label'] = df['action_label'].astype(int)
    
    clean_count = len(df)
    print(f"Dropped {initial_count - clean_count} garbage/short records.")
    
    # 4.5 Label Validation against action_space.csv
    print("Validating labels against action_space.csv...")
    action_space_path = 'data/action_space.csv'
    if not os.path.exists(action_space_path):
        raise FileNotFoundError(f"Missing {action_space_path}")
    
    df_action_space = pd.read_csv(action_space_path)
    valid_labels = set(df_action_space['Index'].astype(int))
    
    unique_labels = set(df['action_label'])
    invalid_labels = unique_labels - valid_labels
    
    if invalid_labels:
        raise ValueError(f"CRITICAL FAILURE: Found invalid action_labels in data that are not in action_space.csv: {invalid_labels}")
        
    if not all(0 <= label <= 19 for label in unique_labels):
        raise ValueError(f"CRITICAL FAILURE: action_labels out of bounds (0-19): {unique_labels}")
        
    print("Label Validation Passed: All labels exist in action_space.csv and are bounded 0-19.")
    
    # 5. Exact Duplicate Removal
    df.drop_duplicates(subset=['text'], inplace=True)
    exact_dedup_count = len(df)
    print(f"Dropped {clean_count - exact_dedup_count} exact duplicates.")
    
    # 6. Near-Duplicate Detection (TF-IDF + Cosine Similarity)
    print("Running Near-Duplicate Detection (TF-IDF Cosine Similarity >= 0.90)...")
    start_time = time.time()
    
    # Reset index for clean mapping
    df.reset_index(drop=True, inplace=True)
    
    # Create TF-IDF Vectorizer (use character n-grams + word n-grams for robustness against IP changes)
    vectorizer = TfidfVectorizer(max_features=10000, analyzer='word', ngram_range=(1, 3))
    tfidf_matrix = vectorizer.fit_transform(df['text'])
    
    # Compute sparse cosine similarity matrix: dot product of L2 normalized tfidf vectors
    # TfidfVectorizer outputs L2 normalized vectors by default.
    similarity_matrix = tfidf_matrix * tfidf_matrix.T
    
    # Get the upper triangle of the similarity matrix, excluding the diagonal (k=1)
    upper_tri = triu(similarity_matrix, k=1)
    
    # Find coordinates where similarity is >= 0.90
    row_indices, col_indices = upper_tri.nonzero()
    similarities = upper_tri.data
    
    # Filter for threshold
    threshold = 0.90
    high_sim_mask = similarities >= threshold
    
    # The columns represent the "duplicate" documents that we want to drop
    cols_to_drop = col_indices[high_sim_mask]
    
    # Get unique indices to drop
    indices_to_drop = set(cols_to_drop)
    
    print(f"Found {len(indices_to_drop)} near-duplicates in {time.time() - start_time:.2f} seconds.")
    
    # Drop them
    df.drop(index=list(indices_to_drop), inplace=True)
    
    final_count = len(df)
    print(f"Dropped {exact_dedup_count - final_count} near-duplicates.")
    print(f"Final pristine dataset size: {final_count} records.")
    
    # 7. Save final dataset
    output_path = 'data/processed/final_training_data.csv'
    df.to_csv(output_path, index=False)
    print(f"Saved pristine training data to {output_path}")

if __name__ == '__main__':
    process_datasets()
