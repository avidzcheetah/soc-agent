#!/usr/bin/env python3
"""
Diagnostic Script: Class 13 vs Class 15 — Training Data Separability Analysis

Analyzes the training dataset for classes 13 (remove_persistence) and 15
(restore_defense_config) to determine whether they are textually distinguishable.

Does not modify anything or train any model.
"""

import os
import sys
import textwrap
from collections import Counter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


ACTION_NAMES = {13: "remove_persistence", 15: "restore_defense_config"}


def keyword_analysis(texts, label):
    """Count samples containing persistence/defense-related keywords."""
    persistence_kw = [
        "persistence", "persist", "autorun", "startup", "scheduled task",
        "registry key", "service", "backdoor", "webshell", "cron",
        "boot", "logon", "run key", "schtask"
    ]
    defense_kw = [
        "defense", "defender", "antivirus", "av ", "security tool",
        "disable", "tamper", "log", "logging", "edr", "siem",
        "configuration", "config", "restore", "re-enable", "reenable",
        "firewall", "protection"
    ]

    print(f"\n  Keyword presence in class {label} ({ACTION_NAMES[label]}):")
    print(f"  {'Keyword':<25} {'Count':>6} {'%':>7}")
    print("  " + "-" * 40)

    for kw_group_name, kw_list in [("PERSISTENCE keywords", persistence_kw),
                                     ("DEFENSE keywords", defense_kw)]:
        print(f"\n  {kw_group_name}:")
        for kw in kw_list:
            count = sum(1 for t in texts if kw.lower() in t.lower())
            pct = count / len(texts) * 100 if texts else 0
            if count > 0:
                print(f"    {kw:<25} {count:>6} {pct:>6.1f}%")


def show_samples(texts, label, n=20):
    """Print first n samples truncated to 200 chars."""
    print(f"\n  Representative samples for class {label} ({ACTION_NAMES[label]}):")
    print("  " + "=" * 90)
    for i, text in enumerate(texts[:n]):
        truncated = text[:200].replace("\n", " ")
        print(f"  [{i+1:>2}] {truncated}...")
        print()


def main():
    print("=" * 90)
    print("Class 13 vs Class 15 — Training Data Separability Analysis")
    print("=" * 90)

    train_df = pd.read_csv("data/processed/train.csv")

    c13 = train_df[train_df["action_label"] == 13]
    c15 = train_df[train_df["action_label"] == 15]

    print(f"\n  Class 13 (remove_persistence):      {len(c13)} training samples")
    print(f"  Class 15 (restore_defense_config):   {len(c15)} training samples")
    print(f"  Ratio: {len(c13)/len(c15):.1f}:1")

    texts_13 = c13["text"].astype(str).tolist()
    texts_15 = c15["text"].astype(str).tolist()

    # ─── 1. Keyword Analysis ───
    print("\n" + "=" * 90)
    print("1. KEYWORD ANALYSIS")
    print("=" * 90)
    keyword_analysis(texts_13, 13)
    keyword_analysis(texts_15, 15)

    # ─── 2. Text Length Statistics ───
    print("\n" + "=" * 90)
    print("2. TEXT LENGTH STATISTICS")
    print("=" * 90)
    for label, texts in [(13, texts_13), (15, texts_15)]:
        lengths = [len(t) for t in texts]
        print(f"\n  Class {label} ({ACTION_NAMES[label]}):")
        print(f"    Mean length: {np.mean(lengths):.0f} chars")
        print(f"    Median:      {np.median(lengths):.0f} chars")
        print(f"    Min:         {min(lengths)} chars")
        print(f"    Max:         {max(lengths)} chars")

    # ─── 3. TF-IDF Distinguishing Terms ───
    print("\n" + "=" * 90)
    print("3. TF-IDF DISTINGUISHING TERMS (class 13 vs class 15)")
    print("=" * 90)

    all_texts = texts_13 + texts_15
    labels = [13] * len(texts_13) + [15] * len(texts_15)

    tfidf = TfidfVectorizer(
        max_features=5000, stop_words="english",
        ngram_range=(1, 2), min_df=2, max_df=0.95
    )
    X = tfidf.fit_transform(all_texts)
    feature_names = tfidf.get_feature_names_out()

    # Mean TF-IDF per class
    mask_13 = np.array(labels) == 13
    mask_15 = np.array(labels) == 15

    mean_13 = np.asarray(X[mask_13].mean(axis=0)).flatten()
    mean_15 = np.asarray(X[mask_15].mean(axis=0)).flatten()

    # Terms most distinctive for class 13 (higher in 13 than 15)
    diff_13 = mean_13 - mean_15
    top_13_idx = diff_13.argsort()[-20:][::-1]

    print("\n  Top 20 terms MORE associated with class 13 (remove_persistence):")
    print(f"  {'Term':<30} {'Mean TF-IDF 13':>15} {'Mean TF-IDF 15':>15} {'Diff':>10}")
    print("  " + "-" * 72)
    for idx in top_13_idx:
        print(f"  {feature_names[idx]:<30} {mean_13[idx]:>15.4f} {mean_15[idx]:>15.4f} {diff_13[idx]:>+10.4f}")

    # Terms most distinctive for class 15 (higher in 15 than 13)
    diff_15 = mean_15 - mean_13
    top_15_idx = diff_15.argsort()[-20:][::-1]

    print(f"\n  Top 20 terms MORE associated with class 15 (restore_defense_config):")
    print(f"  {'Term':<30} {'Mean TF-IDF 15':>15} {'Mean TF-IDF 13':>15} {'Diff':>10}")
    print("  " + "-" * 72)
    for idx in top_15_idx:
        print(f"  {feature_names[idx]:<30} {mean_15[idx]:>15.4f} {mean_13[idx]:>15.4f} {diff_15[idx]:>+10.4f}")

    # ─── 4. Cosine Similarity Between Class Centroids ───
    print("\n" + "=" * 90)
    print("4. COSINE SIMILARITY BETWEEN CLASS CENTROIDS")
    print("=" * 90)
    from sklearn.metrics.pairwise import cosine_similarity
    cos_sim = cosine_similarity(mean_13.reshape(1, -1), mean_15.reshape(1, -1))[0, 0]
    print(f"\n  Cosine similarity (TF-IDF centroid): {cos_sim:.4f}")
    print(f"  Interpretation: {'HIGH overlap — hard to separate' if cos_sim > 0.5 else 'Moderate overlap' if cos_sim > 0.3 else 'Low overlap — should be separable'}")

    # ─── 5. Representative Samples ───
    print("\n" + "=" * 90)
    print("5. REPRESENTATIVE SAMPLES")
    print("=" * 90)
    show_samples(texts_13, 13, n=20)
    show_samples(texts_15, 15, n=20)

    # ─── 6. Cross-class keyword overlap summary ───
    print("\n" + "=" * 90)
    print("6. CROSS-CLASS OVERLAP SUMMARY")
    print("=" * 90)
    persistence_in_13 = sum(1 for t in texts_13 if "persist" in t.lower())
    persistence_in_15 = sum(1 for t in texts_15 if "persist" in t.lower())
    defense_in_13 = sum(1 for t in texts_13 if any(k in t.lower() for k in ["defense", "defender", "antivirus", "security tool"]))
    defense_in_15 = sum(1 for t in texts_15 if any(k in t.lower() for k in ["defense", "defender", "antivirus", "security tool"]))

    print(f"\n  'persist*' appears in class 13: {persistence_in_13}/{len(texts_13)} ({persistence_in_13/len(texts_13)*100:.1f}%)")
    print(f"  'persist*' appears in class 15: {persistence_in_15}/{len(texts_15)} ({persistence_in_15/len(texts_15)*100:.1f}%)")
    print(f"  defense/AV keywords in class 13: {defense_in_13}/{len(texts_13)} ({defense_in_13/len(texts_13)*100:.1f}%)")
    print(f"  defense/AV keywords in class 15: {defense_in_15}/{len(texts_15)} ({defense_in_15/len(texts_15)*100:.1f}%)")


if __name__ == "__main__":
    main()
