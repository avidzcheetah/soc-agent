import os
import sys
import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.secbert.dataset import SecBERTDataset
from src.secbert.metrics import generate_classification_report, compute_metrics, plot_confusion_matrices

def main():
    model_dir = "models/secbert_finetuned"
    test_csv = "data/processed/test.csv"
    
    if not os.path.exists(model_dir):
        print(f"Error: Model directory not found at {model_dir}")
        sys.exit(1)
        
    if not os.path.exists(test_csv):
        print(f"Error: Test dataset not found at {test_csv}")
        sys.exit(1)

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    print(f"Loading test data from {test_csv}...")
    test_df = pd.read_csv(test_csv)
    
    # Needs to match the preprocessing used in Phase 1
    action_space_df = pd.read_csv("data/action_space.csv")
    phase_map = dict(zip(action_space_df['Index'], action_space_df['CISA Phase']))
    
    def prepend_phase(df):
        df['text'] = df.apply(lambda row: f"[{phase_map[row['action_label']]}] {row['text']}", axis=1)
        return df

    test_df = prepend_phase(test_df)
    
    test_texts = test_df['text'].tolist()
    test_labels = test_df['action_label'].tolist()
    
    test_dataset = SecBERTDataset(test_texts, test_labels, tokenizer, max_length=512)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collator)

    all_preds = []
    all_labels = []
    all_logits = []

    print("Evaluating test set...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(outputs.logits.cpu().float().numpy())

    print("Computing metrics...")
    test_metrics = compute_metrics(all_labels, all_preds, logits=all_logits)
    
    print("\nTest Metrics:")
    print(f"Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"Macro F1:    {test_metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {test_metrics['weighted_f1']:.4f}")
    print("\nEvaluation complete. All metrics printed above.")

if __name__ == "__main__":
    main()
