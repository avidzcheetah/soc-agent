# Responsibility: Test evaluation

import os
import torch
from transformers import AutoModelForSequenceClassification
from src.metrics import generate_classification_report, plot_confusion_matrices

def evaluate_model_on_test_split(dirs, test_loader, cfg):
    """
    Run the best model checkpoints on the hold-out test set to produce final evaluation scores.
    """
    print("======================================")
    print("Final Evaluation on Test Split")
    print("======================================")
    
    best_model_dir = os.path.join(dirs["checkpoints"], "best_model")
    if not os.path.exists(best_model_dir):
        print(f"  [FAIL] best_model directory not found at {best_model_dir}")
        return
        
    device = torch.device(cfg.system.device)
    model = AutoModelForSequenceClassification.from_pretrained(best_model_dir)
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("  Evaluating test set...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Generate Reports
    results_dir = dirs["results"]
    
    # Classification Report
    report_path = os.path.join(results_dir, "classification_report.json")
    generate_classification_report(all_labels, all_preds, report_path)
    print(f"  [+] Saved classification report -> {report_path}")
    
    # Confusion Matrices
    plot_confusion_matrices(all_labels, all_preds, results_dir)
    print(f"  [+] Saved confusion matrices -> {results_dir}")
    
    print("======================================\n")
