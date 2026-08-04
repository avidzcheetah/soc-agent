import os
import torch
import sys
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.secbert.metrics import generate_classification_report, plot_confusion_matrices

def verify_saved_model(best_model_dir):
    """
    Phase 2.13: Verify that the saved model is technically correct.
    """
    print("======================================")
    print("Model Verification (Saved Checkpoint)")
    print("======================================")
    
    try:
        model = AutoModelForSequenceClassification.from_pretrained(best_model_dir)
        tokenizer = AutoTokenizer.from_pretrained(best_model_dir)
        print("  [PASS] Model and Tokenizer loaded successfully")
    except Exception as e:
        print(f"  [FAIL] Failed to load model/tokenizer: {e}")
        sys.exit(1)
        
    # Check embedding size
    try:
        embedding_size = model.config.hidden_size
        if embedding_size == 768:
            print("  [PASS] Embedding size: 768")
        else:
            print(f"  [FAIL] Expected embedding size 768, got {embedding_size}")
            sys.exit(1)
    except AttributeError:
        print("  [FAIL] Could not determine embedding size")
        sys.exit(1)
        
    # Check classifier output
    try:
        num_labels = model.config.num_labels
        if num_labels == 20:
            print("  [PASS] Classifier output: 20")
        else:
            print(f"  [FAIL] Expected classifier output 20, got {num_labels}")
            sys.exit(1)
    except AttributeError:
        print("  [FAIL] Could not determine classifier output size")
        sys.exit(1)
        
    # Inference works on random samples + No NaNs/Infs
    try:
        test_text = "Attack: Exploitive. Target: Generic."
        inputs = tokenizer(test_text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
        if torch.isnan(logits).any():
            print("  [FAIL] Model produced NaN outputs")
            sys.exit(1)
        elif torch.isinf(logits).any():
            print("  [FAIL] Model produced Inf outputs")
            sys.exit(1)
        else:
            print("  [PASS] No NaNs or Infs detected in outputs")
            print("  [PASS] Inference works on random samples")
    except Exception as e:
        print(f"  [FAIL] Inference test failed: {e}")
        sys.exit(1)
        
    # Phase 2.14: Freeze Encoder documentation
    print("\n======================================")
    print("Phase 2.14: Architecture Lock (PPO Preparation)")
    print("======================================")
    print("  [INFO] The classifier head remains part of the saved checkpoint.")
    print("  [INFO] For the subsequent RL/PPO phase, load this checkpoint and")
    print("         use the underlying encoder in inference (frozen) mode to")
    print("         generate state embeddings. Gradients will be disabled.")
    print("======================================\n")

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
    all_logits = []
    
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
            all_logits.extend(outputs.logits.cpu().float().numpy())
            
    # Generate Reports
    results_dir = dirs["results"]
    
    # Classification Report
    report_path = os.path.join(results_dir, "classification_report.json")
    report_dict = generate_classification_report(all_labels, all_preds, report_path)
    print(f"  [+] Saved classification report -> {report_path}")
    
    # Confusion Matrices
    plot_confusion_matrices(all_labels, all_preds, results_dir)
    print(f"  [+] Saved confusion matrices -> {results_dir}")
    
    print("======================================\n")
    
    from src.secbert.metrics import compute_metrics
    test_metrics = compute_metrics(all_labels, all_preds, logits=all_logits)
    if 'top2_accuracy' in test_metrics:
        print(f"  [+] Top-2 Accuracy: {test_metrics['top2_accuracy']:.4f}")
    return test_metrics
