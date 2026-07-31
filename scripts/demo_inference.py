import os
import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_label_map(csv_path="data/action_space.csv"):
    df = pd.read_csv(csv_path)
    label_map = dict(zip(df['Index'], df['Action Name']))
    return label_map

def predict(text, model, tokenizer, label_map, device="cpu"):
    # Tokenize input
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512
    ).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]
        
    # Get top 3 predictions
    top_probs, top_indices = torch.topk(probs, 3)
    
    print(f"Input Text: {text}")
    print("-" * 50)
    print("Top 3 Predictions:")
    for i in range(3):
        prob = top_probs[i].item()
        label_idx = top_indices[i].item()
        label_name = label_map.get(label_idx, f"Unknown (ID: {label_idx})")
        print(f"{i+1}. {label_name} ({prob:.2%})")
    print("=" * 50)
    print()

def main():
    model_path = "experiments/EXP_20260713_001/checkpoints/best_model"
    action_space_path = "data/action_space.csv"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    print("Loading model and tokenizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()
    
    label_map = load_label_map(action_space_path)
    print(f"Model loaded successfully on {device}!\n")
    
    # Demonstration samples
    samples = [
        "Multiple failed login attempts detected from IP 192.168.1.55 targeting the admin account via SSH.",
        "An unauthorized process 'mimikatz.exe' was blocked from executing on workstation-04 by the EDR.",
        "Malicious traffic observed reaching out to known C2 domain malware-tracker.com.",
        "A user reported receiving a suspicious email with an attachment 'invoice.pdf.exe'."
    ]
    
    for sample in samples:
        predict(sample, model, tokenizer, label_map, device)
        
    # Interactive loop
    print("Entering interactive mode. Type 'quit' or 'exit' to stop.")
    while True:
        try:
            user_input = input("Enter an alert or log text: ")
        except EOFError:
            break
            
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        if user_input.strip():
            predict(user_input, model, tokenizer, label_map, device)

if __name__ == "__main__":
    main()
