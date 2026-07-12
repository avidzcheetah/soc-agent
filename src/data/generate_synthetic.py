import os
import json
import time
import argparse
import pandas as pd
from google import genai
from google.genai import types

def generate_synthetic_data(api_key, output_raw, output_csv, train_csv):
    """
    Generate synthetic SOC logs for the weakest classes using Gemini API.
    Saves raw generation details (prompts/responses) and outputs an augmented train.csv.
    """
    client = genai.Client(api_key=api_key)
    model_name = 'gemini-2.5-flash'
    
    # Target 4 weakest classes:
    targets = {
        6: {
            "name": "block_port",
            "context": "Unauthorized Service / Exploit on Specific Port. E.g., RDP brute force, SMB exploits (EternalBlue), unauthorized SSH access, database port scanning."
        },
        14: {
            "name": "restore_registry",
            "context": "Registry Tampering / Execution Flow Hijack. E.g., modified Run/RunOnce keys, Image File Execution Options (IFEO) injection, disabled UAC via registry."
        },
        15: {
            "name": "restore_defense_config",
            "context": "Defender/AV Disabled / Logging Tampered. E.g., Windows Defender disabled via PowerShell, EDR service stopped, Event Log service cleared or stopped (T1562.001)."
        },
        17: {
            "name": "snapshot_forensics",
            "context": "High-Severity Incident Requiring Investigation. E.g., active ransomware encryption detected, advanced memory-resident malware, suspicious lateral movement requiring deep memory analysis."
        }
    }
    
    num_samples_per_class = 50
    
    prompt_template = """
You are a cybersecurity expert. Generate exactly {num_samples} highly realistic SOC (Security Operations Center) log entries or alerts that would require the incident response action: '{action_name}'. 

Context: {context}

The logs must vary widely in structure, format, and source (e.g., Windows Event Logs, Syslog, EDR alerts, Firewall logs, Zeek/Suricata IDS alerts). They must be highly technical and realistic, incorporating random but plausible IP addresses, file paths, hashes, registry keys, or specific CVEs where appropriate. DO NOT number them. Output strictly a valid JSON array of strings, where each string is a single log entry. Example:
[
  "WinEventLog: Security: 4688: A new process has been created. C:\\\\Windows\\\\System32\\\\cmd.exe...",
  "Alert: High severity - Suspicious network connection detected..."
]
"""

    results = []
    metadata = {}
    
    print(f"Starting synthetic generation for {len(targets)} classes ({num_samples_per_class} samples each)...")
    
    for label_idx, info in targets.items():
        print(f"Generating for Class {label_idx} ({info['name']})...")
        prompt = prompt_template.format(
            num_samples=num_samples_per_class,
            action_name=info["name"],
            context=info["context"]
        )
        
        metadata[info["name"]] = {"prompt": prompt}
        
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.7
                )
            )

            
            # Parse the JSON response
            logs = json.loads(response.text)
            
            if not isinstance(logs, list):
                print(f"Warning: Expected a JSON array for {info['name']}, but got something else. Skipping.")
                continue
                
            metadata[info["name"]]["success"] = True
            metadata[info["name"]]["count_generated"] = len(logs)
            
            for log in logs:
                results.append({
                    "text": str(log),
                    "action_label": label_idx
                })
                
            print(f"  [+] Successfully generated {len(logs)} samples.")
        
        except Exception as e:
            print(f"  [-] Failed to generate for {info['name']}: {e}")
            metadata[info["name"]]["success"] = False
            metadata[info["name"]]["error"] = str(e)
            
        # Polite sleep to avoid rate limits
        time.sleep(2)
        
    # Save the raw generation metadata (for reproducibility)
    os.makedirs(os.path.dirname(output_raw), exist_ok=True)
    with open(output_raw, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"\nSaved generation metadata to {output_raw}")
    
    # Save the new samples alone
    new_df = pd.DataFrame(results)
    print(f"Total synthetic samples generated: {len(new_df)}")
    
    # Append to existing train.csv and save as train_augmented.csv
    print(f"Loading original training data from {train_csv}...")
    orig_df = pd.read_csv(train_csv)
    
    combined_df = pd.concat([orig_df, new_df], ignore_index=True)
    # Shuffle the dataset
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    combined_df.to_csv(output_csv, index=False)
    print(f"Saved augmented training dataset ({len(combined_df)} records) to {output_csv}")
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic SOC logs for rare classes.")
    parser.add_argument("--api-key", type=str, required=True, help="Gemini API Key")
    parser.add_argument("--output-raw", type=str, default="data/raw/synthetic_generation_metadata.json", help="Path to save metadata/prompts")
    parser.add_argument("--output-csv", type=str, default="data/processed/train_augmented.csv", help="Path to save augmented dataset")
    parser.add_argument("--train-csv", type=str, default="data/processed/train.csv", help="Original training dataset to augment")
    
    args = parser.parse_args()
    
    generate_synthetic_data(args.api_key, args.output_raw, args.output_csv, args.train_csv)
