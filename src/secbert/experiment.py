import os
import json
import datetime
import subprocess
import torch

def get_git_revision_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown"

def generate_experiment_id(base_dir="experiments"):
    os.makedirs(base_dir, exist_ok=True)
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    prefix = f"EXP_{date_str}_"
    
    existing = [d for d in os.listdir(base_dir) if d.startswith(prefix) and os.path.isdir(os.path.join(base_dir, d))]
    
    indices = []
    for d in existing:
        try:
            parts = d.split("_")
            indices.append(int(parts[2]))
        except (IndexError, ValueError):
            pass
            
    next_idx = max(indices) + 1 if indices else 1
    return f"{prefix}{next_idx:03d}"

def init_experiment(cfg, base_dir="experiments"):
    """
    Initialize a new experiment directory and save experiment.json metadata before training starts.
    """
    # 1. Generate Experiment ID
    exp_id = generate_experiment_id(base_dir)
    exp_dir = os.path.join(base_dir, exp_id)
    
    # 2. Create Directory Layout
    subdirs = ["metadata", "logs", "results", "checkpoints"]
    paths = {}
    for sub in subdirs:
        path = os.path.join(exp_dir, sub)
        os.makedirs(path, exist_ok=True)
        paths[sub] = path
        
    paths["exp_dir"] = exp_dir
    paths["exp_id"] = exp_id
    
    # 3. Gather System Details
    git_commit = get_git_revision_hash()
    device = torch.device(cfg.system.device)
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" and torch.cuda.is_available() else "None"
    
    # 4. Construct metadata
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metadata = {
        "experiment_id": exp_id,
        "timestamp": timestamp,
        "git_commit": git_commit,
        "gpu": gpu_name,
        "seed": cfg.system.seed,
        "dataset_paths": {
            "train": cfg.dataset.train_path,
            "val": cfg.dataset.val_path,
            "test": cfg.dataset.test_path
        },
        "config": cfg.to_dict()
    }
    
    # 5. Save experiment.json inside metadata/ directory
    meta_file_path = os.path.join(paths["metadata"], "experiment.json")
    with open(meta_file_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Initialized Experiment: {exp_id}")
    print(f"Saved initial experiment.json to {meta_file_path}")
    
    return paths

def finalize_experiment(dirs, cfg, best_metrics, best_epoch=None, base_dir="experiments"):
    """
    Phase 2.15: Archive Experiment metadata and append to summary.csv.
    """
    import csv
    import hashlib
    
    def _file_hash(path):
        """Compute MD5 hash of a file for reproducibility tracking."""
        h = hashlib.md5()
        try:
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    h.update(chunk)
            return h.hexdigest()
        except FileNotFoundError:
            return "file_not_found"
    
    # 1. Update experiment.json with best_metrics and dataset hashes
    meta_file_path = os.path.join(dirs["metadata"], "experiment.json")
    if os.path.exists(meta_file_path):
        with open(meta_file_path, "r") as f:
            metadata = json.load(f)
            
        metadata["best_metrics"] = best_metrics
        metadata["best_epoch"] = best_epoch
        metadata["dataset_hash"] = {
            "train": _file_hash(cfg.dataset.train_path),
            "val": _file_hash(cfg.dataset.val_path),
            "test": _file_hash(cfg.dataset.test_path)
        }
        
        with open(meta_file_path, "w") as f:
            json.dump(metadata, f, indent=4)
            
    # 2. Append to summary.csv
    summary_file = os.path.join(base_dir, "summary.csv")
    file_exists = os.path.exists(summary_file)
    
    with open(summary_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Experiment", "LR", "Batch", "Epochs", "Macro F1", "Weighted F1", "MCC", "Best Epoch"])
            
        writer.writerow([
            dirs["exp_id"],
            cfg.optimizer.learning_rate,
            cfg.training.batch_size,
            cfg.training.epochs,
            f"{best_metrics.get('macro_f1', 0):.4f}",
            f"{best_metrics.get('weighted_f1', 0):.4f}",
            f"{best_metrics.get('mcc', 0):.4f}",
            best_epoch if best_epoch else "N/A"
        ])
    print(f"  [+] Appended results to {summary_file}")
