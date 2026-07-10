# Responsibility: Shared helper functions
import sys
import random
import subprocess
import numpy as np
import torch

def get_git_revision_hash():
    """
    Retrieve current git commit hash.
    """
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown"

def set_seeds(seed):
    """
    Set reproducibility seeds across PyTorch, NumPy, and random packages.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def log_environment(cfg):
    """
    Print environment parameters (python, torch, transformers, cuda and gpu).
    If CUDA is unavailable, stop immediately unless --device cpu was explicitly requested.
    """
    print("======================================")
    print("Environment Verification")
    print("======================================")
    
    print(f"Python       : {sys.version.split(' ')[0]}")
    print(f"Torch        : {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA         : {'Available' if cuda_available else 'Unavailable'}")
    
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = round(vram_bytes / (1024 ** 3), 2)
        print(f"GPU          : {gpu_name}")
        print(f"VRAM         : {vram_gb} GB")
    else:
        print(f"GPU          : None")
        print(f"VRAM         : 0 GB")
        
    try:
        import transformers
        print(f"Transformers : {transformers.__version__}")
    except ImportError:
        print(f"Transformers : Not Installed")
        
    try:
        import datasets
        print(f"Datasets     : {datasets.__version__}")
    except ImportError:
        print(f"Datasets     : Not Installed")
        
    try:
        import accelerate
        print(f"Accelerate   : {accelerate.__version__}")
    except ImportError:
        print(f"Accelerate   : Not Installed")
        
    print(f"Device       : {cfg.system.device}")
    print("======================================")
    
    if cfg.system.device == "cuda" and not cuda_available:
        print("\nERROR: CUDA is unavailable but device 'cuda' was requested.")
        print("Stopping immediately. Use --device cpu to run on CPU explicitly.\n")
        sys.exit(1)

def print_training_header(cfg, train_size, val_size, test_size, exp_id="N/A"):
    """
    Format and output the experiment runner details box.
    """
    print("======================================")
    print("Training Configuration")
    print("======================================")
    print(f"  Experiment ID         : {exp_id}")
    print(f"  Train / Val / Test    : {train_size} / {val_size} / {test_size} records")
    print(f"  Batch Size            : {cfg.training.batch_size}")
    print(f"  Gradient Accumulation : {cfg.training.gradient_accumulation_steps} steps")
    print(f"  Epochs                : {cfg.training.epochs}")
    print(f"  Mixed Precision       : {cfg.training.mixed_precision}")
    print()
    print("Optimizer Configuration:")
    print(f"  Optimizer             : {cfg.optimizer.type}")
    print(f"  Learning Rate         : {cfg.optimizer.learning_rate}")
    print(f"  Weight Decay          : {cfg.optimizer.weight_decay}")
    print(f"  Gradient Clipping     : {cfg.optimizer.max_grad_norm}")
    print()
    print("Scheduler Configuration:")
    print(f"  Scheduler             : {cfg.scheduler.type}")
    print(f"  Warmup Ratio          : {cfg.scheduler.warmup_ratio}")
    print()
    print("Early Stopping Configuration:")
    print(f"  Monitor               : {cfg.checkpoint.monitor}")
    print(f"  Mode                  : {cfg.checkpoint.mode}")
    print(f"  Patience              : {cfg.early_stopping.patience} epochs")
    print(f"  Min Delta             : {cfg.early_stopping.min_delta}")
    print("======================================\n")
