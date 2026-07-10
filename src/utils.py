# Responsibility: Shared helper functions

def get_git_revision_hash():
    """
    Retrieve current git commit hash.
    """
    pass

def set_seeds(seed):
    """
    Set reproducibility seeds across PyTorch, NumPy, and random packages.
    """
    pass

def log_environment(config):
    """
    Print environment parameters (python, torch, transformers, cuda and gpu).
    """
    pass

def print_training_header(config, train_size, val_size, test_size):
    """
    Format and output the experiment runner details box.
    """
    pass
