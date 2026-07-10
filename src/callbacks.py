# Responsibility: Early stopping, checkpointing

class EarlyStopping:
    """
    Early stopping monitor to track validation loss trends.
    """
    def __init__(self, patience=2, min_delta=0.0):
        pass

    def __call__(self, val_loss):
        pass

def save_epoch_checkpoint(epoch, model, tokenizer, optimizer, scheduler, checkpoints_dir):
    """
    Save PyTorch weights and training state (optimizer, scheduler) for resume.
    """
    pass
