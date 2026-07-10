# Responsibility: Early stopping, checkpointing
import os
import torch
import numpy as np

class EarlyStopping:
    """
    Early stopping monitor to track validation loss trends.
    """
    def __init__(self, patience=2, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def save_epoch_checkpoint(epoch, model, tokenizer, optimizer, scheduler, checkpoints_dir):
    """
    Save PyTorch weights and training state (optimizer, scheduler) for resume.
    """
    checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint-epoch-{epoch}")
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Save model and tokenizer
    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)
    
    # Save training states
    torch.save({
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None
    }, os.path.join(checkpoint_path, "training_state.pt"))
    return checkpoint_path
