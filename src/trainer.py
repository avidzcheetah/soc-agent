# Responsibility: Training pipeline
import os
import time
import json
import torch
import numpy as np
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter

from src.callbacks import EarlyStopping, save_epoch_checkpoint
from src.metrics import compute_metrics

class SecBERTTrainer:
    """
    Core executor that manages optimization steps, backpropagation, validation checks,
    learning rate scheduling, gradient clipping, early stopping, and logs metrics.
    """
    def __init__(self, model, cfg, dirs, train_loader, val_loader, tokenizer):
        self.model = model
        self.cfg = cfg
        self.dirs = dirs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        
        self.device = torch.device(cfg.system.device)
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(), 
            lr=cfg.optimizer.learning_rate, 
            weight_decay=cfg.optimizer.weight_decay
        )
        
        # Scheduler
        total_steps = len(self.train_loader) * cfg.training.epochs // cfg.training.gradient_accumulation_steps
        warmup_steps = int(total_steps * cfg.scheduler.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps
        )
        
        # Callbacks & Settings
        self.early_stopping = EarlyStopping(
            patience=cfg.early_stopping.patience, 
            min_delta=cfg.early_stopping.min_delta
        )
        
        self.scaler = torch.amp.GradScaler('cuda') if (cfg.training.mixed_precision and self.device.type == "cuda") else None
        
        # Tensorboard & Tracking
        self.writer = SummaryWriter(log_dir=self.dirs["logs"])
        self.history = []
        
        print("======================================")
        print("Trainer Assembly Verification")
        print("======================================")
        print("  [PASS] Model loaded & moved to device")
        print("  [PASS] DataLoaders & Tokenizer attached")
        print("  [PASS] Optimizer (AdamW) initialized")
        print("  [PASS] Scheduler (Linear Warmup) initialized")
        print("  [PASS] Callbacks attached")
        print("======================================\n")

    def _get_gpu_memory(self):
        if self.device.type == "cuda":
            return torch.cuda.memory_allocated() / (1024 ** 3)
        return 0.0

    def train_epoch(self, epoch):
        """
        Execute one epoch of training batches.
        """
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(self.train_loader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward
            if self.scaler:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss / self.cfg.training.gradient_accumulation_steps
            else:
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / self.cfg.training.gradient_accumulation_steps
                
            # Backward
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
                
            total_loss += loss.item() * self.cfg.training.gradient_accumulation_steps
            
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Step
            if (step + 1) % self.cfg.training.gradient_accumulation_steps == 0 or (step + 1) == len(self.train_loader):
                # Gradient Norm
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optimizer.max_grad_norm)
                
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                    
                self.scheduler.step()
                self.optimizer.zero_grad()
                
        metrics = compute_metrics(all_labels, all_preds)
        metrics["loss"] = total_loss / len(self.train_loader)
        metrics["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        return metrics

    def validate_epoch(self):
        """
        Execute validation pass and compute metrics.
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                if self.scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                else:
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    
                total_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        metrics = compute_metrics(all_labels, all_preds)
        metrics["loss"] = total_loss / len(self.val_loader)
        return metrics

    def fit(self):
        """
        Main run loop with checkpointing, selection, and early stopping checks.
        """
        print("Starting training loop...")
        best_val_loss = float('inf')
        
        for epoch in range(1, self.cfg.training.epochs + 1):
            start_time = time.time()
            
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate_epoch()
            
            epoch_time = time.time() - start_time
            current_lr = self.scheduler.get_last_lr()[0]
            gpu_mem = self._get_gpu_memory()
            
            # Print metrics
            print(f"Epoch {epoch}/{self.cfg.training.epochs} | Time: {epoch_time:.1f}s | LR: {current_lr:.2e} | GPU Mem: {gpu_mem:.2f}GB")
            print(f"  Train -> Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.4f} | Macro F1: {train_metrics['macro_f1']:.4f} | Grad Norm: {train_metrics['grad_norm']:.4f}")
            print(f"  Val   -> Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | Macro F1: {val_metrics['macro_f1']:.4f}")
            
            # Write to Tensorboard
            self.writer.add_scalar("Loss/Train", train_metrics['loss'], epoch)
            self.writer.add_scalar("Loss/Val", val_metrics['loss'], epoch)
            self.writer.add_scalar("Accuracy/Train", train_metrics['accuracy'], epoch)
            self.writer.add_scalar("Accuracy/Val", val_metrics['accuracy'], epoch)
            self.writer.add_scalar("Macro_F1/Train", train_metrics['macro_f1'], epoch)
            self.writer.add_scalar("Macro_F1/Val", val_metrics['macro_f1'], epoch)
            self.writer.add_scalar("Weighted_F1/Val", val_metrics['weighted_f1'], epoch)
            self.writer.add_scalar("Precision/Val", val_metrics['precision'], epoch)
            self.writer.add_scalar("Recall/Val", val_metrics['recall'], epoch)
            self.writer.add_scalar("Learning_Rate", current_lr, epoch)
            self.writer.add_scalar("Epoch_Time", epoch_time, epoch)
            self.writer.add_scalar("Gradient_Norm", train_metrics['grad_norm'], epoch)
            self.writer.add_scalar("GPU_Memory_GB", gpu_mem, epoch)
            
            # Save history
            epoch_record = {
                "epoch": epoch,
                "time_sec": epoch_time,
                "learning_rate": current_lr,
                "gpu_mem_gb": gpu_mem,
                "train": train_metrics,
                "val": val_metrics
            }
            self.history.append(epoch_record)
            
            # Save JSON
            with open(os.path.join(self.dirs["metadata"], "metrics.json"), "w") as f:
                json.dump(self.history, f, indent=4)
                
            # Checkpoint
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                print(f"  [+] Validation loss improved to {best_val_loss:.4f}. Saving best model.")
                # Save to best_model dir
                best_model_dir = os.path.join(self.cfg.checkpoint.best_dir)
                os.makedirs(best_model_dir, exist_ok=True)
                self.model.save_pretrained(best_model_dir)
                self.tokenizer.save_pretrained(best_model_dir)
                
                # Also save standard checkpoint
                save_epoch_checkpoint(epoch, self.model, self.tokenizer, self.optimizer, self.scheduler, self.dirs["checkpoints"])
                
            # Early Stopping
            self.early_stopping(val_metrics['loss'])
            if self.early_stopping.early_stop:
                print(f"  [!] Early stopping triggered at epoch {epoch}.")
                break
                
            print("-" * 60)
            
        self.writer.close()
        print("Training complete.")
