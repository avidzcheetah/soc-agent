# Responsibility: Entry point only

import os
import sys
from src.config import parse_args, load_and_validate_config
from src.utils import set_seeds, log_environment, print_training_header
from src.experiment import init_experiment, finalize_experiment
from src.dataset import load_data_splits, verify_data_splits, get_dataloaders
from src.tokenizer import get_tokenizer, verify_tokenizer
from src.model import get_model
from src.trainer import SecBERTTrainer
from src.evaluator import evaluate_model_on_test_split, verify_saved_model

def main():
    # 1. Config & CLI
    args = parse_args()
    cfg = load_and_validate_config(args)

    # 2. Setup & Environment
    set_seeds(cfg.system.seed)
    log_environment(cfg)

    # 3. Experiment Manager
    exp_paths = init_experiment(cfg)

    # 4. Dataset Loading & Verification
    train_df, val_df, test_df = load_data_splits(cfg)
    verify_data_splits(train_df, val_df, test_df, cfg)

    # 5. Tokenizer Pipeline
    tokenizer = get_tokenizer(cfg)
    verify_tokenizer(tokenizer, train_df["text"].values, cfg)

    # 6. Model Construction
    model = get_model(cfg)

    # 7. Trainer Assembly
    train_loader, val_loader, test_loader = get_dataloaders(train_df, val_df, test_df, tokenizer, cfg)
    trainer = SecBERTTrainer(model, cfg, exp_paths, train_loader, val_loader, tokenizer)

    print_training_header(cfg, len(train_df), len(val_df), len(test_df), exp_id=exp_paths["exp_id"])

    # 8. Train
    trainer.fit()

    # 9. Final Evaluation
    test_metrics = evaluate_model_on_test_split(exp_paths, test_loader, cfg)

    # 10. Model Verification & Freeze Documenting
    best_model_dir = os.path.join(exp_paths["checkpoints"], "best_model")
    if os.path.exists(best_model_dir):
        verify_saved_model(best_model_dir)

    # 11. Archive Experiment
    if test_metrics:
        best_epoch = getattr(trainer, 'best_epoch', None)
        finalize_experiment(exp_paths, cfg, test_metrics, best_epoch=best_epoch)

    print("\nPhase 2 Complete: Model is completely finalized and verified for the PPO environment.")
    sys.exit(0)

if __name__ == "__main__":
    main()
