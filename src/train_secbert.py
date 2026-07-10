# Responsibility: Entry point only

import sys
from src.config import parse_args, load_and_validate_config
from src.utils import set_seeds, log_environment
from src.experiment import init_experiment
from src.dataset import load_data_splits, verify_data_splits
from src.tokenizer import get_tokenizer, verify_tokenizer
from src.model import get_model

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

    print("\nPhase 2 partial verification completed successfully!")
    sys.exit(0)

if __name__ == "__main__":
    main()
