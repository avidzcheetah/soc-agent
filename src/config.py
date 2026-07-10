import os
import argparse
import yaml
import torch

class ConfigNode:
    """
    Nested configuration object allowing attribute (dot-notation) access.
    """
    def __init__(self, data):
        for k, v in data.items():
            if isinstance(v, dict):
                setattr(self, k, ConfigNode(v))
            else:
                setattr(self, k, v)

    def to_dict(self):
        result = {}
        for k, v in self.__dict__.items():
            if isinstance(v, ConfigNode):
                result[k] = v.to_dict()
            else:
                result[k] = v
        return result

    def __repr__(self):
        return f"ConfigNode({self.__dict__})"


def parse_args():
    parser = argparse.ArgumentParser(description="Config-driven SecBERT training pipeline.")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml configuration file.")
    parser.add_argument("--debug", action="store_true", help="Launch in debug mode (200 records, 1 epoch, CPU).")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest epoch checkpoint.")
    parser.add_argument("--seed", type=int, default=None, help="Override configuration random seed.")
    parser.add_argument("--device", type=str, default=None, help="Override target device (cpu/cuda).")
    parser.add_argument("--output", type=str, default=None, help="Override default experiment directory.")
    parser.add_argument("--dry-run", action="store_true", help="Verify configurations and datasets then exit.")
    return parser.parse_args()


def load_and_validate_config(args):
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")
        
    with open(args.config, 'r') as f:
        data = yaml.safe_load(f)
        
    # Validate required top-level sections
    required_sections = [
        'experiment', 'model', 'dataset', 'tokenizer', 'training',
        'optimizer', 'scheduler', 'early_stopping', 'checkpoint',
        'evaluation', 'logging', 'system'
    ]
    for section in required_sections:
        if section not in data:
            raise KeyError(f"Missing required configuration section: '{section}'")
            
    # Apply CLI overrides
    if args.debug:
        data['system']['debug'] = True
        data['training']['epochs'] = 1
        data['system']['device'] = 'cpu'
        
    if args.resume:
        data['checkpoint']['resume'] = True
        
    if args.seed is not None:
        data['system']['seed'] = args.seed
        
    if args.device is not None:
        data['system']['device'] = args.device
        
    if args.output is not None:
        # Override output directories dynamically
        data['checkpoint']['save_dir'] = os.path.join(args.output, "checkpoints")
        data['checkpoint']['best_dir'] = os.path.join(args.output, "best_model")
        data['logging']['log_dir'] = os.path.join(args.output, "logs")
        data['experiment']['output_dir'] = args.output
    else:
        data['experiment']['output_dir'] = None
        
    if args.dry_run:
        data['system']['dry_run'] = True
        
    return ConfigNode(data)
