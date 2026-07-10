# Responsibility: Load CSVs and build datasets

class SecBERTDataset:
    """
    Custom PyTorch Dataset for SecBERT.
    """
    def __init__(self, texts, labels, tokenizer, max_length):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

def load_data_splits(debug=False, seed=42):
    """
    Load train, validation, and test CSV splits from data/processed/.
    """
    pass

def load_class_weights(weights_path="data/class_weights.json", device="cpu"):
    """
    Load action class weights array from JSON file.
    """
    pass

def get_dataloaders(train_df, val_df, test_df, tokenizer, max_length, batch_size, weights_path="data/class_weights.json"):
    """
    Initialize DataLoader objects, applying WeightedRandomSampler to the train split.
    """
    pass
