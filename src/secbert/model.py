# Responsibility: Build SecBERT model
import sys
import torch
from transformers import AutoModelForSequenceClassification


def get_model(cfg):
    """
    Load jackaduma/SecBERT with a 20-class classification head.
    Immediately verify output dimension and print parameter counts.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.name,
        num_labels=cfg.model.num_labels
    )

    print("======================================")
    print("Model Construction")
    print("======================================")
    print(f"  Base model       : {cfg.model.name}")
    print(f"  Num labels       : {cfg.model.num_labels}")

    # ── Output dimension check ────────────────
    classifier = model.classifier if hasattr(model, "classifier") else None
    if classifier is None and hasattr(model, "out_proj"):
        classifier = model.out_proj

    if classifier is not None:
        out_features = classifier.out_features
        status = "PASS" if out_features == cfg.model.num_labels else "FAIL"
        print(f"  [{status}] Output dimension  : {out_features}")
        if out_features != cfg.model.num_labels:
            print(f"\nExpected {cfg.model.num_labels} output logits, got {out_features}. Stopping.\n")
            sys.exit(1)
    else:
        print(f"  [FAIL] Could not locate classification head.")
        sys.exit(1)

    # ── Parameter counts ──────────────────────
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params    = total_params - trainable_params

    print(f"  Total parameters      : {total_params:>12,}")
    print(f"  Trainable parameters  : {trainable_params:>12,}")
    print(f"  Frozen parameters     : {frozen_params:>12,}")
    print("======================================\n")

    return model


