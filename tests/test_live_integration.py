# Responsibility: End-to-End Live Integration Test for SOCEnvironment + SecBERT Encoder
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import torch
from src.environment import SOCEnvironment
from src.encoder import SecBERTStateEncoder


def run_live_integration_test():
    print("=" * 60)
    print("RUNNING LIVE END-TO-END SOC ENVIRONMENT + SECBERT INTEGRATION")
    print("=" * 60)

    train_path = os.path.join(PROJECT_ROOT, "data", "processed", "train.csv")
    if not os.path.exists(train_path):
        print(f"[SKIP] train.csv not found at {train_path}")
        return

    # 1. Initialize real frozen encoder and real dataset
    print("\n[1/4] Initializing SecBERTStateEncoder & loading train.csv...")
    encoder = SecBERTStateEncoder()
    df = pd.read_csv(train_path).head(50)  # Test with 50 real alerts
    env = SOCEnvironment(df=df, encoder=encoder, mode="train")
    print("      Initialized environment successfully.")

    # 2. Reset
    print("\n[2/4] Executing env.reset()...")
    obs, info = env.reset(seed=42)
    assert isinstance(obs, torch.Tensor), "Obs must be a Tensor"
    assert obs.shape == (768,), f"Expected shape (768,), got {obs.shape}"
    assert obs.dtype == torch.float32, f"Expected float32, got {obs.dtype}"
    assert torch.isnan(obs).sum().item() == 0, "Obs contains NaNs!"
    assert torch.isinf(obs).sum().item() == 0, "Obs contains Infs!"
    print(f"      Initial state shape: {obs.shape}, norm: {torch.norm(obs):.4f}, NaNs: 0, Infs: 0")

    # 3. Step with ground truth
    print("\n[3/4] Testing ground-truth action step...")
    gt = int(env.current_sample["action_label"])
    next_obs, reward, term, trunc, step_info = env.step(action=gt)
    assert reward == 1.0, f"Expected +1.0 for GT action, got {reward}"
    assert step_info["correct"] is True
    print(f"      GT Action: {gt} -> Reward: {reward} [PASS]")

    # 4. Multi-step live run
    print("\n[4/4] Running 10 live steps with real SecBERT embeddings...")
    for i in range(10):
        action = i % 20
        next_obs, reward, term, trunc, step_info = env.step(action=action)
        assert next_obs.shape == (768,)
        assert torch.isnan(next_obs).sum().item() == 0
        print(f"      Step {i+1:02d}: Action={action:02d}, GT={step_info['ground_truth']:02d}, Reward={reward:+2.0f}, Status={'CORRECT' if step_info['correct'] else 'WRONG'}")

    print("\n" + "=" * 60)
    print("[PASS] LIVE END-TO-END INTEGRATION TEST PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_live_integration_test()
