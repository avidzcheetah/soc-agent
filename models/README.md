# Models Directory Structure

This directory stores trained models, weights, and evaluation checkpoints across the project phases.

---

## 🏆 Production / Final Model Checkpoints (Phase 1 & Phase 2)

All production-ready, final models are kept directly accessible in standard canonical locations:

### 1. Fine-Tuned SecBERT (Phase 1 Final Model)
- **Path**: [`models/secbert_finetuned/`](file:///models/secbert_finetuned)
- **Files**:
  - `model.safetensors`: Trained weights
  - `config.json`: SecBERT architecture configuration (20 classes)
  - `tokenizer.json` & `tokenizer_config.json`: Domain vocabulary and tokenizer
- **Role**:
  - Direct alert classification baseline (Acc: 95.13%, Macro F1: 0.7303)
  - Frozen state encoder for PPO RL agent (`src/encoder.py`)

### 2. Final PPO Agent (Phase 2 Final Model)
- **Directory**: [`models/ppo_final/`](file:///models/ppo_final)
- **Key Files**:
  - `best_ppo_policy.pt`: **Phase 2 Champion Policy (Experiment 4D ATGP)**
    - Test Set Accuracy: **94.74%**
    - Test Set Macro F1: **0.7164**
    - Test Set Weighted F1: **0.9461**
    - Test Set MCC: **+0.9378**
    - Features: Adaptive Teacher-Guided Preservation ($\tau=0.3, \lambda=1.0$), balanced incident sampling, entropy bonus $c_2=0.02$.
  - `warmstart_actor.pt`: Supervised Behavioral Cloning Warm-Start Actor (Val Macro F1: 0.7438)

---

## 🔬 Experimental Checkpoints & Ablations

| Directory | Experiment | Description |
|---|---|---|
| [`models/ppo_exp4d_atgp/`](file:///models/ppo_exp4d_atgp) | EXP_PPO_004D | ATGP Teacher-Preservation PPO (New Best Policy) |
| [`models/ppo_exp4c5_balanced_lr1e4_ent002_periodic/`](file:///models/ppo_exp4c5_balanced_lr1e4_ent002_periodic) | EXP_PPO_004C-5 | Periodic Checkpoint PPO without early stopping |
| [`models/ppo_exp4c4_balanced_lr1e4_ent002_earlystop/`](file:///models/ppo_exp4c4_balanced_lr1e4_ent002_earlystop) | EXP_PPO_004C-4 | Balanced sampling + entropy with patience=15 early stop |
| [`models/ppo_exp4c3_balanced_lr1e4_ent002/`](file:///models/ppo_exp4c3_balanced_lr1e4_ent002) | EXP_PPO_004C-3 | Balanced sampling + entropy ablation |
| [`models/ppo_exp4c2_balanced_lr1e4_bc001/`](file:///models/ppo_exp4c2_balanced_lr1e4_bc001) | EXP_PPO_004C-2 | Supervised BC auxiliary loss ablation |
| [`models/ppo_exp4c1_balanced_lr75e5/`](file:///models/ppo_exp4c1_balanced_lr75e5) | EXP_PPO_004C-1 | Intermediate learning rate ablation |
| [`models/ppo_exp4c0_balanced_lowlr/`](file:///models/ppo_exp4c0_balanced_lowlr) | EXP_PPO_004C-0 | Low learning rate ablation |
| [`models/ppo_exp4b_balanced/`](file:///models/ppo_exp4b_balanced) | EXP_PPO_004B | Frequency-weighted balanced sampling |
| [`models/ppo_exp4a2_bc/`](file:///models/ppo_exp4a2_bc) | EXP_PPO_004A-2 | Warm-start + static KL penalty to frozen reference |
| [`models/ppo_exp4a/`](file:///models/ppo_exp4a) | EXP_PPO_004A | Pure warm-start initialization |
| [`models/ppo/`](file:///models/ppo) | Base PPO | Initial training runs and base artifacts |
| [`models/ppo_sanity/`](file:///models/ppo_sanity) | Sanity Check | Step 10.1 pipeline sanity check |
