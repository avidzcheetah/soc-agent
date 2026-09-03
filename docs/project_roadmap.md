# Project Roadmap & Checklist

This document tracks the overall progress of the Autonomous SOC Agent project across its two core phases: SecBERT Representation Learning (Phase 1) and Deep Reinforcement Learning (Phase 2).

## Phase 1: Representation Learning (Completed)
- [x] **Data Preprocessing & Curation**
  - [x] Process CISSM Cyber Events dataset
  - [x] Process rcATT Threat Intelligence reports
  - [x] Generate synthetic data for rare classes to balance the action space
- [x] **SecBERT Fine-Tuning**
  - [x] Implement training pipeline (`src/secbert/`)
  - [x] Train SecBERT using Hugging Face Transformers
  - [x] Log experiments and metrics to TensorBoard
- [x] **Model Evaluation & Selection**
  - [x] Evaluate top-1 and top-2 accuracy
  - [x] Generate confusion matrices and classification reports
  - [x] Select the absolute best checkpoint (`EXP_20260727_001` / FP32 Ablation)
- [x] **Phase 1 Finalization**
  - [x] Cleanly separate Phase 1 codebase into `src/secbert/` and `scripts/secbert/`
  - [x] Set up the `models/` directory structure for production usage

## Phase 2: Deep Reinforcement Learning (PPO v1.0 Core Complete)
- [x] **SecBERT State Encoder** (`src/encoder.py`)
  - Frozen fine-tuned SecBERT model extracting verified 768-dimensional contextual state embeddings.
- [x] **SOC Environment** (`src/environment.py`)
  - Gymnasium contextual bandit environment mapping incident embeddings to the 20 mitigation actions.
  - Verified across reset, step, reward calculation (+1 / -1), eval traversal, and full reproducibility seeding (NumPy + PyTorch).
- [x] **Actor (Policy) Network** (`src/ppo/networks.py:ActorNetwork`)
  - 3-layer MLP (`768 → 512 → 256 → 20`) outputting unnormalized action logits for categorical sampling.
- [x] **Critic (Value) Network** (`src/ppo/networks.py:CriticNetwork`)
  - 3-layer MLP (`768 → 512 → 256 → 1`) estimating scalar state values $V(s)$.
- [x] **Trajectory Buffer** (`src/ppo/memory.py:PPOMemory`)
  - Trajectory memory buffer storing `(state, action, reward, log_prob, value, done)` transitions with clean lifecycle management.
- [x] **Generalized Advantage Estimation (GAE)** (`src/ppo/agent.py:compute_gae`)
  - Backward-pass GAE $(\gamma=0.99, \lambda=0.95)$ and target returns calculation $R_t = \hat{A}_t + V(s_t)$.
- [x] **PPO Core Agent & Optimization Loop** (`src/ppo/agent.py:PPOAgent`)
  - PPO clipped surrogate objective ($L^{CLIP}, \epsilon=0.2$).
  - Policy entropy bonus ($S[\pi], c_2=0.01$) for controlled exploration.
  - Value function MSE loss ($L^{VF}$) with detached targets.
  - Independent Adam optimizers (`lr_actor=3e-4`, `lr_critic=1e-3`) with gradient clipping (`max_grad_norm=1.0`).
  - Passed rigorous formal code audit (95/100).
- [x] **Step 8: Mini-Batch Updates**
  - Shuffled mini-batch rollout training for sample-efficient gradient updates across epochs.
- [x] **Step 9: PPO Training Pipeline** (`src/ppo/trainer.py` / `train_ppo.py`)
  - Complete training loop with epoch-level logging, TensorBoard metrics, and early stopping.
- [x] **Step 10: Checkpointing & Research Evaluation Upgrade**
  - Model weight saving, loading, and best-policy checkpoint selection based on validation Macro F1.
  - Contextual Bandit GAE formulation fixed (`done=True`).
  - Upgraded metrics (Macro F1, Weighted F1, MCC) for class imbalance.
  - PPO Sanity Training passed on real SecBERT embeddings.
- [x] **Step 11: Evaluation Pipeline & Baselines** (`evaluate_ppo.py`)
  - Comprehensive evaluation against SecBERT supervised baseline.
  - Research analysis documented proving RL convergence on contextual bandit.
- [ ] **Phase 3: Connect to the Ubuntu SOC Lab**
  - Real-world integration with Wazuh SIEM, osquery, and Suricata for live telemetry and response automation.
