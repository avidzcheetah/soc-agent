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
- [x] **Step 12: Reward Engineering (Exp 0–3)**
  - Naive +1/−1, class-aware, response-aware, and bounded hybrid rewards.
  - Conclusion: reward shaping alone insufficient. Pivot to structural interventions.

## Phase 2B: PPO Recovery — Structural Interventions (Completed)
- [x] **Step 13: Behavioral Cloning Warm-Start** (`scripts/ppo/warmstart_actor.py`)
  - [x] Knowledge distillation: SecBERT classifier logits → PPO Actor MLP (KL divergence)
  - [x] Independent warm-start evaluation: Val Acc 94.74%, Macro F1 0.7438, Weighted F1 0.9467, MCC +0.9377
- [x] **Step 14: Warm-Start PPO Training** (`scripts/ppo/train.py`)
  - [x] PPO fine-tuning from warm-started Actor with simple +1/−1 reward (Exp 4A)
  - [x] KL-Regularization ablation (Exp 4A-1): over-constrained, dropped to Macro F1 0.7027
  - [x] BC Preservation Loss ablation (Exp 4A-2): preserved accuracy but suppressed Macro F1 to 0.7350
- [x] **Step 15: Balanced Experience Sampling** (`src/environment.py`)
  - [x] Capped inverse-sqrt class frequency sampling implemented and verified
  - [x] Exp 4B (lr=3e-4): First PPO variant to beat SecBERT across all 4 metrics. Val Macro F1 0.7423
- [x] **Step 16: Learning Rate Optimization & Entropy Tuning** (Exp 4C series)
  - [x] Conservative lr=1e-4 with capped balanced sampling (Exp 4C-0): Val Macro F1 0.7928
  - [x] Refined lr=7.5e-5 (Exp 4C-1): Confirmed stability around 1e-4 region
  - [x] Mild BC loss hybrid (Exp 4C-2): BC term suppressed Macro F1 to 0.7451
  - [x] Higher entropy c₂=0.02 (Exp 4C-3): Matched 4C-0 peak, late collapse still occurred
  - [x] Early stopping (Exp 4C-4): Stopped prematurely (patience too aggressive)
  - [x] Periodic checkpointing (Exp 4C-5): Confirmed stable optimal region at iterations 125-135
  - [x] Test evaluation of 4C-5: **Test Acc 94.41%, Macro F1 0.7057, WF1 0.9422, MCC +0.9340**

## Phase 2C: Generalization & Constraint Research (Completed)
- [x] **Step 17: Val-to-Test Generalization Gap Diagnosis**
  - [x] Identified Class 15 (`restore_defense_config`) as primary test-set regression
  - [x] Corrected representation mismatch bug in diagnostic scripts (`last_hidden_state[:, 0, :]` → `pooler_output`)
  - [x] Confirmed PPO-135 has no `isolate_host` collapse under corrected representation
  - [x] Root cause: Class 15 vs Class 13 semantic overlap at data level, not PPO failure
- [x] **Step 18: ATGP Loss Design & Implementation** (`src/ppo/agent.py`)
  - [x] Formal mathematical design of per-sample confidence-gated KL divergence loss
  - [x] Implementation in `PPOAgent.update()`: atgp_lambda, atgp_tau hyperparameters added
  - [x] CLI arguments added to `scripts/ppo/train.py`: `--atgp_lambda`, `--atgp_tau`
- [x] **Step 19: ATGP Training & Evaluation** (Exp 4D)
  - [x] Training: τ=0.3, λ=1.0, lr=1e-4, c₂=0.02, balanced sampling, 150 iterations
  - [x] Validation: Acc 94.80%, Macro F1 0.7485†, WF1 0.9474, MCC +0.9385
  - [x] **Test (untouched): Acc 94.74%, Macro F1 0.7164, WF1 0.9461, MCC +0.9378**
  - [x] Class 13 F1: 0.7843 (+0.024 vs 4C-5). Class 15 F1: 0.3333 (+0.153 vs 4C-5)
  - [x] **ATGP improved all metrics over 4C-5 on the untouched test set** → New best policy
  - Checkpoint: `models/ppo_exp4d_atgp/best_ppo_policy.pt`
  - Report: `results/final_evaluation_exp4d_atgp.txt`

  *† Macro F1 difference vs 4C-5 on validation is due to single N=1 class (snapshot_forensics) changing prediction. Excluding it, 4D avg F1 across 17 classes is 0.7925 vs 4C-5's 0.7804.*

## Phase 3: Integration (Pending)
- [ ] Configure Wazuh, Caldera, and OSQuery simulation setups.
- [ ] Connect the agent to the Ubuntu SOC Lab for live telemetry and response automation.
- [ ] Run integrated evaluation and generate performance reports.
