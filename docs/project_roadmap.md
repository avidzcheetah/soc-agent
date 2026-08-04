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

## Phase 2: Deep Reinforcement Learning (In Progress)
- [x] **SecBERT State Encoder** (`src/encoder.py`)
  - Freeze the chosen SecBERT model to extract 768-dimensional state embeddings securely.
- [ ] **SOC Environment** (`environment.py`)
  - Gymnasium contextual bandit environment mapping alerts to the 20 mitigation actions.
- [ ] **Policy Network** (`policy_network.py`)
  - The actor network deciding which action to take based on the state.
- [ ] **Value Network** (`value_network.py`)
  - The critic network estimating the expected reward for a given state.
- [ ] **PPO Agent** (`ppo_agent.py`)
  - The core Proximal Policy Optimization reinforcement learning algorithm.
- [ ] **Training Script** (`train_ppo.py`)
  - The entry point to train the PPO agent within the SOC Environment.
- [ ] **Evaluation Script** (`evaluate_ppo.py`)
  - Metrics and baseline comparisons against DQN/LLMs/Rule-based systems.
- [ ] **Connect to the Ubuntu SOC Lab**
  - Real-world integration with Wazuh SIEM, osquery, and Suricata.
