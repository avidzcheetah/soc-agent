# Sequential PPO Training Experiment Report: exp001

**Date:** 2026-10-12 11:42:06 UTC  
**Status:** Completed  
**Experiment ID:** `exp001`  
**Master Random Seed:** 42  
**Total Training Steps:** 3,200 environment transitions (50 iterations × 64 steps)  
**Execution Time:** 5.26 seconds  
**Initialization Checkpoint:** `models/ppo_exp4d_atgp/best_ppo_policy.pt` (Exp 4D ATGP)  
**Trained Checkpoint Artifacts:** `models/ppo_sequential_exp001/`  
- `initial_policy.pt`: Frozen copy of warm-start policy prior to training.
- `best_sequential_policy.pt`: Policy with highest episodic return on validation seeds ($-0.100$).
- `final_sequential_policy.pt`: Policy state at final iteration 50.

---

## 1. Executive Summary & Core Research Findings

This experiment executes the **first controlled sequential PPO training experiment** in the multi-stage SOC Simulation Environment (`SequentialSOCEnvironment`). The objective was to train PPO to navigate multi-step incident stages and evaluate whether reinforcement learning improves performance over existing static and heuristic baselines.

### Primary Empirical Findings:
1. **Apparent Numerical Improvement:** The trained sequential PPO policy increased its mean episodic return from **$-1.23 \pm 0.59$** (frozen pretrained PPO) to **$-0.10 \pm 0.00$** (best checkpoint).
2. **Discovery of the "Defensive Escalation Trap":** Despite the numerical return improvement, **malicious containment remained at 0.0%** and **benign clearance remained at 0.0%**. Rather than learning active, stage-specific containment (e.g. killing processes, isolating hosts, resetting credentials), the policy experienced policy collapse into **Action 19 (`escalate_to_human`)**, escalating 100% of incidents.
3. **Comparative Standing:** Sequential PPO matched the fixed **"Always Escalate"** baseline ($-0.10 \pm 0.00$), but **failed to outperform the observable-telemetry Rule-Based SOC policy** ($+0.42 \pm 1.92$ return, $75.0\%$ containment).
4. **Research Integrity:** Reinforcement learning in this unconstrained setting did *not* autonomously discover autonomous incident containment. It discovered a risk-avoidance local optimum created by the relative penalty weighting between breach ($-3.0$) and escalation ($-0.10$).

---

## 2. Experiment Setup & Hyperparameters

The experiment maintained strict separation between training exploration, periodic validation, and final held-out benchmark evaluation. All environment dynamics, state embeddings (frozen SecBERT), and scenario reward parameters were held strictly constant.

| Hyperparameter / Setting | Value | Description |
|:---|:---|:---|
| `experiment_id` | `exp001` | Unique experiment identifier |
| `master_seed` | `42` | Master RNG seed for reproducibility |
| `state_encoder` | `SecBERTStateEncoder` | 768-dimensional frozen transformer embeddings |
| `action_dim` | `20` | Full CISA incident response action space |
| `lr_actor` | `1e-4` | Actor learning rate (fine-tuning regime) |
| `lr_critic` | `5e-4` | Critic learning rate |
| `gamma` | `0.99` | MDP discount factor across stage transitions |
| `gae_lambda` | `0.95` | Generalized Advantage Estimation smoothing factor |
| `clip_eps` | `0.2` | PPO surrogate objective clipping range |
| `c2_entropy` | `0.01` | Policy entropy bonus coefficient |
| `k_epochs` | `4` | SGD epochs per PPO update |
| `batch_size` | `32` | PPO mini-batch size |
| `rollout_steps` | `64` | Transitions collected per iteration |
| `total_iterations` | `50` | Total training iterations (3,200 transitions) |
| `eval_interval` | `5` | Checkpoint evaluation frequency |
| `eval_episodes` | `20` | Episodes evaluated per periodic check |
| `eval_seed_start` | `2000` | Periodic validation seed sequence [2000..2019] |
| `benchmark_seeds` | `1000..1059` | Held-out comparative evaluation seed sequence ($N=60$) |

---

## 3. Comparative Evaluation Benchmark (Held-Out Seeds 1000..1059, N=60)

All policies were evaluated against the identical 60 deterministic scenario episodes covering all 7 incident scenarios (4 malicious multi-stage attacks and 3 benign operational workflows).

| Policy | Mean Return ± Std | Malicious Containment | Malicious Breach | Benign Clearance | False Disruption | Escalation Rate | Mean Length |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Uniform Random** | $-0.63 \pm 1.67$ | 43.8% | 46.9% | 7.1% | 10.7% | 5.0% | 1.48 |
| **2. Always Monitor** | $-0.81 \pm 2.54$ | 0.0% | 100.0% | 100.0% | 0.0% | 0.0% | 1.60 |
| **3. Always Kill Process** | $-1.37 \pm 1.74$ | 31.2% | 68.8% | 0.0% | 100.0% | 0.0% | 1.37 |
| **4. Always Escalate** | **$-0.10 \pm 0.00$** | **0.0%** | **0.0%** | **0.0%** | **0.0%** | **100.0%** | **1.00** |
| **5. Rule-Based SOC Policy** | **$+0.42 \pm 1.92$** | **75.0%** | **25.0%** | **42.9%** | **57.1%** | **0.0%** | **1.13** |
| **6. Frozen Pretrained PPO** | $-1.23 \pm 0.59$ | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% | 1.60 |
| **7. Sequential PPO (Best Checkpoint)** | **$-0.10 \pm 0.00$** | **0.0%** | **0.0%** | **0.0%** | **0.0%** | **100.0%** | **1.00** |
| **8. Sequential PPO (Final Iteration)** | **$-0.10 \pm 0.00$** | **0.0%** | **0.0%** | **0.0%** | **0.0%** | **100.0%** | **1.00** |

---

## 4. Training Progression & Dynamics

| Iteration | Step Reward | Rollout Ep Ret | Containment | Clearance | Breach | Policy Loss | Value Loss | Entropy | Dominant Rollout Action |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **00 (Init)** | — | $-1.075$ | 0.0% | 0.0% | 40.0% | — | — | — | Act 17 (`snapshot_forensics`): 100% |
| **01** | $-0.761$ | $-1.316$ | 0.0% | 0.0% | 51.4% | $-0.0292$ | 3.3291 | 0.295 | Act 17 (90.6%) |
| **05** | $-0.703$ | $-1.182$ | 2.6% | 0.0% | 55.3% | $-0.0066$ | 0.5444 | 0.131 | Act 17 (93.8%) |
| **10** | $-0.692$ | $-1.233$ | 2.8% | 0.0% | 58.3% | $-0.0100$ | 0.4339 | 0.217 | Act 17 (92.2%) |
| **15** | $-0.637$ | $-0.971$ | 2.4% | 0.0% | 38.1% | $-0.0108$ | 0.5617 | 0.235 | Act 17 (87.5%) |
| **20** | $-0.705$ | $-1.159$ | 2.6% | 2.6% | 48.7% | $-0.0147$ | 0.9996 | 0.876 | Exploration widening |
| **25** | $-0.562$ | $-0.785$ | 2.2% | 0.0% | 21.7% | $-0.0636$ | 1.0215 | 1.246 | Entropy peak; Act 19 sampled |
| **30** | $-0.169$ | $-0.180$ | 3.3% | 0.0% | 1.7% | $-0.0079$ | 0.2522 | 0.600 | Act 19 (`escalate`): 85.9% |
| **35** | $-0.061$ | $-0.062$ | 4.8% | 0.0% | 0.0% | $-0.0109$ | 0.1406 | 0.187 | Act 19 (`escalate`): 95.3% |
| **40** | $-0.155$ | $-0.160$ | 0.0% | 0.0% | 0.0% | $-0.0304$ | 0.0825 | 0.113 | Act 19 (`escalate`): 98.4% |
| **45** | $-0.100$ | $-0.100$ | 0.0% | 0.0% | 0.0% | $-0.0000$ | 0.0000 | 0.016 | Act 19 (`escalate`): 100% |
| **50** | $-0.100$ | $-0.100$ | 0.0% | 0.0% | 0.0% | $-0.0000$ | 0.0000 | 0.007 | Act 19 (`escalate`): 100% |

---

## 5. In-Depth Root Cause Analysis: The "Defensive Escalation Trap"

### 5.1 The Initial Policy State
The warm-start checkpoint (`best_ppo_policy.pt`) was trained in a single-step contextual bandit setup. On high-severity alerts in that dataset, the supervised/bandit reward heavily favored Action 17 (`snapshot_forensics`). Consequently, at initialization, the policy selected Action 17 almost exclusively ($100\%$ on validation episodes).

### 5.2 The Transition Failure in Sequential MDP
In the sequential MDP, `snapshot_forensics` is classified as an **investigative action**:
- It does **not contain** an active threat (such as ransomware encryption, reverse shells, or credential dumping).
- When taken, the incident transitions to subsequent stages.
- When an attack progresses through all stages without an effective containment action (such as `isolate_host`, `kill_process`, or `reset_credentials`), the scenario terminates with `breach_compromised`, penalizing the agent with **$-3.0$**.
- On benign scenarios, investigative actions fail to clear the incident, resulting in `benign_unresolved` (penalty of **$-0.20$** per step).
Consequently, the warm-started policy suffered severe returns ($-1.23$ to $-1.31$) and a catastrophic breach rate of $100\%$ on malicious incidents.

### 5.3 The Gradient Incentive toward Risk Avoidance
During training iterations 20 to 25, the policy entropy bonus ($c_2 = 0.01$) expanded exploration (entropy peaked at $1.246$). During this exploration, the agent encountered Action 19 (`escalate_to_human`).
In the environment:
$$\text{Reward}(\text{escalate}) = -0.10 \quad \text{and} \quad \text{done} = \text{True}$$
Evaluating the expected return under epistemic uncertainty:
- Selecting an active containment action: Risk of picking the wrong action $\to$ incident transitions $\to$ breach penalty of **$-3.0$** or disruptive false positive penalty of **$-1.5$**.
- Selecting an investigative action: Ineffective $\to$ incident transitions $\to$ breach penalty of **$-3.0$**.
- Selecting escalation: Guaranteed immediate termination with return **$-0.10$**.

Because $-0.10 \gg -3.0$, the advantage function for Action 19 was overwhelmingly positive across all states. The policy gradient rapidly collapsed all probability mass into Action 19 between iterations 25 and 40. By iteration 50, entropy had decayed to $0.007$, and the policy had degenerated into a trivial **"Always Escalate"** agent.

---

## 6. Recommendations for Sequential PPO Phase C

To break out of the Defensive Escalation Trap in future experiments without altering ground-truth evaluation standards:

1. **Escalation Gating & Cost Realism:**
   In real SOC operations, escalating every ticket to Tier-3 engineers causes alert fatigue and analyst burnout. In future experiments, escalation should either:
   - Incur a higher operational cost (e.g. $-1.0$ or $-1.5$), making premature escalation less attractive than attempting correct mitigation.
   - Require an epistemic precondition (e.g., only permitted after at least one investigative step or when policy entropy is high).

2. **Imitation Warm-Starting / ATGP Regularization:**
   Unconstrained RL fine-tuning from a contextual bandit policy rapidly overfits to avoidance. Using the rule-based policy or expert demonstrations to warm-start the sequential policy via Behavioral Cloning (BC) or Adaptive Teacher-Guided Preservation (ATGP) would preserve containment reflexes.

3. **Intermediate Progress Rewards:**
   Multi-stage containment currently yields $+2.0$ only upon successful containment, while intermediate steps yield $-0.10$. Providing denser credit assignment for reducing threat velocity would encourage active intervention.

---

## 7. Artifact Integrity Verification

The following files were generated and verified during Experiment 001:
- Checkpoints:
  - `models/ppo_sequential_exp001/initial_policy.pt` (preserved warm-start state)
  - `models/ppo_sequential_exp001/best_sequential_policy.pt` (selected by validation return)
  - `models/ppo_sequential_exp001/final_sequential_policy.pt` (iteration 50 terminal state)
- Logs and Data:
  - `results/soc_env/exp001/training_config.json` (exact hyperparameter and environment snapshot)
  - `results/soc_env/exp001/training_logs.json` (50 iteration rollout and update logs)
  - `results/soc_env/exp001/comparative_evaluation_results.json` (granularity across all 8 policies)
  - `results/soc_env/exp001/experiment_report.md` (this report)

All existing baselines, pretraining checkpoints (`models/ppo_exp4d_atgp/best_ppo_policy.pt`), and the contextual bandit pipeline remain unmodified.