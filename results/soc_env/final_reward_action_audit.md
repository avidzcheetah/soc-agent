# Final Reward and Action-Selection Audit for the SOC Simulation Environment

**Investigation Scope:** Reward Dynamics, Action-Effectiveness Landscape, and Policy Collapse Audit  
**Artifact:** `results/soc_env/final_reward_action_audit.md`  
**Date:** 2026-10-22  
**Benchmark Configuration:** 7 Scenarios, 12 Stages, 60 Evaluation Episodes (Seeds 1000–1059)  

---

## 1. Executive Summary

This audit establishes why sequential PPO training converged to degenerate single-action policies across both completed sequential experiments:
- **Experiment 001 (`escalation_cost = 0.0`):** Collapsed to **`always_escalate` (Action 19)** (100% escalation rate, 0% containment, 0% clearance, Return: $-0.10 \pm 0.00$).
- **Experiment 002 (`escalation_cost = 1.0`):** Collapsed to **`always_isolate_host` (Action 7)** (100% containment, 100% false disruption, 0% clearance, Return: $-0.09 \pm 1.02$).

To identify the root cause without ungrounded assumptions, we audited the environment transition mechanics, executed an empirical sweep of **all 20 discrete actions** across the standard 60-episode benchmark under both reward configurations, and evaluated the training and evaluation harnesses.

### Primary Audit Findings

1. **The Exp001 Zero-Variance Escalation Trap:**
   Under `escalation_cost = 0.0`, Action 19 terminates on step 1 with **zero variance** ($\mathbb{E}[R] = -0.10 \pm 0.00$), perfectly bypassing all downstream attack progression penalties ($-2.00$ breach, $-0.50$ ineffective) and false disruption penalties ($-1.00$). PPO rapidly locked onto this deterministic shortcut.

2. **The Exp002 Universal Containment Trap:**
   When escalation is penalized (`escalation_cost = 1.0`, yielding $-1.10$), Action 7 (`isolate_host`) becomes the mathematical optimum among all containment actions. In this simulation, host isolation cuts network egress, neutralizing all multi-stage attacks across all 4 malicious scenarios (yielding $+0.79$ average malicious return). While it incurs a $-1.00$ disruption penalty on the 3 benign scenarios ($-1.10$ return), the weighted expected return across the benchmark (32 malicious, 28 benign) is **$-0.09$**, outperforming all other fixed containment actions (e.g., `kill_process` is $-1.37$, `block_dest_ip` is $-0.73$, `disable_account` is $-1.37$).

3. **Investigation vs. Clearance Paradox:**
   Action 1 (`enable_deep_logging`) achieved the highest overall numerical return (**$-0.07 \pm 1.84$**), but it achieved **100% malicious breach** (0% containment). Because `enable_deep_logging` is classified as both effective on benign scenarios ($+2.00$ reward) and investigative on malicious scenarios ($+0.20$ reward), it allows attacks to progress unchecked to full compromise.

4. **Evaluation Harness Bug Uncovered:**
   In `src/ppo/trainer_sequential.py:evaluate()`, the periodic evaluation loop checked `is_benign = bool(info.get("is_benign", False))`. However, `SequentialSOCEnvironment.step()` omits `"is_benign"` from its return `info` dictionary (it is only present in `reset()`). Consequently, during training, all evaluation episodes were logged with `eval_benign.num_episodes = 0`, falsely attributing benign outcomes to the malicious metric bucket. (The standalone script `scripts/evaluate_soc_baselines.py` avoided this bug by reading `is_benign` from `reset()`).

5. **Representation & Reward Interaction:**
   Because frozen SecBERT pooler embeddings exhibit severe anisotropy (pairwise cosine similarities $>0.985$ and linear inseparability across scenarios under grouped cross-validation), the policy cannot reliably distinguish benign from malicious states. Given high breach penalties ($-2.00$ breach $+ -0.50$ ineffective) versus lower disruption penalties ($-1.00$), PPO acts as a risk-averse expected utility maximizer, converging to unconditional host isolation.

---

## 2. Exact Reward Calculation Table

The environment's reward components are defined in `RewardConfig` (`src/environment_sequential.py:L24-39`). On every step, `step_reward = step_cost (-0.10)` plus any outcome-triggered modifiers.

| Outcome State | Step Cost | Reward Modifiers | Net Step Reward | Termination Status | Next State / Transition |
| :--- | :---: | :--- | :---: | :---: | :--- |
| **Immediate Malicious Containment** | $-0.10$ | `+containment_reward` ($+2.00$) | **$+1.90$** | Terminated | Episode ends; `contained=True`, `cleared=False` |
| **Prior-Effect Neutralization** (Network Cut) | $-0.10$ | Mismatched disruption ($-1.50$) + `+containment_reward` ($+2.00$) | **$+0.40$** | Terminated | Prior `isolate_host` neutralizes Stage 1; episode ends |
| **Correct Benign Clearance** | $-0.10$ | `+benign_clearance_reward` ($+2.00$) | **$+1.90$** | Terminated | Episode ends; `cleared=True`, `contained=False` |
| **False Disruption (Benign)** | $-0.10$ | `+disruption_penalty` ($-1.00$) | **$-1.10$** | Terminated | Disruptive action executed on benign event; episode ends |
| **Mismatched Disruption (Malicious)** | $-0.10$ | `+disruption_penalty` ($-1.00$) + `+ineffective_penalty` ($-0.50$) | **$-1.60$** | Advances Stage | Advances to next attack stage; registers persistent effect |
| **Investigative Action (Malicious)** | $-0.10$ | `+investigation_reward` ($+0.20$) | **$+0.10$** | Advances Stage | Non-disruptive triage; attack advances |
| **Investigative Action (Benign)** | $-0.10$ | `+investigation_reward` ($+0.20$) | **$+0.10$** | Advances Stage | Triage; benign scenario advances toward bounded limit |
| **Ineffective Action (Malicious)** | $-0.10$ | `+ineffective_penalty` ($-0.50$) | **$-0.60$** | Advances Stage | Ineffective response; attack advances |
| **Ineffective Action (Benign)** | $-0.10$ | `+ineffective_penalty` ($-0.50$) | **$-0.60$** | Advances Stage | Ineffective response; advances toward bounded limit |
| **Breach / Compromise** (Final Stage) | $-0.10$ | `+ineffective_penalty` ($-0.50$) + `+breach_penalty` ($-2.00$) | **$-2.60$** | Terminated | Attack reaches end uncontained; `compromised=True` |
| **Benign Unresolved** (Final Stage) | $-0.10$ | None (bounded triage limit reached) | **$-0.10$** | Terminated | Bounded triage exhausted without clearance |
| **Human Escalation (Exp001)** | $-0.10$ | `+escalation_reward` ($0.00$) - `escalation_cost` ($0.00$) | **$-0.10$** | Terminated | Episode ends immediately; `escalated=True` |
| **Human Escalation (Exp002)** | $-0.10$ | `+escalation_reward` ($0.00$) - `escalation_cost` ($1.00$) | **$-1.10$** | Terminated | Episode ends immediately; `escalated=True` |

---

## 3. Action-Effectiveness Landscape Across Scenarios

Across the 7 scenarios and 12 stages, each of the 20 actions falls into one of four operational categories: **Effective** (E), **Investigative** (I), **Disruptive** (D), or **Ineffective** (—).

| Index | Action Name | `c2_s0` | `c2_s1` | `c2_s2` | `priv_s0` | `priv_s1` | `back_s0` (Ben) | `lat_s0` | `lat_s1` | `rce_s0` | `rce_s1` | `dev_s0` (Ben) | `aud_s0` (Ben) | Persistent Effect Registered |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | `monitor` | I | — | — | — | — | **E / I** | — | — | — | — | **E / I** | **E / I** | None |
| **1** | `enable_deep_logging` | I | I | I | I | I | **E / I** | I | I | I | I | **E / I** | **E / I** | None |
| **2** | `create_ioc_alert` | — | I | — | — | — | — | — | — | I | — | — | — | None |
| **3** | `block_source_ip` | — | — | — | — | — | — | — | — | **E** | — | — | D | None |
| **4** | `block_dest_ip` | — | **E** | **E** | — | — | D | — | — | **E** | **E** | D | D | `blocked_ips` |
| **5** | `dns_sinkhole` | — | — | — | — | — | — | — | — | — | — | — | — | None |
| **6** | `block_port` | — | — | — | — | — | — | — | — | — | — | — | — | None |
| **7** | `isolate_host` | D | **E** | **E** | D | **E** | D | D | **E** | — | **E** | D | D | `isolated_hosts` |
| **8** | `kill_process` | **E** | **E** | — | **E** | — | D | — | — | — | — | D | D | `terminated_pids` |
| **9** | `quarantine_file` | — | — | — | — | — | — | — | — | — | — | — | — | None |
| **10** | `quarantine_email` | — | — | — | — | — | — | — | — | — | — | — | — | None |
| **11** | `reset_credentials` | — | — | — | **E** | — | — | **E** | **E** | — | — | — | — | `disabled_accounts` |
| **12** | `disable_account` | D | D | — | **E** | **E** | D | **E** | **E** | D | — | D | — | `disabled_accounts` |
| **13** | `remove_persistence` | — | — | — | — | **E** | — | — | — | — | — | — | — | None |
| **14** | `restore_registry` | — | — | — | — | — | — | — | — | — | — | — | — | None |
| **15** | `restore_defense_config` | — | — | — | — | — | — | — | — | — | — | — | — | None |
| **16** | `patch_vulnerability` | — | — | — | — | — | — | — | — | **E** | — | — | — | None |
| **17** | `snapshot_forensics` | I | I | I | I | I | — | I | I | I | I | — | — | None |
| **18** | `sandbox_redirect` | — | — | — | — | — | — | — | — | — | — | — | — | None |
| **19** | `escalate_to_human` | *Term* | *Term* | *Term* | *Term* | *Term* | *Term* | *Term* | *Term* | *Term* | *Term* | *Term* | *Term* | None |

### Key Observations on Action Effectiveness:
1. **Narrow Effectiveness of Actions 5, 6, 9, 10, 14, 15, 18:** Seven actions in the discrete action space are never effective, investigative, or disruptive in any of the seven scenarios. They always incur the `ineffective_penalty` ($-0.50$) on every step.
2. **Persistent Response Asymmetry:** Action 7 (`isolate_host`) sets `isolated_hosts`. In `scen_c2_exfil` and `scen_network_rce_tunnel`, Stage 1 has `requires_network=True`. As a result, taking Action 7 on Stage 0 terminates on Step 1 with containment reward ($+2.00$) via `contained_by_prior_isolation`.
3. **Disruption Overload on Benign Incidents:** Actions 4, 7, 8, and 12 are marked as disruptive across benign scenarios. Taking any of them on Stage 0 terminates the benign episode with $-1.10$.

---

## 4. Controlled 20-Action Sweep: Empirical Results

We evaluated all 20 fixed actions across the reproducible 60-episode benchmark (`seeds 1000–1059`, yielding 32 malicious and 28 benign episodes) under both reward configurations.

### Empirical Sweep Table

| Policy | Mean Return (Config A: `cost=0.0`) | Mean Return (Config B: `cost=1.0`) | Malicious Containment | Malicious Breach Rate | Benign Clearance | False Disruption | Escalation Rate | Mean Length |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Action 0 (`monitor`)** | $-0.81 \pm 2.54$ | $-0.81 \pm 2.54$ | 0.0% | 100.0% | **100.0%** | 0.0% | 0.0% | 1.60 |
| **Action 1 (`enable_deep_logging`)** | **$-0.07 \pm 1.84$** | **$-0.07 \pm 1.84$** | 0.0% | 100.0% | **100.0%** | 0.0% | 0.0% | 1.60 |
| **Action 2 (`create_ioc_alert`)** | $-1.82 \pm 1.16$ | $-1.82 \pm 1.16$ | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% | 1.60 |
| **Action 3 (`block_source_ip`)** | $-0.90 \pm 1.91$ | $-0.90 \pm 1.91$ | 43.8% | 56.2% | 0.0% | 28.6% | 0.0% | 1.37 |
| **Action 4 (`block_dest_ip`)** | $-0.73 \pm 1.84$ | $-0.73 \pm 1.84$ | 56.2% | 43.8% | 0.0% | 100.0% | 0.0% | 1.30 |
| **Action 5 (`dns_sinkhole`)** | $-2.03 \pm 1.34$ | $-2.03 \pm 1.34$ | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% | 1.60 |
| **Action 6 (`block_port`)** | $-2.03 \pm 1.34$ | $-2.03 \pm 1.34$ | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% | 1.60 |
| **Action 7 (`isolate_host`)** | **$-0.09 \pm 1.02$** | **$-0.09 \pm 1.02$** | **100.0%** | **0.0%** | 0.0% | 100.0% | 0.0% | 1.23 |
| **Action 8 (`kill_process`)** | $-1.37 \pm 1.74$ | $-1.37 \pm 1.74$ | 31.2% | 68.8% | 0.0% | 100.0% | 0.0% | 1.37 |
| **Action 9 (`quarantine_file`)** | $-2.03 \pm 1.34$ | $-2.03 \pm 1.34$ | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% | 1.60 |
| **Action 10 (`quarantine_email`)** | $-2.03 \pm 1.34$ | $-2.03 \pm 1.34$ | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% | 1.60 |
| **Action 11 (`reset_credentials`)** | $-0.84 \pm 1.91$ | $-0.84 \pm 1.91$ | 43.8% | 56.2% | 0.0% | 0.0% | 0.0% | 1.37 |
| **Action 12 (`disable_account`)** | $-1.37 \pm 2.40$ | $-1.37 \pm 2.40$ | 43.8% | 56.2% | 0.0% | 71.4% | 0.0% | 1.37 |
| **Action 13 (`remove_persistence`)** | $-1.58 \pm 1.60$ | $-1.58 \pm 1.60$ | 18.8% | 81.2% | 0.0% | 0.0% | 0.0% | 1.60 |
| **Action 14 (`restore_registry`)** | $-2.03 \pm 1.34$ | $-2.03 \pm 1.34$ | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% | 1.60 |
| **Action 15 (`restore_defense_config`)** | $-2.03 \pm 1.34$ | $-2.03 \pm 1.34$ | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% | 1.60 |
| **Action 16 (`patch_vulnerability`)** | $-0.84 \pm 1.91$ | $-0.84 \pm 1.91$ | 43.8% | 56.2% | 0.0% | 0.0% | 0.0% | 1.37 |
| **Action 17 (`snapshot_forensics`)** | $-1.23 \pm 0.59$ | $-1.23 \pm 0.59$ | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% | 1.60 |
| **Action 18 (`sandbox_redirect`)** | $-2.03 \pm 1.34$ | $-2.03 \pm 1.34$ | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% | 1.60 |
| **Action 19 (`escalate_to_human`)** | **$-0.10 \pm 0.00$** | **$-1.10 \pm 0.00$** | 0.0% | 0.0% | 0.0% | 0.0% | **100.0%** | 1.00 |

---

## 5. Comparative Evaluation: Baseline & Learned Policies

| Policy | Mean Return (Config A: `cost=0.0`) | Mean Return (Config B: `cost=1.0`) | Malicious Containment | Malicious Breach Rate | Benign Clearance | False Disruption | Escalation Rate | Mean Length |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Uniform Random** | $-0.63 \pm 1.67$ | $-0.68 \pm 1.67$ | 43.8% | 46.9% | 7.1% | 10.7% | 5.0% | 1.48 |
| **Rule-Based SOC Policy** | **$+0.42 \pm 1.92$** | **$+0.42 \pm 1.92$** | **75.0%** | **25.0%** | **42.9%** | **57.1%** | **0.0%** | 1.13 |
| **Pretrained PPO (Exp 4D ATGP)** | $-1.23 \pm 0.59$ | $-1.23 \pm 0.59$ | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% | 1.60 |
| **Sequential PPO Exp001** | **$-0.10 \pm 0.00$** | $-1.10 \pm 0.00$ | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 1.00 |
| **Sequential PPO Exp002** | **$-0.09 \pm 1.02$** | **$-0.09 \pm 1.02$** | **100.0%** | **0.0%** | 0.0% | 100.0% | 0.0% | 1.23 |

### Critical Analytical Insights:
1. **PPO Mimics Fixed Actions Exactly:**
   - Pretrained PPO output is identical in every metric to **Action 17 (`snapshot_forensics`)**.
   - Exp001 output is identical in every metric to **Action 19 (`escalate_to_human`)**.
   - Exp002 output is identical in every metric to **Action 7 (`isolate_host`)**.
2. **The Return Illusion Across Experiments:**
   - In Exp001, PPO achieved $-0.10$ under `cost=0.0`. Re-evaluated under `cost=1.0`, its return falls to $-1.10$.
   - Exp002 achieved $-0.09$. Comparing $-0.10$ to $-0.09$ without accounting for the reward function change created the false appearance that Exp002 performed similarly to Exp001. Under the standardized `cost=1.0` benchmark, Exp002 improved episodic return by $+1.01$ over Exp001, but did so entirely by substituting one fixed action for another.
3. **The Rule-Based Baseline Proves Learnability:**
   - The rule-based policy is the **only policy** that achieves a positive return ($+0.42$) and non-zero performance across both classes (75% containment, 42.9% clearance). It does so strictly using observable telemetry attributes from `info["telemetry"]`.

---

## 6. Evaluation Methodology Audit Findings

We conducted a line-by-line audit of the evaluation and training harnesses:

### Finding 1: Evaluation Logging Defect in `trainer_sequential.py` (Confirmed)
- **Defect Location:** `src/ppo/trainer_sequential.py:L270-282`
- **Mechanism:** In the periodic evaluation loop of `TrainerSequential.evaluate()`, the code inspects `is_benign = bool(info.get("is_benign", False))` after the episode terminates. However, `SequentialSOCEnvironment.step()` does **not** include `"is_benign"` in its return `info` dictionary; `"is_benign"` is only included in the `info` returned by `reset()`.
- **Consequence:** During training, `is_benign` always evaluated to `False`. In `training_logs.json`, `eval_benign.num_episodes` was recorded as 0 for all evaluations, and all 20 evaluation episodes were miscategorized under `eval_malicious`.
- **Impact on Policy Selection:** Checkpoint selection is based on `eval_mean_return` (which aggregates `ep_returns` directly), so the bug did **not** corrupt checkpoint selection. However, it completely corrupted the in-training sub-metric reporting.
- **Standalone Benchmark Validation:** `scripts/evaluate_soc_baselines.py` did not suffer from this defect because it explicitly captured `is_benign` from `reset()` before entering the step loop (`lines 50–58`).

### Finding 2: Hidden-Label Leakage Audit (Passed Clean)
- Neither the PPO agent nor the rule-based heuristic accesses hidden simulator attributes.
- The PPO agent accepts only the 768-dimensional float tensor `obs`.
- `rule_based_policy` reads strictly from `info["telemetry"]` (process name, cmdline, rule description, alert level, MITRE tags). A unit test in `tests/test_environment_sequential.py` verifies this constraint.

### Finding 3: Seed Reproducibility (Passed Clean)
- Evaluated seeds (`range(1000, 1060)`) generate a deterministic sequence of 32 malicious and 28 benign episodes across all policies.

---

## 7. Evidence-Supported Explanation of PPO's Action Collapse

The collapse is not an arbitrary optimization anomaly. It is the mathematically predictable outcome of three intersecting mechanisms:

```
┌────────────────────────────────────────────────────────┐
│  Mechanism 1: State Inseparability (SecBERT Anisotropy)│
│  - Frozen pooler embeddings have cosine similarity >0.985│
│  - Linear probe LOGO-CV Balanced Accuracy: 0.5000       │
│  - Agent cannot differentiate benign from malicious text│
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  Mechanism 2: Asymmetric Risk Landscape                │
│  - Breach cost: -2.00 (penalty) + -0.50 (ineffective)  │
│  - False disruption cost: -1.00                        │
│  - Risk of monitoring: E[R] = -0.81 (100% breach)      │
│  - Risk of isolation:  E[R] = -0.09 (0% breach)        │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  Mechanism 3: The Universal Network Kill-Switch        │
│  - Persistent isolated_hosts neutralizes Stage 1 C2    │
│    and reverse tunnels in the same step (+0.40 / +1.40) │
│  - Stage 1 of DB dump and sudo persistence accepts Act 7│
│  - Act 7 achieves 100% containment across all attacks  │
└────────────────────────────────────────────────────────┘
```

1. **State Ambiguity:** Because the 768-dim embeddings reside in an anisotropic cone (average cosine similarity $0.9914$), the policy network faces high observation ambiguity.
2. **Rational Risk Minimization:** Faced with an alert it cannot reliably classify, the policy must choose between:
   - Playing `monitor` (Action 0): If it is an attack, the agent suffers catastrophic breach penalties ($-3.20$ to $-3.80$). Expected return: **$-0.81$**.
   - Playing `isolate_host` (Action 7): If it is benign, the agent suffers a disruption penalty ($-1.10$). If it is an attack, it contains it with positive return ($+0.30$ to $+1.40$). Expected return: **$-0.09$**.
3. **PPO's Policy Convergence:** PPO's actor and critic networks converged to the global maximum of the expected return surface available under observation ambiguity: **Action 7**.

---

## 8. The Single Most Justified Next Experiment

To resolve this collapse scientifically, we must change **only one principal variable**: the **Observation Representation**.

Changing reward weights alone will not resolve the problem:
- If we increase the false disruption penalty to penalize isolation, an agent that cannot separate states will simply switch to **Action 1 (`enable_deep_logging`)**, which earns **$-0.07$** while allowing 100% of attacks to breach, or to **Action 0 (`monitor`)**, allowing complete system compromise.
- Reward shaping cannot fix an uninformative state.

### Recommended Next Experiment: Experiment 003 — Telemetry-Augmented Observation (Hybrid State Representation)

- **Independent Variable (Changed):** State Representation $\mathbf{s}$.
  - Replace the pure 768-dim text embedding with a concatenated hybrid representation:
    $$\mathbf{s} = [\mathbf{e}_{\text{SecBERT}} \, (768), \, \mathbf{v}_{\text{telemetry}} \, (k)]$$
  - $\mathbf{v}_{\text{telemetry}}$ is an observable, normalized numerical/categorical vector encoding:
    1. SIEM Alert Level (normalized $0.0 - 1.0$).
    2. User Privilege Tier (Root/Admin $= 1.0$, Service Account $= 0.5$, Unprivileged User $= 0.0$).
    3. Host Role Classification (Domain Controller / Bastion $= 1.0$, Web Portal / DB $= 0.5$, Workstation $= 0.0$).
    4. Process Origin Category (Binary in `/tmp` or `/dev/shm` $= 1.0$, Standard system path $= 0.0$).
    5. Network Egress Flag (External unapproved IP present $= 1.0$, Internal/none $= 0.0$).
- **Controlled Variables (Kept Strictly Identical to Exp002):**
  - Environment transition logic and scenario catalog (7 scenarios, 12 stages).
  - Reward configuration (`RewardConfig(escalation_cost=1.0)`).
  - Master seed (42), rollout budget (64 steps), total iterations (50), PPO learning rates (`lr_actor=3e-4`, `lr_critic=1e-3`), clipping ($\epsilon=0.2$).
  - Frozen SecBERT checkpoint (`models/secbert_finetuned`).
- **Hypothesis:** Augmenting the observation space with structured telemetry features will provide the linear and non-linear separability that the frozen text pooler obscures, enabling PPO to learn context-aware actions (monitoring benign events and containing attacks) and achieve a positive return surpassing the rule-based baseline ($+0.42$).

---

## 9. Limitations & Claims Not Supported by Evidence

1. **Not a Proof of SecBERT's General Incapacity:** This audit evaluated frozen pooler representations on 7 simulated scenarios. It does **not** prove that fine-tuned transformers cannot distinguish cyber text in broader domains.
2. **Not a Simulator Flaw:** The simulated response mechanics (network isolation, account disabling, process termination) behave consistently and realistically according to their design. The collapse is driven by information starvation at the policy input layer under asymmetric risk.
3. **Simulation Scale Limit:** With 7 scenarios, 12 stages, and 60 benchmark episodes, results demonstrate local MDP convergence properties rather than broad enterprise distribution generalization.

---

## 10. Verification of Existing Tests

All 38 non-pytest unit tests were executed and passed cleanly:
- `tests/test_telemetry.py` (12 tests): PASS
- `tests/test_environment_sequential.py` (14 tests): PASS
- `tests/test_trainer_sequential.py` (12 tests): PASS
- **Total:** 38 tests, 0 failures, 0 errors (Runtime: 2.03s).
