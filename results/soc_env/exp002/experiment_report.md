# Sequential PPO Training Experiment Report: exp002

**Date:** 2026-10-12 12:07:36 UTC  
**Status:** Completed  
**Experiment ID:** `exp002`  
**Master Random Seed:** 42  
**Total Training Steps:** 3,200 environment transitions (50 iterations × 64 steps)  
**Execution Time:** 4.14 seconds  
**Initialization Checkpoint:** `models/ppo_exp4d_atgp/best_ppo_policy.pt` (Exp 4D ATGP)  
**Experimental Variable Changed:** `escalation_cost = 1.0` (all other settings identical to Exp 001)  
**Trained Checkpoint Artifacts:** `models/ppo_sequential_exp002/`  
- `initial_policy.pt`: Initial policy snapshot prior to training.
- `best_sequential_policy.pt`: Policy with highest evaluation episodic return ($-0.370$ on validation seeds).
- `final_sequential_policy.pt`: Policy state at final iteration 50.

---

## 1. Executive Summary & Hypothesis Testing

Experiment 002 tested the **Escalation-Cost Hypothesis**:
> *Does penalizing escalation by setting `escalation_cost = 1.0` (making premature escalation equal in cost to false disruption, return = $-1.10$) break the `always_escalate` trap and incentivize PPO to learn autonomous multi-stage threat containment and benign clearance?*

### Measured Empirical Outcomes:
1. **The Escalation Trap was Broken:** Increasing `escalation_cost` to 1.0 completely suppressed premature escalation. The policy's escalation rate dropped from **100.0% in Exp 001 to 0.0% in Exp 002**.
2. **Emergence of 100% Threat Containment:** On malicious incident scenarios, the trained policy achieved **100.0% threat containment** (32/32 malicious episodes contained, 0% breach rate), driving malicious return from $-1.10$ up to **$+0.79 \pm 0.99$**.
3. **Replacement by a New Degenerate Equilibrium ("Scorched-Earth Host Isolation"):** The policy did **not** learn balanced incident discrimination. Instead of discerning benign administrative alerts from malicious attacks, the policy collapsed into **Action 7 (`isolate_host`) across 100% of benchmark steps**.
4. **Catastrophic False Disruption on Benign Alerts:** Because `isolate_host` was applied indiscriminately to all incidents, benign operational workflows (administrative backups, CI/CD builds, and vulnerability audit scans) suffered a **100.0% false disruption rate** (28/28 benign episodes disrupted, 0% benign clearance), yielding a benign return of **$-1.10$**.
5. **Net Episodic Return:** Across all 60 benchmark episodes (32 malicious, 28 benign), Exp 002 achieved a mean return of **$-0.09 \pm 1.02$**, slightly outperforming Exp 001 ($-0.10$ under cost=0 / $-1.10$ under cost=1) and all fixed baselines, but remaining below the feature-discriminating Rule-Based SOC Heuristic (**$+0.42 \pm 1.92$**).

---

## 2. Training Configuration & Controlled Experimental Parameters

All hyperparameters, seeds, and architectural configurations were held strictly identical to Experiment 001, with the sole exception of `escalation_cost`:

| Hyperparameter / Setting | Exp 001 Value | Exp 002 Value | Purpose / Description |
|:---|:---:|:---:|:---|
| `experiment_id` | `exp001` | `exp002` | Unique experiment identifier |
| `master_seed` | `42` | `42` | Master RNG seed for reproducibility |
| `escalation_cost` | `0.0` | **`1.0`** | **The sole independent variable under test** |
| `containment_reward` | `+2.0` | `+2.0` | Reward for mitigating active threat |
| `benign_clearance_reward` | `+2.0` | `+2.0` | Reward for dismissing benign activity |
| `disruption_penalty` | `-1.0` | `-1.0` | Cost of unnecessary service disruption |
| `breach_penalty` | `-2.0` | `-2.0` | Cost when malicious attack completes uncontained |
| `state_encoder` | `SecBERT` (768) | `SecBERT` (768) | Frozen transformer embeddings |
| `action_dim` | `20` | `20` | Full discrete CISA incident response actions |
| `lr_actor` | `1e-4` | `1e-4` | Actor network learning rate |
| `lr_critic` | `5e-4` | `5e-4` | Critic network learning rate |
| `gamma` | `0.99` | `0.99` | MDP discount factor |
| `gae_lambda` | `0.95` | `0.95` | Generalized Advantage Estimation factor |
| `clip_eps` | `0.2` | `0.2` | PPO surrogate clipping threshold |
| `c2_entropy` | `0.01` | `0.01` | Policy entropy bonus coefficient |
| `k_epochs` | `4` | `4` | SGD update epochs per iteration |
| `batch_size` | `32` | `32` | Mini-batch sample size |
| `rollout_steps` | `64` | `64` | Transition collection budget per update |
| `total_iterations` | `50` | `50` | Total training iterations (3,200 transitions) |
| `eval_interval` | `5` | `5` | Periodic checkpoint evaluation interval |
| `eval_episodes` | `20` | `20` | Validation episodes per eval check |
| `eval_seed_start` | `2000` | `2000` | Dedicated validation seeds [2000..2019] |
| `benchmark_seeds` | `1000..1059` | `1000..1059` | Final held-out evaluation seeds ($N=60$) |

---

## 3. Comparative Evaluation Benchmark (Held-Out Seeds 1000..1059, N=60)

Evaluated under the calibrated environment reward setting (`escalation_cost = 1.0`). All policies faced the exact same 60 scenario episodes:

| Policy | Mean Return ± Std | Malicious Containment | Malicious Breach | Benign Clearance | False Disruption | Escalation Rate | Mean Length | Action Distribution (Top Actions) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **1. Uniform Random** | $-0.68 \pm 1.67$ | 43.8% | 46.9% | 7.1% | 10.7% | 5.0% | 1.48 | Act 14: 8, Act 7: 8, Act 11: 7 |
| **2. Always Monitor** | $-0.81 \pm 2.54$ | 0.0% | 100.0% | 100.0% | 0.0% | 0.0% | 1.60 | Act 0: 96 |
| **3. Always Kill Process** | $-1.37 \pm 1.74$ | 31.2% | 68.8% | 0.0% | 100.0% | 0.0% | 1.37 | Act 8: 82 |
| **4. Always Escalate** | $-1.10 \pm 0.00$ | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 1.00 | Act 19: 60 |
| **5. Rule-Based SOC Policy** | **$+0.42 \pm 1.92$** | **75.0%** | **25.0%** | **42.9%** | **57.1%** | **0.0%** | **1.13** | Act 8: 28, Act 4: 22, Act 0: 12 |
| **6. Frozen Pretrained PPO** | $-1.23 \pm 0.59$ | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% | 1.60 | Act 17: 96 |
| **7. Exp001 Sequential PPO (Escalate Trap)** | $-1.10 \pm 0.00$ | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 1.00 | Act 19: 60 |
| **8. EXP002 Sequential PPO (Best Checkpoint)** | **$-0.09 \pm 1.02$** | **100.0%** | **0.0%** | **0.0%** | **100.0%** | **0.0%** | **1.23** | **Act 7: 74 (100%)** |
| **9. EXP002 Sequential PPO (Final Iteration)** | **$-0.09 \pm 1.02$** | **100.0%** | **0.0%** | **0.0%** | **100.0%** | **0.0%** | **1.23** | **Act 7: 74 (100%)** |

---

## 4. Training Progression & Policy Shift in Exp 002

| Iteration | Step Reward | Rollout Ep Ret | Containment | Clearance | Breach | Policy Loss | Value Loss | Entropy | Dominant Rollout Actions |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **00 (Init)** | — | $-1.075$ | 0.0% | 0.0% | 40.0% | — | — | — | Act 17: 29 (100%) |
| **01** | $-0.670$ | $-1.100$ | 2.6% | 0.0% | 43.6% | $-0.0117$ | 2.9299 | 0.473 | Act 17 (89.1%), Act 7 (1.6%) |
| **05** | $-0.783$ | $-1.429$ | 0.0% | 0.0% | 65.7% | $-0.0090$ | 0.4005 | 0.313 | Act 17 (93.8%) |
| **10** | $-0.661$ | $-1.178$ | 2.8% | 0.0% | 55.6% | $-0.0118$ | 0.4472 | 0.248 | Act 17 (90.6%) |
| **15** | $-0.650$ | $-1.015$ | 2.4% | 2.4% | 41.5% | $-0.0116$ | 0.7869 | 0.658 | Act 17 (82.8%), Act 7 (9.4%) |
| **18** | $-0.716$ | $-1.115$ | 9.8% | 0.0% | 43.9% | $-0.0071$ | 1.2353 | 1.176 | Act 17 (75.0%), Act 7 (6.2%) |
| **19** | $-0.452$ | $-0.702$ | 21.9% | 0.0% | 34.1% | $-0.0334$ | 2.4973 | 1.218 | Act 17 (60.9%), Act 7 (18.8%) |
| **20** | $-0.216$ | $-0.331$ | 33.3% | 0.0% | 19.0% | $-0.0229$ | 2.7940 | 1.292 | Act 7: 21 (32.8%), Act 17: 35 (54.7%) |
| **22** | $-0.325$ | $-0.498$ | 42.2% | 0.0% | 20.0% | $-0.0659$ | 2.0312 | 1.151 | Act 7: 32 (50.0%), Act 17: 22 (34.4%) |
| **24** | $+0.027$ | $+0.037$ | 58.7% | 0.0% | 6.5% | $-0.0182$ | 2.5345 | 0.882 | Act 7: 46 (71.9%), Act 17: 11 (17.2%) |
| **28** | $-0.031$ | $-0.038$ | 57.7% | 0.0% | 1.9% | $-0.0164$ | 1.3085 | 0.510 | Act 7: 57 (89.1%), Act 17: 3 (4.7%) |
| **30** | $-0.153$ | $-0.200$ | 53.1% | 0.0% | 4.1% | $-0.0069$ | 2.1138 | 0.536 | Act 7: 55 (85.9%), Act 11: 4 (6.2%) |
| **35** | $-0.133$ | $-0.157$ | 61.4% | 0.0% | 6.8% | $-0.0073$ | 1.6857 | 0.445 | Act 7: 58 (90.6%) |
| **40** | $-0.098$ | $-0.140$ | 55.6% | 0.0% | 4.4% | $-0.0148$ | 1.0523 | 0.608 | Act 7: 56 (87.5%) |
| **50** | $-0.070$ | $-0.092$ | 55.1% | 0.0% | 2.0% | $-0.0073$ | 0.7710 | 0.932 | Act 7: 59 (92.2%) |

---

## 5. Analytical Root-Cause: Why PPO Replaced Action 19 with Action 7

### 5.1 Why Escalation Was Abandoned
In Exp 001, Action 19 offered an immediate reward of $-0.10$.
In Exp 002, with `escalation_cost = 1.0`:
$$\text{Reward}(\text{escalate}) = -0.10 - 1.00 = \mathbf{-1.10}$$
Escalation was no longer an attractive, zero-risk shelter. Its return ($-1.10$) was roughly on par with a false disruption penalty ($-1.10$) and inferior to benign unresolved triage ($-0.60$). Consequently, during iterations 18–22, when exploration entropy peaked at $1.292$, Action 19 was sampled but failed to generate positive advantages over emerging containment actions.

### 5.2 The Structural Supremacy of Action 7 (`isolate_host`)
In the simulation environment, `isolate_host` has unique persistence dynamics:
1. **Persistent Network Severance:** When Action 7 is executed, the affected host is added to `self.simulated_state["isolated_hosts"]`.
2. **Universal Threat Blockade:**
   - In `scen_c2_exfil`: Even if applied prematurely at Stage 0 (incurring a $-1.60$ disruption), Stage 1 requires network egress $\to$ automatically triggers `contained_by_prior_isolation` and $+2.0$ containment reward!
   - In `scen_privesc_persist`: Action 7 is effective at Stage 1 $\to$ contains persistence.
   - In `scen_credential_lateral`: Action 7 is effective at Stage 1 $\to$ contains lateral DB exfiltration.
   - In `scen_network_rce_tunnel`: Action 7 is effective at Stage 1 $\to$ contains reverse stream exfiltration.
3. **Zero Breaches Across All Attacks:** By executing Action 7 indiscriminately, the policy **completely prevents attack compromise across 100% of malicious scenarios**, entirely evading the catastrophic breach penalty of $-2.0$ to $-3.80$.

### 5.3 Return Economics of "Always Isolate Host"
Evaluating the expected return of playing Action 7 unconditionally:
- **Malicious Scenarios (4/7):** Mean return = **$+0.79375$**
- **Benign Scenarios (3/7):** Triggers `disruptive_false_positive` on Stage 0 $\to$ Return = **$-1.1000$**
- **Weighted Expected Return:**
  $$\mathbb{E}[R] = \frac{4}{7}(+0.79375) + \frac{3}{7}(-1.1000) = \frac{3.175 - 3.300}{7} = \mathbf{-0.018}$$
  (On the 60-episode benchmark distribution with 32 malicious and 28 benign episodes, this measured **$-0.090$**).

Comparing the returns of all available fixed strategies under `escalation_cost = 1.0`:
- **Always Isolate Host (Act 7):** **$-0.09$**
- **Uniform Random:** $-0.68$
- **Always Monitor (Act 0):** $-0.81$
- **Always Escalate (Act 19):** $-1.10$
- **Frozen Pretrained PPO (Act 17):** $-1.23$
- **Always Kill Process (Act 8):** $-1.37$

**Conclusion:** Action 7 (`isolate_host`) is the **new global optimum for any policy that cannot distinguish benign from malicious telemetry**. Because discriminating between 7 scenarios across 20 actions requires learning subtle feature correlates, coarse reinforcement learning converged to the single blunt-force containment action that eliminates all breach risk at the cost of operational disruption.

---

## 6. Research Integrity Assessment

1. **Did PPO learn active containment?**
   **Yes, but as a degenerate blunt instrument.** The policy achieved 100% malicious threat containment, completely resolving the passive inaction and breach catastrophe of Exp 001. However, it did not learn selective containment; it learned "contain everything immediately."
2. **Did PPO learn benign clearance?**
   **No.** Benign clearance remained at 0.0%, and false disruptions reached 100%. The policy never learned to monitor or deep-log benign administrative operations.
3. **Did episode return improve?**
   Yes, mean episodic return improved from $-1.10$ (Exp 001 under cost=1) to $-0.09$ (Exp 002). However, this improvement must not be heralded as a successful autonomous SOC agent; it represents a classic "scorched-earth" policy collapse.
4. **Validation of the Diagnostic Prediction:**
   Section 5 of `results/soc_env/exp002_reward_diagnostics.md` explicitly warned:
   > *"Penalizing escalation alone does not guarantee that unconstrained PPO will discover multi-stage containment. The agent may simply collapse to the next lowest-penalty passive or coarse action."*
   Experiment 002 empirically confirmed this prediction.

---

## 7. Action Plan for Experiment 003

To move beyond blunt-force single-action collapse toward true discriminatory incident response, Experiment 003 must introduce structural constraints rather than scalar reward tweaks alone:

1. **Operational Host Isolation Precondition (Action Gating):**
   In enterprise IT environments, an analyst or EDR system is prohibited from isolating production hosts or database bastions on Stage 0 without prior verification. Restricting `isolate_host` to Stage $\ge 1$ or requiring a prerequisite investigative action will eliminate the "isolate-everything-on-step-0" shortcut.
2. **Adaptive Teacher-Guided Preservation (ATGP) or Imitation Warm-Start:**
   The Observable-Telemetry Rule-Based Policy achieves **$+0.42$ return, $75\%$ containment, and $42.9\%$ clearance** using simple observable heuristics. Seeding the sequential PPO policy with behavioral cloning or ATGP loss against this heuristic will provide the initial state-conditional reflexes needed to separate benign from malicious telemetry.
3. **Escalation Gating:**
   Permit escalation only when epistemic uncertainty is genuinely high or following an initial investigative step.