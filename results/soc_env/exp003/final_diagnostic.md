# SOC Simulation Environment Experiment 003: Final Diagnostic Report
**Date:** October 22, 2026  
**Project:** `/home/avidz/Research/soc-agent`  
**Status:** Complete  

---

## 1. Executive Summary

This diagnostic investigates the empirical behavior of the telemetry-augmented sequential PPO agent in Experiment 003. Specifically, it resolves:
1. Why Seeds 42 and 43 collapsed to Action 7 (`isolate_host`) while Seed 44 branched into Actions 7 and 11.
2. Why Seed 44 still caused $42.9\%$ false disruptions on benign incidents in initial Exp003.
3. Why benign clearance was $0.0\%$ in initial Exp003 across all seeds.
4. Whether the 34-dimensional observable telemetry vector genuinely influences policy action selection.
5. Whether outcome bookkeeping, simulation step rewards, and episode metrics are mathematically consistent across all evaluated policies.
6. What evaluation and statistical reporting considerations remain regarding scenario archetype representation.

---

## 2. Diagnostic Investigation & Findings

### 2.1 Why Seeds 42 and 43 Collapsed to `isolate_host`
In initial Exp003, Seeds 42 and 43 terminated training prematurely with a static return of $-0.370$ and 100% selection of Action 7 (`isolate_host`).

#### Root Cause A: Trainer Early-Stopping Defect
Inspection of [src/ppo/trainer_sequential.py](file:///home/avidz/Research/soc-agent/src/ppo/trainer_sequential.py#L47-L69) and [scripts/ppo/train_augmented_sequential.py](file:///home/avidz/Research/soc-agent/scripts/ppo/train_augmented_sequential.py) reveals:
- `SequentialPPOTrainer` defines a default `early_stopping_patience = 5`.
- Periodic validation evaluation occurs every `eval_interval = 5` iterations on seeds 2000–2019.
- Warm-starting from Exp002 set the initial policy validation return to $-0.370$.
- In iterations 5, 10, 15, 20, and 25, deterministic validation evaluations produced mean returns of $-0.370$.
- Because the condition for resetting the patience counter is strictly `if mean_ret > self.best_eval_return:`, five consecutive evaluations with `mean_ret == -0.370` incremented `early_stopping_counter` from 0 to 5.
- At iteration 25, `trainer.train()` triggered early stopping:
  ```text
  Early stopping triggered at iteration 25 (patience=5).
  [OK] Training complete for seed 42 in 2.14s.
  ```
- **Consequence:** Seeds 42 and 43 were terminated after only 1,600 environment steps (25 iterations), never reaching the training horizon (iterations 35–50) where policy updates overcame the warm-started logits.

#### Root Cause B: Initial Logit Magnitude of the Warm-Start Policy
In the Exp002 checkpoint (`models/ppo_sequential_exp002/best_sequential_policy.pt`), Action 7 logits ranged from $+3.5$ to $+4.8$, while Action 0 (`monitor`) logits were negative ($-2.5$ to $-3.2$). The initial probability of Action 7 exceeded $76\%$, while Action 0 was $<0.5\%$. Under low entropy regularization ($c_2 = 0.01$), stochastic sampling during 64-step rollouts rarely sampled Action 0, preventing gradient updates from receiving positive feedback from benign clearance.

#### Why Seed 44 Escaped
In Seed 44, stochastic exploration during rollout collection in iteration 25 happened to sample Action 11 (`reset_credentials`) on a credential scenario, achieving an evaluation return of $-0.130$ ($> -0.370$). This reset `early_stopping_counter` to 0, allowing Seed 44 to train for all 50 iterations and discover the +0.145 validation return at iteration 40.

---

### 2.2 Why Seed 44 Still Disrupted Benign Incidents in Initial Exp003
In the initial Exp003 benchmark evaluation, Seed 44 produced $42.9\%$ false disruptions (12 of 28 benign episodes).

#### Telemetry Confounding on the Backup Archetype
Inspection of the per-scenario breakdown in [results/soc_env/exp003/comparative_evaluation_results.json](file:///home/avidz/Research/soc-agent/results/soc_env/exp003/comparative_evaluation_results.json) demonstrates:
- Seed 44 achieved **0.0% disruption** on `scen_benign_devops_tooling` (8 episodes).
- Seed 44 achieved **0.0% disruption** on `scen_benign_security_audit` (8 episodes).
- Seed 44 achieved **100.0% disruption** on `scen_benign_backup` (12 episodes).

On `scen_benign_backup`, the executable binary is `/usr/bin/rsync`. In the malicious scenario library, `rsync` is also the primary execution vehicle for active data exfiltration in `scen_c2_exfil` (Stage 2) and `scen_credential_lateral` (Stage 1). Because Action 7 (`isolate_host`) is the effective containment action for malicious network exfiltration, the policy associated `proc_rsync = 1.0` with host isolation. On `server-backup-02`, host isolation is defined as disruptive, triggering the $-1.00$ disruption penalty.

---

### 2.3 Why Benign Clearance Was Zero in Initial Exp003
In the simulation environment, achieving `outcome = "correctly_cleared_benign"` strictly requires selecting an action in `stage.effective_actions`, which is `[0, 1]` (`0: monitor`, `1: enable_deep_logging`) for all benign scenarios.

In initial Exp003:
1. Logit inspection revealed that the probability of Action 0 across all scenarios was between **$0.004$ and $0.006$** ($0.4\% - 0.6\%$).
2. The agent never sampled Action 0 during deterministic evaluation.
3. Instead, on `scen_benign_devops_tooling` and `scen_benign_security_audit`, the policy selected Action 11 (`reset_credentials`).
4. Action 11 is non-disruptive on those two hosts (not in `disruptive_actions = [8, 7, 12, 4]`).
5. However, Action 11 is not effective either. The environment classified the outcome as `benign_unresolved` with a step return of $-0.60$ (step cost $-0.10$ + ineffective penalty $-0.50$).
6. Because $-0.60$ is vastly superior to the disruptive penalty of $-1.10$, PPO converged to Action 11 as a "safe non-disruptive exit", achieving zero false disruption on those two scenarios but failing to earn the $+1.40$ clearance reward.

---

### 2.4 Telemetry Influence on Action Selection: Empirical Verification
To verify whether the 34-dimensional telemetry features demonstrably drive policy decisions, we measured the softmax output distribution of `AugmentedPPOAgent` across distinct scenario archetypes:

| Scenario Archetype | Key Observable Telemetry Bits | Action 7 (`isolate_host`) Prob | Action 11 (`reset_credentials`) Prob | Action 0/1 (`monitor`/`logging`) Prob | Dominant Action |
|:---|:---|:---:|:---:|:---:|:---:|
| `scen_privesc_persist` | `proc_sudo=1`, `uid=0`, `tactic_priv_esc=1` | $11.8\%$ | **72.7%** | $0.4\%$ | **11** |
| `scen_credential_lateral`| `user=devops_lead`, `tactic_lat_mvmt=1` | $20.2\%$ | **64.3%** | $0.5\%$ | **11** |
| `scen_network_rce_tunnel`| `proc_node=1`, `is_net_cmdline=1`, `tactic_c2=1` | **57.4%** | $32.1\%$ | $0.5\%$ | **7** |
| `scen_benign_devops` | `cwd=/tmp/ci_build`, `proc_python=1`, `user=ci_service` | $41.3\%$ | **45.0%** | $0.6\%$ | **11** |
| `scen_benign_security_audit`| `proc_nmap=1`, `user=sec_audit`, `tactic_disc=1` | $30.0\%$ | **54.5%** | $0.6\%$ | **11** |
| `scen_benign_backup` | `proc_rsync=1`, `alert_level=2`, `has_mitre=0` | **47.3%** | $39.8\%$ | $0.6\%$ | **7** |

**Conclusion:** Telemetry features demonstrably and systematically shift action probabilities. In Exp002 (without telemetry), Action 7 probability was uniform ($>76\%$) across all scenarios. In Exp003, observing credential or privilege tokens shifts probability to Action 11 by over $+60$ percentage points, while network and tunnel indicators preserve Action 7 dominance.

---

### 2.5 Consistency Verification of Outcomes and Rewards
We conducted an audit of [src/environment_sequential.py](file:///home/avidz/Research/soc-agent/src/environment_sequential.py#L765-L865):
1. **Reward Parity:** All policies (heuristic, random, fixed, Exp001, Exp002, Exp003) execute through the identical `step()` method.
2. **Transition Costs:** Every step incurs `step_cost = -0.10`.
3. **Containment:** Malicious effective actions earn $+2.00$ net $+1.90$.
4. **Clearance:** Benign effective actions earn $+1.50$ net $+1.40$.
5. **Disruption:** Disruptive actions on benign hosts earn $-1.00$ net $-1.10$.
6. **Breach:** Uncontained final stage malicious incidents earn $-2.00$ net $-2.10$.
7. **Escalation:** Action 19 incurs `escalation_cost` ($1.0$ in primary, $0.0$ in sensitivity).

No policy receives privileged reward treatment or alternative transition dynamics.

---

### 2.6 Statistical and Benchmark Considerations
The 60 benchmark episodes are generated by seeding 60 random reset calls (seeds 1000–1059) over the 7 scenario archetypes:
- 32 malicious episodes (spanning 4 malicious archetypes: C2 Exfiltration, Privilege Escalation, Lateral Dump, Network Tunnel).
- 28 benign episodes (spanning 3 benign archetypes: Backup Maintenance, DevOps Build, Security Audit Scan).

While each episode evaluates an independent seed draw with stochastic initial parameters, they represent repeated instances of 7 scenario classes. Therefore, statistical comparisons should report paired differences on identical episode seeds, but acknowledge that generalizability is bounded by the scenario archetype diversity.

---

### 2.7 Resolution of Rule-Based Baseline False-Disruption Rate Discrepancy
An earlier summary draft incorrectly reported that the Observable Rule-Based Policy achieved a "0.0% Benign False-Disruption Rate". A comprehensive audit of [results/soc_env/exp003/final_comparison_results.json](file:///home/avidz/Research/soc-agent/results/soc_env/exp003/final_comparison_results.json) and [src/environment_sequential.py](file:///home/avidz/Research/soc-agent/src/environment_sequential.py) resolves this discrepancy:
1. **Actual Empirical Outcome:** In the 28 benign benchmark episodes, the Observable Rule-Based Policy produced:
   - **12 Correct Clearances (42.9%):** On `scen_benign_backup` (12 episodes), the rule checked `if "backup" in host: return 0 (monitor)`, which is non-disruptive and effective, correctly earning $+1.40$.
   - **16 Disruptive False Positives (57.1%):**
     - On `scen_benign_devops_tooling` (8 episodes), the rule encountered Python process execution and triggered Action 8 (`kill_process`). The host `devops-ci-runner` explicitly specifies `disruptive_actions = [8, 7, 12, 4]`. Killing the CI process disrupted active production builds, incurring a $-1.00$ disruption penalty.
     - On `scen_benign_security_audit` (8 episodes), the rule detected an `nmap` port scan and triggered Action 8 (`kill_process`). Host `audit-scanner-01` likewise marks Action 8 as disruptive, triggering the $-1.00$ disruption penalty.
2. **Root Cause of the Discrepancy:** The earlier "0.0%" text was a human transcription oversight that conflated "unresolved episodes" with non-disruptive actions or misread the initial table where disruptive actions were only counted on host isolation (Action 7). The environment's ground-truth definition of disruption encompasses all host-specific `disruptive_actions`. Therefore, the true, verified false-disruption rate of the rule-based policy is **57.1%** (16/28), and the mean return is **+0.420**, exactly as logged in the benchmark results.


---

## 3. Targeted Scientific Solution

Based on this diagnostic, the optimal minimal improvement requires **zero reward modifications**, **zero scenario changes**, and **zero label leakage**:
1. **Algorithmic Defect Correction:** Disable premature early stopping (`early_stopping_patience = 0`) to allow all seeds to train through the planned 50 iterations.
2. **Restoring Action Exploration:** Increase the PPO entropy regularization coefficient from $c_2 = 0.01$ to $c_2 = 0.03$, enabling the policy to sample low-probability actions (Actions 0 and 1) during rollouts and discover the $+1.40$ benign clearance reward.

The implementation and empirical verification of this targeted refinement are reported in [results/soc_env/exp003/final_comparison_report.md](file:///home/avidz/Research/soc-agent/results/soc_env/exp003/final_comparison_report.md).
