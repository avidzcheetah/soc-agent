# Audit Report: SOC Baseline Evaluation Validity & Information Leakage

**Date:** 2026-10-12  
**Target:** SOC Simulation Environment (`soc_env`) Baseline Evaluation Suite  
**Evaluated Artifacts:**
- `scripts/evaluate_soc_baselines.py`
- `src/ppo/trainer_sequential.py`
- `src/environment_sequential.py`
- `results/soc_env/baseline_evaluation_report.md`
- `results/soc_env/baseline_evaluation_results.json`

---

## 1. Executive Summary & Verdict

| Audit Dimension | Finding | Verdict |
| :--- | :--- | :---: |
| **Rule-Based SOC Policy** | Directly accesses ground-truth hidden labels (`is_benign`, `scenario_id`, `stage_id`). | **FAILED (Severe Information Leakage)** |
| **Other Baselines (Random, Fixed, Pretrained PPO)** | Strictly evaluate within valid environment rules and standard SecBERT embeddings. | **PASSED (Valid)** |
| **Pretrained Checkpoint Loading** | Correct weights loaded with `.eval()` and deterministic argmax action selection. | **PASSED (Valid)** |
| **Metric Formulas & Denominators** | Correctly separates malicious containment from benign clearance with independent denominators. | **PASSED (Trustworthy)** |
| **Reproducibility** | Seeded resets guarantee identical scenario sequences and reproducible random trajectories. | **PASSED (Reproducible)** |
| **Curriculum Scope** | 3 scenario templates repeated across 60 episodes; lack of attack variety inflates "Always Kill Process". | **RESTRICTED (Curriculum Limitation)** |

**Key Takeaway:** The reported performance of the Rule-Based SOC Policy (`+1.90` return, `100.0%` containment, `100.0%` clearance) is **invalid** because it acts as an oracle cheat using hidden ground-truth simulation variables rather than observable telemetry. In contrast, the remaining baselines, metric calculations, and evaluation harnesses are methodologically sound.

---

## 2. In-Depth Audit of the Rule-Based Policy

### 2.1 Fields Inspected by the Policy
In `src/ppo/trainer_sequential.py` (lines 451–477), `rule_based_policy` is implemented as:
```python
def rule_based_policy(obs: torch.Tensor, info: Dict[str, Any]) -> int:
    if info.get("is_benign", False):
        return 0  # monitor

    scen_id = info.get("scenario_id", "")
    stage_id = info.get("stage_id", 0)

    if "c2" in scen_id or "exfil" in scen_id:
        if stage_id >= 1:
            return 4  # block_dest_ip
        return 8      # kill_process (enumeration stage)

    if "privesc" in scen_id or "persist" in scen_id:
        if stage_id >= 1:
            return 13 # remove_persistence
        return 12     # disable_account

    return 8  # fallback: kill_process
```

### 2.2 Distinguishing Legitimate Telemetry from Ground-Truth Information
- **Legitimate Observable Telemetry:**
  In a real SOC, an incident response agent or rule engine observes:
  - Formatted alert text and embeddings (`obs`).
  - SIEM rule metadata (`rule_id`, `rule_description`, `alert_level`).
  - Host and process telemetry (`host`, `process_name`, `cmdline`, `pid`, `user`).
  - Network and behavioral indicators (`dest_ip`, `mitre_tactics`, `mitre_techniques`).
- **Hidden Ground-Truth State:**
  - `is_benign`: The underlying ground-truth label of whether an alert is malicious or benign. In real operations, distinguishing benign from malicious alerts is the core triage task.
  - `scenario_id`: The simulator's internal scenario template ID (`"scen_c2_exfil"`, `"scen_privesc_persist"`, `"scen_benign_backup"`).
  - `stage_id`: The internal index of the multi-stage progression graph.

### 2.3 Mechanism of 100% Containment and 100% Clearance
1. **Benign Clearance (100.0%):**
   The policy inspects `info.get("is_benign", False)`. Whenever `is_benign == True`, it immediately selects Action 0 (`monitor_only`). In `scen_benign_backup`, Action 0 is defined as effective (`effective_actions=[0, 1]`), causing the environment to immediately mark the incident as `cleared=True` on step 1 with reward `+1.90` and zero false disruption penalty.
2. **Malicious Containment (100.0%):**
   - For `scen_c2_exfil` at `stage_id == 0`, it chooses Action 8 (`kill_process`), which matches `stage_disc.effective_actions=[8]`.
   - For `scen_privesc_persist` at `stage_id == 0`, it chooses Action 12 (`disable_account`), which matches `stage_sudo.effective_actions=[11, 12, 8]`.
   - Both malicious templates are contained on step 1 without advancing to later stages, achieving 100% containment, 0% breach rate, and net return `+1.90`.

**Conclusion on Rule-Based Policy:** The rule-based policy suffers from **complete information leakage**. It does not perform pattern matching on observable telemetry; it is an oracle lookup table. Its benchmark score must be discounted as an upper-bound oracle rather than an authentic heuristic baseline.

---

## 3. Audit of Other Baseline Policies

### 3.1 Observation Equivalence & Rule Compliance
- All policies are evaluated using the identical function `evaluate_policy_on_episodes()` in `scripts/evaluate_soc_baselines.py`.
- **Uniform Random Policy:** Ignores `info` and `obs`. Returns pseudo-random actions in `[0, 19]`. Evaluated fairly under standard environment transitions.
- **Fixed-Action Policies (`Always Monitor`, `Always Kill Process`, `Always Escalate`):** Ignore `info` and `obs`. Return static actions (0, 8, 19). Fully compliant.
- **Pretrained PPO Agent (`Exp 4D ATGP`):** Evaluated strictly via `agent.select_action(obs, deterministic=True)[0]`. It does not inspect `info` and relies exclusively on the 768-dimensional SecBERT observation vector `obs`.

### 3.2 Pretrained Checkpoint Loading & State Representation
- Checkpoint `models/ppo_exp4d_atgp/best_ppo_policy.pt` is loaded into `PPOAgent(state_dim=768, action_dim=20, device="cpu")`.
- State encoder is initialized with `models/secbert_finetuned`.
- Both actor and critic weights are loaded and set to `.eval()`.
- Deterministic inference uses `torch.argmax(logits)`, ensuring reproducible non-stochastic action selection.
- **Explanation of Pretrained PPO Collapse:**
  The pretrained PPO actor predicted Action 17 (`snapshot_forensics`) on 100% of steps (115/115 steps) across all 60 episodes. This is not a software bug; it is an authentic domain-shift failure:
  - Phase 2 PPO was trained on contextual-bandit text samples from threat reports (CISSM/rcATT), where forensic investigation was heavily rewarded on ambiguous initial descriptions.
  - In the sequential environment, taking purely investigative actions gathers intelligence but never contains threats or clears incidents. The attacks progress to full breach, yielding `-1.37` mean return.

---

## 4. Metrics & Evaluation Methodology Audit

### 4.1 Denominator and Metric Integrity
- **Malicious Containment Rate:** $\frac{\text{malicious\_contained}}{\text{total\_malicious}} = \frac{17}{40} = 42.5\%$ for random policy. Denominator is strictly malicious episodes ($N=40$).
- **Benign Clearance Rate:** $\frac{\text{benign\_cleared}}{\text{total\_benign}} = \frac{5}{20} = 25.0\%$ for random policy. Denominator is strictly benign episodes ($N=20$).
- **Malicious Breach Rate:** $\frac{\text{malicious\_breached}}{\text{total\_malicious}} = \frac{20}{40} = 50.0\%$ for random policy.
- **False Disruption Rate:** $\frac{\text{benign\_disrupted}}{\text{total\_benign}} = \frac{2}{20} = 10.0\%$ for random policy.
- **Escalation Rate:** $\frac{\text{escalated\_episodes}}{\text{total\_episodes}} = \frac{3}{60} = 5.0\%$ for random policy.
- **Episodic Return:** $\sum_{t=0}^T r_t$ correctly accumulated step-by-step per episode.
- **Episode Length:** Step count correctly tracked until terminal condition or truncation.

### 4.2 Outcome Mapping
The environment states and transitions properly distinguish between:
- `contained`: Threat neutralized by effective containment or persistent prior isolation/blocking.
- `cleared`: Benign activity validated and closed without business disruption.
- `breach_compromised`: Malicious attack progressed past final stage without containment.
- `disruptive_false_positive`: Heavy containment action taken on benign maintenance.
- `benign_unresolved`: Ineffective actions taken on benign alert until triage limit reached.
- `escalated`: Human tier-2 escalation invoked.

---

## 5. Reproducibility & Environment State Integrity

1. **Deterministic Scenario Sequences:**
   - Calling `env.reset(seed=s)` with $s \in [1000, 1059]$ generates the identical sequence of scenarios for all policies:
     - `scen_privesc_persist`: 23 episodes (38.3%)
     - `scen_c2_exfil`: 17 episodes (28.3%)
     - `scen_benign_backup`: 20 episodes (33.3%)
     - Total: 40 malicious, 20 benign.
2. **Random Policy Reproducibility:**
   - `make_random_policy(seed=42)` uses an isolated `np.random.RandomState(42)`. Independent verification confirmed identical returns and outcome distributions across runs.
3. **Absence of State Cross-Talk:**
   - `SequentialSOCEnvironment.reset()` cleans all state variables (`action_history`, `simulated_state`, `step_count`, termination flags).
   - SecBERT embedding cache keys on alert text hash, introducing no policy or state bias.

---

## 6. Identified Deficiencies & Curriculum Limitations

1. **Curriculum Vulnerability to Single-Action Bias:**
   Because 2 out of the 3 templates in `get_default_scenarios()` involve process execution, the heuristic `Always Kill Process` achieves an artificially high score of `+0.90` and 100% malicious containment. It achieves this despite suffering a 100% false-disruption rate on benign incidents.
2. **Information Leakage in `rule_based_policy`:**
   Directly inspecting `info["is_benign"]` and `info["scenario_id"]` makes the rule-based policy an unrealistic baseline.

---

## 7. Recommended Minimum Corrections Before Sequential PPO Training

To ensure rigorous comparisons before sequential PPO fine-tuning begins, the following minimal corrections are recommended:

1. **Replace Oracle Rule-Based Policy with Telemetry-Based Heuristic:**
   - Refactor `rule_based_policy` so it **does not inspect** `info["is_benign"]`, `info["scenario_id"]`, or `info["stage_id"]`.
   - Instead, have the rule-based heuristic parse observable alert properties (e.g. from an `observable_telemetry` dictionary exposed in `info` or raw alert text):
     - Low alert level ($\le 3$) or backup/rsync commands $\rightarrow$ Action 0 (`monitor_only`).
     - Network exfiltration rules or curl commands $\rightarrow$ Action 4 (`block_dest_ip`).
     - Sudo escalation or crontab modifications $\rightarrow$ Action 12 (`disable_account`) or Action 13 (`remove_persistence`).
     - Default suspicious process $\rightarrow$ Action 8 (`kill_process`).
2. **Sanitize Agent Evaluation Interface:**
   - Standardize policy evaluation so agents receive only `obs` (for neural agents) or `obs` + `info["observable"]` (for symbolic rule heuristics), ensuring hidden ground-truth simulation variables cannot be read by any policy.
3. **Expand Scenario Library for Benchmark Balance:**
   - Add at least 2 additional scenario templates before PPO training:
     - A network-only / credential-theft scenario where `kill_process` fails.
     - A benign incident with higher alert severity (e.g. administrative script) to test false-positive resistance.
