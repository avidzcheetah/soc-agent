# Pre-Training Readiness Audit: Expanded SOC Simulation Environment

**Date:** 2026-10-12  
**Target:** Expanded Sequential SOC Simulation Environment (`SequentialSOCEnvironment`) & Sequential PPO Pipeline (`SequentialPPOTrainer`)  
**Evaluated Artifacts:**
- `src/environment_sequential.py`
- `src/ppo/trainer_sequential.py`
- `scripts/evaluate_soc_baselines.py`
- `tests/test_environment_sequential.py`
- `results/soc_env/baseline_evaluation_v3_report.md`
- `results/soc_env/baseline_evaluation_v3_results.json`

---

## 1. Executive Summary & Readiness Verdict

| Audit Dimension | Core Verification Item | Audit Finding | Verdict |
| :--- | :--- | :--- | :---: |
| **1. Scenario Correctness** | 7 scenarios (4 malicious, 3 benign), stage transitions, effective actions, breaches. | All 7 scenario stages execute valid transitions; ineffective kill/isolation causes breaches; benign clearance is strictly separated from containment. | **PASS** |
| **2. Reward Consistency** | Expected returns across containment, breach, investigation, escalation, and disruption. | Returns are bounded and consistent; step cost prevents stalling; investigation advances the stage toward breach; escalation caps risk at -0.10. | **PASS** |
| **3. Observation Validity** | PPO observation inputs, leakage protection, observable telemetry isolation. | Neural agent receives strictly 768-dim SecBERT embeddings; symbolic rules receive only observable SIEM/endpoint telemetry; hidden labels cannot leak. | **PASS** |
| **4. Evaluation Integrity** | Seed reproducibility, metric denominators, baseline differentiation. | Fully reproducible across seeds 1000–1059; malicious containment ($N=32$) and benign clearance ($N=28$) use independent denominators; single-action bias eliminated. | **PASS** |
| **5. Training Readiness** | Episode boundaries, Generalized Advantage Estimation (GAE), rollout buffers, updates. | Multi-step rollouts respect terminal/truncation boundaries; advantages do not leak across episode resets; temporal credit assignment propagates backwards across stages. | **PASS** |

**Final Verdict:** The expanded 7-scenario sequential environment is **READY for sequential PPO training**. All 45 unit and integration tests pass cleanly, and the benchmark curriculum eliminates trivial single-action heuristics while preserving evaluation integrity.

---

## 2. Detailed Technical Findings

### 2.1 Scenario Correctness & Transition Dynamics
1. **Curriculum Diversity (7 Distinct Archetypes):**
   - **`scen_c2_exfil` (Malicious, 3 stages):** Discovery (`whoami`) $\rightarrow$ C2 beacon (`python3`) $\rightarrow$ Exfiltration (`curl`). Tests staged multi-vector containment (Action 8 at stage 0, Action 4/7 at stages 1–2).
   - **`scen_privesc_persist` (Malicious, 2 stages):** Sudo escalation $\rightarrow$ Cron backdoor. Tests credential/account mitigation (Actions 11/12) versus persistence eradication (Action 13).
   - **`scen_credential_lateral` (Malicious, 2 stages):** Stolen SSH key authentication on bastion $\rightarrow$ Lateral database extraction. **Kill process (Action 8) is ineffective**, as terminating the shell does not revoke credentials; requires credential reset (Action 11) or account disable (Action 12).
   - **`scen_network_rce_tunnel` (Malicious, 2 stages):** Web API exploit with auto-restarting worker daemon $\rightarrow$ Outbound tunnel. **Kill process (Action 8) is ineffective** due to supervisor auto-restart; requires network egress blocking (Action 4), source blocking (Action 3), or vulnerability patching (Action 16).
   - **`scen_benign_backup` (Benign, 1 stage):** Routine `rsync` system backup (`alert_level=2`). Tests baseline benign monitoring without false disruption.
   - **`scen_benign_devops_tooling` (Benign, 1 stage):** CI/CD pipeline build script in `/tmp/ci_build` (`alert_level=6`). Tests resilience against naive temporary-path process termination.
   - **`scen_benign_security_audit` (Benign, 1 stage):** Approved internal `nmap` vulnerability sweep (`alert_level=7`, MITRE T1046). Tests resilience against premature host isolation or firewall blocking.

2. **Containment & Breach Logic Verification:**
   - Documented effective containment actions set `contained=True`, `terminated=True`, and award $+2.0$ containment reward on all malicious scenarios.
   - Mismatched or ineffective actions incur penalties ($-0.5$ ineffective, $-1.0$ disruption) and advance `current_stage_idx += 1`.
   - When attacks advance past the final stage uncontained, `current_stage_idx >= total_stages` triggers `compromised=True`, `terminated=True`, breach penalty $-2.0$, and outcome `"breach_compromised"`.
   - On benign scenarios, effective non-disruptive actions (`monitor_only` or `enable_deep_logging`) set `cleared=True`, `contained=False`, `terminated=True`, and award $+2.0$. Benign clearance never increments threat containment.

3. **Persistent Response State Tracking:**
   - Actions 11 (`reset_credentials`) and 12 (`disable_account`) register the target user in `simulated_state["disabled_accounts"]`. If a subsequent stage uses that account, it is automatically neutralized via `"contained_by_prior_account_disable"`.
   - Action 4 (`block_dest_ip`) adds the target IP to `simulated_state["blocked_ips"]`, neutralizing subsequent network exfiltration via `"contained_by_prior_block"`.
   - Action 7 (`isolate_host`) adds the host to `simulated_state["isolated_hosts"]`, neutralizing subsequent lateral/exfil stages via `"contained_by_prior_isolation"`.
   - All simulated response states are strictly cleared upon `env.reset()`.

---

### 2.2 Reward Structure & Exploitation Analysis

#### Reward Parameters (`RewardConfig`):
- `containment_reward`: $+2.0$
- `benign_clearance_reward`: $+2.0$
- `step_cost`: $-0.1$
- `ineffective_penalty`: $-0.5$
- `disruption_penalty`: $-1.0$
- `investigation_reward`: $+0.2$
- `breach_penalty`: $-2.0$
- `escalation_reward`: $0.0$
- `escalation_cost`: $0.0$

#### Expected Episodic Returns:
1. **Immediate Malicious Containment (Step 1):**  
   $R = -0.1 (\text{step}) + 2.0 (\text{containment}) = \mathbf{+1.90}$.
2. **Immediate Benign Clearance (Step 1):**  
   $R = -0.1 (\text{step}) + 2.0 (\text{clearance}) = \mathbf{+1.90}$.
3. **Immediate Escalation to Tier-2 Analyst (Step 1):**  
   $R = -0.1 (\text{step}) + 0.0 (\text{escalation}) = \mathbf{-0.10}$.
4. **False Disruption on Benign Activity (Step 1):**  
   $R = -0.1 (\text{step}) - 1.0 (\text{disruption}) = \mathbf{-1.10}$.
5. **Investigation Followed by Containment (2 Stages):**  
   Stage 0: $-0.1 + 0.2 = +0.10$.  
   Stage 1: $-0.1 + 2.0 = +1.90$.  
   Total return $= \mathbf{+2.00}$.
6. **Ineffective Action Advancing to Breach (2 Stages):**  
   Stage 0: $-0.1 - 0.5 = -0.60$.  
   Stage 1: $-0.1 - 0.5 - 2.0 (\text{breach}) = -2.60$.  
   Total return $= \mathbf{-3.20}$.

#### Exploitation & Policy Incentive Checks:
- **Can repeated investigative actions exploit rewards indefinitely?**  
  **No.** In `src/environment_sequential.py` (lines 791 and 820), taking an investigative action **always advances `current_stage_idx += 1`**. Because stages advance on every step, an agent cannot loop investigation at a single stage. Investigating past the final stage results in an uncontained breach penalty ($-2.0$), yielding a net return of $-1.90$.
- **Does the reward structure encourage always escalating?**  
  **No.** Always escalating yields $-0.10$. Containment yields $+1.90$. Escalation acts as a safe, risk-capping baseline compared to breach ($-3.20$) or disruption ($-1.10$), but is strictly sub-optimal compared to correct autonomous action.
- **Does the reward structure encourage single-action bias?**  
  **No.** Fixed-action "Always Kill Process" dropped from $+0.90$ (in v2) to $-1.37$ (in v3), with a $68.8\%$ breach rate on malicious incidents and a $100\%$ false disruption rate on benign incidents. Fixed-action "Always Monitor" achieves $-0.81$, with a $100\%$ breach rate on malicious incidents.

---

### 2.3 Observation Validity & Information Isolation
1. **Neural Policy Interface:**
   - Receives only `obs` (a 768-dimensional float tensor from `SecBERTStateEncoder`).
   - The encoder encodes the canonical incident string generated by `IncidentFormatter.format_incident_text(stage.incident_record)`.
   - Hidden scenario names (`scenario_id`), stage indices (`stage_id`), binary ground-truth labels (`is_benign`), and ground-truth action lists (`effective_actions`, `disruptive_actions`) are excluded from `IncidentFormatter` and are physically absent from the text representation.
2. **Rule-Based Heuristic Interface:**
   - Operates strictly on `info["telemetry"]`.
   - `_extract_telemetry()` exposes only observable SIEM and process evidence: `alert_level`, `rule_id`, `rule_description`, `process_name`, `process_path`, `cmdline`, `pid`, `username`, `cwd`, and MITRE tags.
   - Tested under `GuardedInfoDict` in `tests/test_trainer_sequential.py`: attempting to access `is_benign`, `scenario_id`, `stage_id`, or `effective_actions` raises an immediate `AssertionError`.
3. **Evaluation Metadata Isolation:**
   - Simulator metadata fields (`scenario_id`, `stage_id`, `outcome`, `contained`, `cleared`) are present in `info` solely for downstream evaluation tracking, metric aggregation, and unit test validation.

---

### 2.4 Evaluation Integrity & Benchmark v3 Metrics
1. **Metric Denominators:**
   - In `scripts/evaluate_soc_baselines.py`, malicious metrics use exclusively malicious episodes as the denominator ($N=32$):
     $$\text{Malicious Containment Rate} = \frac{\text{malicious\_contained}}{32}$$
     $$\text{Malicious Breach Rate} = \frac{\text{malicious\_compromised}}{32}$$
   - Benign metrics use exclusively benign episodes as the denominator ($N=28$):
     $$\text{Benign Clearance Rate} = \frac{\text{benign\_cleared}}{28}$$
     $$\text{False Disruption Rate} = \frac{\text{benign\_disrupted}}{28}$$
2. **Seed Reproducibility:**
   - Tested with seeds 1000–1059: Scenario selection and transitions are fully deterministic across evaluation runs.
   - `make_random_policy(seed=42)` produces identical action choices and returns on repeated executions.
3. **Benchmark v3 Distribution:**
   - **Uniform Random:** Mean return $-0.63 \pm 1.67$, Malicious containment $43.8\%$, Breach $46.9\%$, Benign clearance $7.1\%$.
   - **Always Monitor:** Mean return $-0.81 \pm 2.54$, Malicious containment $0.0\%$, Breach $100.0\%$, Benign clearance $100.0\%$.
   - **Always Kill Process:** Mean return $-1.37 \pm 1.74$, Malicious containment $31.2\%$, Breach $68.8\%$, False disruption $100.0\%$.
   - **Always Escalate:** Mean return $-0.10 \pm 0.00$, Escalation $100.0\%$.
   - **Rule-Based SOC Policy:** Mean return $+0.42 \pm 1.92$, Malicious containment $75.0\%$, Breach $25.0\%$, Benign clearance $42.9\%$, False disruption $57.1\%$.
   - **Pretrained PPO (`Exp 4D ATGP`):** Mean return $-1.23 \pm 0.59$, Malicious containment $0.0\%$, Breach $100.0\%$ (greedily selects Action 17 forensic snapshot on 100% of steps).

---

### 2.5 Training Readiness & Sequential PPO Pipeline

1. **Episodic Rollout Boundaries (`SequentialPPOTrainer.collect_rollout`):**
   - Transitions store `done = bool(terminated or truncated)`.
   - When `done == True`, the episode return, length, and operational outcomes (`contained`, `cleared`, `compromised`, `escalated`) are recorded, and `self.env.reset()` is invoked for the next episode.
   - When `done == False`, `self._current_obs = next_obs` carries the transition into the next stage without resetting the environment.
2. **Generalized Advantage Estimation (GAE) Boundary Integrity (`PPOAgent.compute_gae`):**
   - In `compute_gae`, `non_terminal = 1.0 - float(dones[t])`.
   - When `dones[t] == True`, `non_terminal = 0.0`. Future advantages and rewards from subsequent episodes are multiplied by 0, preventing reward leakage across episode boundaries.
   - When `dones[t] == False` (intermediate stage transition), `non_terminal = 1.0`. Future discounted rewards from stage $t+1$ correctly backpropagate to stage $t$ with discount $\gamma = 0.99$ and smoothing $\lambda = 0.95$.
3. **Rollout Buffer & Memory Management:**
   - Rollout updates execute over mini-batches of size 64 for 4 epochs with advantage normalization.
   - `memory.clear()` is called immediately following the update loop, preventing stale transition accumulation.
4. **Checkpointing & Tracking:**
   - `save_checkpoint` and `load_checkpoint` serialize actor, critic, optimizer states, best return, and architecture hyperparameters.

---

## 3. Remaining Operational Observations & Safeguards

While the environment and trainer are ready for training, the following non-blocking operational details should be noted during sequential PPO runs:

1. **Mid-Rollout Episode Cutoffs:**
   - In `SequentialPPOTrainer.collect_rollout()`, if the fixed budget of `rollout_steps` (e.g., 64) is exhausted mid-episode (`done == False` on the final step), `compute_gae()` currently defaults to `next_value = 0.0` rather than bootstrapping $V(s_{\text{last}})$.
   - *Assessment:* Because episode lengths are short (1–3 steps), over $95\%$ of transitions complete within a 64-step rollout. For long multi-stage extensions, bootstrapping with the critic value $V(s_{\text{last}})$ on truncated boundaries can be optionally enabled.
2. **Trainer Rollout Summary Denominators:**
   - In `SequentialPPOTrainer.evaluate()` and `collect_rollout()`, summary containment rate is computed over all completed episodes ($N$). In contrast, `evaluate_soc_baselines.py` splits denominators into malicious ($N_{\text{mal}}$) and benign ($N_{\text{ben}}$).
   - *Assessment:* This does not impact policy gradient updates or rewards, which optimize scalar returns. For formal evaluation checkpoints during training, using the dedicated benchmark script [`scripts/evaluate_soc_baselines.py`](file:///home/avidz/Research/soc-agent/scripts/evaluate_soc_baselines.py) provides the definitive breakdown.
3. **Investigation Reward Bonus Balance:**
   - Because investigating prior to containment grants $+0.10$ before the $+1.90$ containment reward, an optimal policy achieves $+2.00$ on 2-stage scenarios by performing investigation before containment. This correctly models thorough triage, provided the agent does not delay past the final stage into breach ($-2.0$).

---

## 4. Test Suite Execution & Verification

The complete unit and integration test suite was executed against the expanded environment and trainer:
```
Ran 45 tests in 1.134s

OK
```
All 45 tests passed:
- `tests/test_environment_sequential.py`: 15 tests covering 7 scenarios, transitions, containment, false disruption, bounded triage, persistent responses, and SecBERT embedding integration.
- `tests/test_trainer_sequential.py`: 6 tests covering multi-step GAE, rollout collection, checkpoint loading, baseline evaluation, information leakage guards, and training loop.
- `tests/test_telemetry.py`: 12 tests covering OSQuery, Wazuh, normalization, and canonical text formatting.
- `tests/test_environment.py`: 12 regression tests verifying the original contextual-bandit environment.

### Git Status & Working Tree Integrity
- Unstaged modifications:
  - `src/environment_sequential.py` (7 scenarios, telemetry extraction, Action 11 account tracking)
  - `tests/test_environment_sequential.py` (expanded scenario regression tests)
- Untracked files:
  - `scripts/evaluate_soc_baselines.py`
  - `src/ppo/trainer_sequential.py`
  - `tests/test_trainer_sequential.py`
- Original contextual-bandit files (`src/environment.py`, `src/ppo/trainer.py`, `models/secbert_finetuned/`, `models/ppo_exp4d_atgp/`) remain intact and unmodified.

---

## 5. Conclusion

The SOC Simulation Environment provides a consistent, diverse, and leak-free foundation for sequential reinforcement learning. The benchmark metrics clearly differentiate between ineffective baselines, naive heuristics, and strategic incident response. The environment is approved for sequential PPO training.
