# Phase 3: Operational SOC Simulation Environment (Integration Plan)

This document outlines the high-level roadmap for transitioning the autonomous Security Operations Center (SOC) incident response system from an **offline contextual-bandit system** (Phase 1 & 2) to an **operational SOC simulation environment** (Phase 3).

> **Note:** This is a rough structural plan. Exact architectures, actions, reward formulas, scenarios, and infrastructure choices will be determined based on practical evaluations of the existing project and hardware constraints.

---

## 1. SOC Infrastructure Setup
The foundational step is establishing an isolated, controllable lab environment to generate and monitor cyber incidents.
*   **Isolated Lab:** Set up the core network/endpoint environment for the simulation.
*   **Wazuh SIEM:** Deploy and configure Wazuh as the central aggregation point for security events.
*   **Telemetry Sources:** Deploy necessary endpoint and network monitoring tools (e.g., OSQuery, Zeek) to feed data to Wazuh.
*   **Adversary Simulation:** Install and configure **MITRE Caldera** to act as the automated adversary, executing controlled and repeatable attack behaviors.

## 2. Telemetry and Alert Pipeline
Ensure a reliable flow of data from the simulated attacks to the agent.
*   **Generate Activity:** Use Caldera to trigger specific attack techniques against the lab infrastructure.
*   **Collect Events:** Capture the resulting telemetry through Wazuh and associated tools.
*   **Standardization:** Establish a consistent data pipeline and formatting standard to convert raw telemetry logs into structured incident representations compatible with the agent.

## 3. Agent Integration Middleware
Bridge the gap between the live SOC infrastructure and the Python-based AI models.
*   **Data Flow Pipeline:** Build middleware that actively routes data:
    `SOC Telemetry -> Structured Incident -> SecBERT Encoder -> PPO (+ATGP) Policy -> Response Action`
*   **Live Connection:** Connect the previously trained models (Phase 1 and 2 checkpoints) to the live simulation environment.
*   **Model Preservation:** Ensure the integration reuses existing trained checkpoints (SecBERT and PPO) strictly in inference mode, rather than retraining them from scratch.

## 4. Stateful SOC Environment
Evolve the current `SOCEnvironment` from a static supervised dataset reader into a dynamic, multi-step simulation.
*   **State Transitions:** The environment must support sequential incident progression where an agent's action demonstrably affects the subsequent state.
*   **Action Consequences:** Define clear, programmatic rules for how the 20 discrete response actions impact the live (or simulated) infrastructure and the ongoing attack.

## 5. Response-Outcome and Reward Mechanism
Redesign the reward function to evaluate operational impact rather than static label accuracy.
*   **Dynamic Rewards:** Establish rewards based on the **simulated outcomes** of the agent's actions (e.g., stopping an active Caldera operation vs. failing to contain it).
*   **Outcome Representation:** Implement mechanisms to detect and reward specific operational states:
    *   Successful containment
    *   Unsuccessful response (attack continues)
    *   Unnecessary business disruption (false positive mitigation)
*   *Constraint:* These outcomes must only be implemented where the underlying simulation infrastructure genuinely supports detecting them.

## 6. Controlled Attack Scenarios
Develop a curriculum of attacks for evaluation.
*   **Scenario Creation:** Build a predefined set of repeatable SOC incident scenarios using Caldera profiles.
*   **Reproducibility:** Ensure these scenarios can be triggered identically across multiple evaluation runs to ensure fair comparisons between different agent policies.

## 7. Policy Evaluation
Conduct comparative testing under identical live-simulation conditions to measure actual operational efficacy.
*   **Models to Compare:**
    1.  SecBERT-only (Baseline Classification)
    2.  Standard PPO
    3.  PPO + ATGP (Adaptive Teacher-Guided Preservation)
*   **Evaluation Shift:** Move away from offline classification metrics (Macro F1, Accuracy) and focus on measuring practical operational performance.

## 8. Final Phase 3 Analysis
Synthesize the results to determine the true value of the RL interventions.
*   **Data Collection:** Record the actual sequence of actions, time-to-containment, and final outcomes of each response scenario.
*   **Metric Calculation:** Compute operational metrics that the implemented environment can reliably support (e.g., success rate, false disruption rate).
*   **Conclusion:** Determine whether the advanced PPO + ATGP methodology provides a statistically significant and practical advantage over baseline approaches in a live incident-response setting.
