/**
 * Strict TypeScript interfaces for SOC Simulation Environment Web Dashboard.
 * All interfaces map directly to verified research contracts and backend API responses.
 */

export interface ActionSpaceItem {
  index: number;
  name: string;
  phase: string;
  threat_match: string;
  implementation: string;
}

export interface TelemetrySource {
  name: string;
  category: string;
  status: string;
  endpoint: string | null;
  last_seen: string | null;
  collection_interval_sec: number | null;
  processed_count: number;
  parsing_errors: number;
  availability: "Connected" | "Disconnected" | "Degraded";
  note: string;
}

export interface IncidentEvent {
  id: string;
  split: "test" | "val" | "train";
  index: number;
  text: string;
  prepared_text: string;
  action_label: number;
  action_name: string;
  cisa_phase: "Detection" | "Containment" | "Eradication" | "All Phases" | string;
  threat_match: string;
  implementation: string;
  source_dataset: "CISSM" | "rcATT" | string;
  target: string;
  attack_type: string;
  summary: string;
  linked_decision_id?: string | null;
  mitre_mapping?: string;
  data_integrity?: {
    is_deduplicated: boolean;
    length_characters: number;
    token_estimate: number;
  };
}

export interface DecisionProbability {
  index: number;
  name: string;
  phase: string;
  probability: number;
  is_chosen: boolean;
  is_ground_truth: boolean;
}

export interface Decision {
  id: string;
  incident_id: string;
  timestamp: string;
  policy: string;
  encoder_checkpoint: string;
  frozen_encoder: boolean;
  embedding_dim: number;
  sample_embedding: number[];
  embedding_norm: number;
  embedding_validation: {
    nan_count: number;
    inf_count: number;
    passed: boolean;
  };
  incident_text: string;
  prepared_text: string;
  cisa_phase: string;
  ground_truth_action: number;
  ground_truth_name: string;
  chosen_action: number;
  chosen_action_name: string;
  action_probabilities: number[];
  probabilities_breakdown?: DecisionProbability[];
  critic_value: number;
  reward: number;
  is_correct: boolean;
  simulated_effect: string;
  operational_containment: string;
  source_dataset: string;
  threat_match?: string;
  implementation?: string;
}

export interface SimulationStep {
  step: number;
  incident_id: string;
  incident_summary: string;
  chosen_action: number;
  action_name: string;
  ground_truth: number;
  ground_truth_name: string;
  is_correct: boolean;
  reward: number;
  cumulative_reward: number;
  critic_value: number;
  simulated_state: string;
}

export interface SimulationRun {
  id: string;
  scenario: string;
  scenario_id?: string;
  simulation_mode?: string;
  policy: string;
  checkpoint?: string;
  encoder: string;
  target?: string;
  seed?: number;
  start_time: string;
  end_time?: string;
  steps_count: number;
  max_steps?: number;
  correct_steps?: number;
  accuracy: number;
  final_cumulative_reward: number;
  mean_reward: number;
  status: "Completed" | "Running" | "Failed" | "completed" | "running" | "stopped" | "failed" | string;
  environment_type?: string;
  steps?: SimulationStep[];
  results?: any;
}

export interface EvaluationArchitecture {
  id: string;
  name: string;
  checkpoint: string;
  status: string;
  type: string;
  accuracy: number | null;
  macro_f1: number | null;
  weighted_f1: number | null;
  mcc: number | null;
  top2_accuracy: number | null;
  contextual_bandit_reward: string | number | null;
  top_performing: boolean;
  notes?: string;
}

export interface OperationalMetrics {
  evaluation_mode: string;
  containment_success_rate: string;
  response_success_rate: string;
  time_to_containment: string;
  incident_resolution_rate: string;
  operational_disruption: string;
  cumulative_reward_64_steps: string;
  status_disclaimer: string;
}

export interface PerClassMetric {
  index: number;
  action: string;
  samples: number;
  precision: number;
  recall: number;
  f1: number;
  note?: string;
}

export interface ConfusionMatrixData {
  labels: string[];
  matrix: number[][];
}

export interface FailureCase {
  id: string;
  incident_id: string;
  text: string;
  ground_truth: string;
  predicted: string;
  critic_value: number;
  root_cause: string;
}

export interface SimulationComparisonResponse {
  run_a: {
    id: string;
    scenario: string;
    policy: string;
    checkpoint: string;
    seed: number;
    steps_count: number;
    accuracy: number;
    cumulative_reward: number;
    mean_reward: number;
    status: string;
  };
  run_b: {
    id: string;
    scenario: string;
    policy: string;
    checkpoint: string;
    seed: number;
    steps_count: number;
    accuracy: number;
    cumulative_reward: number;
    mean_reward: number;
    status: string;
  };
  compatibility: "compatible" | "partially_compatible" | "incompatible";
  compatibility_reasons: string[];
  metrics_diff: {
    accuracy_diff: number;
    reward_diff: number;
    mean_reward_diff: number;
    steps_diff: number;
  };
  action_distribution: {
    policy_a: Record<string, number>;
    policy_b: Record<string, number>;
  };
  trajectory_comparison: Array<{
    step: number;
    reward_a: number | null;
    reward_b: number | null;
    cum_reward_a: number | null;
    cum_reward_b: number | null;
    action_a: string;
    action_b: string;
    is_correct_a: boolean;
    is_correct_b: boolean;
  }>;
  scientific_notes: {
    containment_evaluation: string;
    statistical_significance: string;
  };
}

export interface ClassificationMetricsSummary {
  dataset_split: string;
  seed: number;
  experiment_id: string;
  encoder: string;
  accuracy: number;
  macro_f1: number;
  weighted_f1: number;
  mcc: number;
  top2_accuracy: number;
  sample_count: number;
  classes_count: number;
  zero_shot_classes: string[];
  provenance_note: string;
}

export interface SequentialSimulationMetricsSummary {
  total_runs_recorded: number;
  containment_success_rate: string;
  containment_note: string;
  time_to_containment: string;
  operational_disruption: string;
  mean_cumulative_reward: number;
  median_cumulative_reward: number;
  mean_simulation_steps: number;
  terminal_outcomes: {
    completed: number;
    stopped: number;
    failed: number;
    running: number;
  };
  failure_rate: number;
  truncation_rate: number;
  supported_scenarios: string[];
  supported_policies: string[];
}

export interface EvaluationResponse {
  architectures: EvaluationArchitecture[];
  operational_metrics: OperationalMetrics;
  per_class_metrics: PerClassMetric[];
  confusion_matrix_sample: ConfusionMatrixData;
  failure_cases: FailureCase[];
  classification_metrics?: ClassificationMetricsSummary;
  sequential_simulation_metrics?: SequentialSimulationMetricsSummary;
}

export interface Phase1Experiment {
  logical_name: string;
  system_id: string;
  description: string;
  macro_f1: number;
  top2_accuracy: number | null;
  weighted_f1?: number;
  mcc?: number;
  verdict: string;
}

export interface Phase2PPOExperiment {
  milestone: string;
  date: string;
  audit_score: string;
  status: string;
  components: string[];
}

export interface ResearchExperimentsResponse {
  phase1_secbert: Phase1Experiment[];
  phase2_ppo: Phase2PPOExperiment[];
  locked_checkpoint_metadata: {
    checkpoint_id: string;
    framework: string;
    base_model: string;
    embedding_dimension: number;
    max_sequence_length: number;
    classification_classes: number;
    zero_shot_classes_excluded_from_loss: number[];
    freeze_policy: string;
  };
}

export interface SystemStatusResponse {
  infrastructure: {
    docker: { status: string; detail: string };
    wazuh_manager: { status: string; detail: string };
    wazuh_indexer: { status: string; detail: string };
    wazuh_dashboard: { status: string; detail: string };
    osquery: { status: string; detail: string };
    caldera: { status: string; detail: string };
  };
  data_pipeline: {
    cissm_records: number;
    rcatt_records: number;
    train_set_size: number;
    val_set_size: number;
    test_set_size: number;
    total_records: number;
    status: string;
    deduplication_technique: string;
    leakage_prevention: string;
    zero_shot_classes: string[];
  };
  ai_components: {
    secbert: {
      checkpoint: string;
      status: string;
      embedding_dim: number;
      hidden_layers: number;
      attention_heads: number;
      classifier_head_status: string;
      l2_normalization: boolean;
    };
    ppo: {
      checkpoint: string;
      status: string;
      actor_architecture: string;
      critic_architecture: string;
      audit_score: string;
      gae_lambda: number;
      gamma: number;
      clip_eps: number;
      c2_entropy: number;
    };
  };
  environment: {
    formulation: string;
    sequential_soc_env: string;
    reward_function: string;
    containment_mode: string;
  };
}

export interface HealthResponse {
  status: "online" | "degraded" | "offline";
  service: string;
  timestamp: string;
  python_version: string;
  torch_available: boolean;
  torch_note: string;
  backend_mode: string;
  datasets_ready: boolean;
  action_space_size: number;
  total_benchmark_records: number;
  checkpoint_audit?: {
    secbert?: { exists_on_disk?: boolean; status?: string; path?: string };
    ppo?: { exists_on_disk?: boolean; status?: string; path?: string };
    ppo_atgp?: { exists_on_disk?: boolean; status?: string; path?: string };
  };
}

export interface ScenarioStage {
  stage: number;
  name: string;
  target_action: string;
  tactics: string[];
}

export interface ScenarioItem {
  id: string;
  alias?: string;
  name: string;
  description: string;
  simulation_mode: string;
  validation_status: string;
  target_actions?: number[];
  expected_actions?: string[];
  stages?: ScenarioStage[];
  tactics: string[];
  target_default: string;
  initial_state?: string;
  supported_steps?: { min: number; max: number; default: number };
  observation_method?: string;
  outcome_criteria?: string;
  measurable_metrics?: string;
  supported_policies?: string[];
  disabled?: boolean;
  disabled_reason?: string;
}

export interface StartSimulationPayload {
  scenario: string;
  policy: "PPO" | "PPO+ATGP";
  target?: string;
  step_limit?: number;
  steps?: number;
  seed?: number;
}

export interface StartSimulationResponse {
  run_id: string;
  scenario: string;
  scenario_id: string;
  policy: string;
  target: string;
  step_limit: number;
  seed: number;
  status: string;
  created_at: string;
  message: string;
}

export interface SimulationEvent {
  run_id: string;
  seq: number;
  timestamp: string;
  event_type:
    | "simulation_started"
    | "attack_stage_updated"
    | "incident_observed"
    | "embedding_generated"
    | "action_selected"
    | "simulated_effect_applied"
    | "reward_calculated"
    | "state_updated"
    | "simulation_completed"
    | "simulation_stopped";
  data: any;
}

export interface SimulationResults {
  run_id: string;
  scenario: string;
  policy: string;
  target: string;
  total_steps: number;
  correct_steps: number;
  accuracy: number;
  cumulative_reward: number;
  mean_reward: number;
  containment_outcome: string;
  execution_duration_sec: number;
  status: string;
  completed_at: string;
}

