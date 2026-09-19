export interface Kpis {
  hosts_monitored: number;
  forecasts_generated_today: number;
  high_risk_trajectories: number;
  median_lead_time_minutes: number | null;
}

export interface AttackMapping {
  state_label: string;
  technique_id: string | null;
  technique_name: string;
  tactic: string;
}

export interface HighestRiskHost {
  host_id: string;
  window_idx: number;
  predicted_stage: string | null;
  infiltration_probability: number;
  created_at: string;
  attack_mapping: AttackMapping | null;
}

export interface TopContributor {
  timestep_offset: number;
  feature: string;
  attention_weight: number;
  raw_value_normalized: number;
  contribution_score: number;
}

export interface ForecastResponse {
  host_id: string;
  window_idx: number;
  predicted_stage: string;
  attack_mapping: AttackMapping;
  infiltration_probability_world_model: number;
  infiltration_probability_baseline: number;
  stage_probabilities: Record<string, number>;
  explanation: {
    attention_over_past_windows: number[];
    top_contributors: TopContributor[];
  };
  rollout: {
    horizon_windows: number;
    infiltration_probs_world_model: number[];
    predicted_stage_per_horizon: string[];
  };
  true_stage: string | null;
  state_label: string | null;
}

export interface ForecastLogRow {
  host_id: string;
  window_idx: number;
  created_at: string;
  model: string;
  predicted_stage: string | null;
  infiltration_probability: number;
  true_stage: string | null;
  state_label: string | null;
}

export interface StageBreakdown {
  total: number;
  counts: Record<string, number>;
}

export interface BinaryMetrics {
  precision: number;
  recall: number;
  f1: number;
  false_positive_rate: number;
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  n_samples: number;
}

export interface BenchmarkReport {
  threshold: number;
  baseline_logistic_regression: BinaryMetrics;
  world_model_lstm: BinaryMetrics;
  note: string;
}

export interface CalibrationBin {
  bin_lo: number;
  bin_hi: number;
  count: number;
  mean_predicted: number | null;
  observed_frequency: number | null;
}

export interface CalibrationReport {
  horizon_windows: number;
  n_points: number;
  bins: CalibrationBin[];
  brier_score: number;
}

export interface LeadTimeHostEntry {
  host_id: string;
  split: string;
  baseline_fire_window: number | null;
  worldmodel_alert_window: number | null;
  lead_time_windows: number | null;
  lead_time_minutes: number | null;
  correctly_flagged_by_both: boolean;
}

export interface LeadTimeReport {
  threshold: number;
  window_seconds: number;
  per_host: LeadTimeHostEntry[];
  median_lead_time_windows_all_hosts: number | null;
  median_lead_time_minutes_all_hosts: number | null;
  median_lead_time_windows_heldout_only: number | null;
  median_lead_time_minutes_heldout_only: number | null;
  n_hosts_correctly_flagged_by_both: number;
  n_attack_hosts_total: number;
  note: string;
}

export interface FalseAlarmExample {
  host_id: string;
  window_idx: number;
  infiltration_probability: number;
  true_stage: string;
  outcome: string;
}

export interface SandboxTestResult {
  outcome: 'success' | 'failure';
  errors: string[];
  rows: number;
  hosts: number;
}

export interface HostTimelineEntry {
  window_idx: number;
  true_stage: string;
  state_label: string;
}

export interface MitigationInfo {
  id: string;
  label: string;
  description: string;
}

export interface CounterfactualScenario {
  infiltration_probs: number[];
  predicted_stage_per_horizon: string[];
}

export interface ActionDivergence {
  horizon: number;
  without_mitigation_action: string;
  with_mitigation_action: string;
}

export interface CounterfactualResponse {
  host_id: string;
  window_idx: number;
  mitigation: MitigationInfo;
  horizon_windows: number;
  without_mitigation: CounterfactualScenario;
  with_mitigation: CounterfactualScenario;
  action_divergences: ActionDivergence[];
  true_stage: string | null;
  state_label: string | null;
}

export interface LiveInterface {
  name: string;
  description: string;
  ip: string;
}

export interface LiveStatus {
  running: boolean;
  iface: string | null;
  local_ip: string | null;
  started_at: string | null;
  packets_seen: number;
  hosts_seen: number;
  error: string | null;
}

export interface TripwireAlert {
  id: number;
  timestamp: string;
  remote_ip: string;
  message: string;
  severity: 'warning' | 'critical';
  detail: Record<string, unknown>;
}

export interface LiveWindowEntry {
  host_id: string;
  window_idx: number;
  timestamp: string;
  raw_features: Record<string, number>;
  predicted_stage: string | null;
  attack_mapping?: AttackMapping;
  infiltration_probability_world_model: number | null;
  infiltration_probability_baseline: number | null;
  stage_probabilities?: Record<string, number>;
}
