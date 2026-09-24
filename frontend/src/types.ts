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
  likely_tools: string | null;
  likely_system_state: string | null;
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

export interface BranchingCandidate {
  action: string;
  probability: number;
  technique_id: string | null;
  technique_name: string;
  tactic: string;
}

export interface BranchingHorizon {
  horizon: number;
  candidates: BranchingCandidate[];
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
    branching_forecast: BranchingHorizon[];
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

export interface BranchNode {
  stage: string | null;
  step_probability: number | null;
  path_probability: number;
  infiltration_probability: number | null;
  attack_mapping: AttackMapping | null;
  depth: number;
  children: BranchNode[];
}

export interface BranchPath {
  stages: string[];
  path_probability: number;
  final_infiltration_probability: number;
  mitre_kill_chain: (AttackMapping & { stage: string })[];
}

export interface BranchingForecastResponse {
  host_id: string;
  window_idx: number;
  depth: number;
  branch_factor: number;
  tree: BranchNode;
  paths: BranchPath[];
  most_likely_path: BranchPath | null;
  highest_risk_path: BranchPath | null;
  true_stage: string | null;
  state_label: string | null;
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

export interface CounterfactualMetrics {
  mean_without: number;
  mean_with: number;
  peak_risk_without: number;
  peak_risk_with: number;
  risk_reduction_pct: number;
  verdict: string;
}

export interface CounterfactualResponse {
  host_id: string;
  window_idx: number;
  mitigation: MitigationInfo;
  horizon_windows: number;
  without_mitigation: CounterfactualScenario;
  with_mitigation: CounterfactualScenario;
  action_divergences: ActionDivergence[];
  metrics: CounterfactualMetrics;
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

export interface LivePacket {
  timestamp: number;
  direction: 'in' | 'out';
  local_port: number;
  remote_port: number;
  flags: string;
  ttl: number;
  win_size: number;
  pkt_len: number;
  description: string;
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

// ---- SHAP (baseline) vs attention + saliency (LSTM) -------------------------
export interface ShapContribution {
  feature: string;
  raw_value: number;
  scaled_value: number; // z-score vs. training data
  shap_value: number;   // log-odds; + pushes toward "malicious"
  direction: 'raises_risk' | 'lowers_risk';
}

export interface ShapResponse {
  host_id: string;
  window_idx: number;
  true_stage: string | null;
  state_label: string | null;
  shap: {
    base_value_logit: number;
    prediction_logit: number;
    baseline_probability: number;
    model_probability: number;
    sum_of_all_shap_values: number;
    contributions: ShapContribution[];
  };
  lstm: {
    infiltration_probability: number;
    attention_over_past_windows: number[];
    top_contributors: TopContributor[];
  };
  agreement: {
    shap_top_features: string[];
    lstm_top_features: string[];
    shared_features: string[];
    jaccard: number;
  };
}

// ---- Defense advisor ----------------------------------------------------------
export interface DefenseEvidence {
  id: string;
  label: string;
  description: string;
  mean_probability_without: number;
  mean_probability_with: number;
  mean_reduction: number;
  relative_reduction: number;
  peak_reduction: number;
  stages_changed: number;
  disruption_tier: number;
  disruption: 'low' | 'medium' | 'high';
  effective: boolean;
}

export interface PlaybookMitigation {
  id: string | null;
  name: string;
  action: string;
}

export interface DefenseAdvice {
  host_id: string;
  window_idx: number;
  risk_level: 'act_now' | 'watch' | 'monitor';
  peak_probability: number;
  expected_stage: string | null;
  horizon_windows: number;
  recommended: DefenseEvidence | null;
  recommendation_reason: string | null;
  evidence: DefenseEvidence[];
  playbook: {
    summary: string;
    mitigations: PlaybookMitigation[];
    analyst_steps: string[];
    attack_mapping?: AttackMapping | null;
  };
  method: { computed: string; static: string; limits: string; thresholds: Record<string, number> };
  true_stage: string | null;
  state_label: string | null;
  trajectory_without: number[];
  trajectory_with_recommended: number[] | null;
}

// -- step-by-step attacker tracking (GET /track/{host_id}) --------------------
export interface TrackCandidate { action: string; probability: number }
export interface NextMoveCandidate { move: string; probability: number }
export interface NextMoves {
  method: string;
  per_step: NextMoveCandidate[][];
  top_sequences: { moves: string[]; probability: number }[];
}
export interface TrackStep {
  step: number;
  window_idx: number;
  history_windows_used: number;
  warmup: boolean;
  predicted_next_action: string;
  confidence: number;
  top_candidates: TrackCandidate[];
  infiltration_probability: number;
  attack_path_so_far: string[];
  next_moves: NextMoves | null;
  actual_current_action?: string;
  actual_next_action?: string | null;
  correct?: boolean | null;
  correct_top3?: boolean;
  is_transition?: boolean;
}
export interface TrackSummary {
  n_windows: number;
  n_warmup_windows: number;
  first_prediction_after_window: number | null;
  old_first_prediction_after_window: number;
  first_attack_alert_step: number | null;
  attack_path_recognised: string[];
  decision_rule: string;
  accuracy_top1?: number;
  accuracy_top3?: number;
  accuracy_warmup_windows?: number | null;
  accuracy_full_history_windows?: number | null;
  n_transitions?: number;
  accuracy_on_transitions?: number | null;
  attack_path_actual?: string[] | null;
}
export interface TrackResponse { host_id: string; steps: TrackStep[]; summary: TrackSummary }

// -- GET /step-tracking-report -------------------------------------------------
export interface MoveScore { n: number; exact_match: number; tactic_level_match: number; top3_accuracy?: number }
export type MethodScores = Partial<Record<'next_1' | 'next_2' | 'next_3', MoveScore>>;
export interface StepTrackingReport {
  decision_metric: Record<string, string>;
  window_level_tracking: Record<string, any>;
  cold_start: { setup: string; accuracy_by_step: Record<string, number | null>; accuracy_warmup_overall: number | null; old_pipeline_first_prediction_after_window: number };
  path_recognition: { n_attack_hosts: number; tactic_level_exact_match: number; technique_level_exact_match: number; heldout_only: Record<string, number> };
  multi_step_moves: {
    unit: string;
    leave_one_host_out_all_attack_hosts: Record<string, MethodScores>;
    heldout_hosts_vs_lstm: { hosts: string[]; note: string } & Record<string, any>;
  };
}
