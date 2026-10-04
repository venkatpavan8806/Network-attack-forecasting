import axios from 'axios';
import type {
  Kpis, HighestRiskHost, ForecastResponse, ForecastLogRow, StageBreakdown,
  BenchmarkReport, CalibrationReport, LeadTimeReport, FalseAlarmExample,
  SandboxTestResult, AttackMapping, MitigationInfo, CounterfactualResponse,
  HostTimelineEntry, LiveStatus, LiveWindowEntry, LiveInterface, TripwireAlert, LivePacket,
  ShapResponse, DefenseAdvice, BranchingForecastResponse, TrackResponse, StepTrackingReport,
  ThresholdCalibrationReport, RobustnessReport,
} from './types';
import { supabase } from './supabase';

// Backend URL from VITE_API_URL (e.g. the Render service); locally '/api' is proxied by vite.config.ts.
const client = axios.create({ baseURL: ((import.meta.env.VITE_API_URL as string | undefined) || '/api').replace(/\/+$/, '') });

// Send the signed-in user's Supabase token with every request, so the
// backend returns only that user's own data.
client.interceptors.request.use(async (config) => {
  const { data } = (await supabase?.auth.getSession()) ?? { data: { session: null } };
  if (data.session) config.headers.set('Authorization', `Bearer ${data.session.access_token}`);
  return config;
});

export const api = {
  health: () => client.get('/health').then((r) => r.data),
  kpis: () => client.get<Kpis>('/kpis').then((r) => r.data),
  highestRiskHost: () => client.get<HighestRiskHost | null>('/highest-risk-host').then((r) => r.data),
  hosts: () => client.get<string[]>('/hosts').then((r) => r.data),
  liveForecastableHosts: () => client.get<string[]>('/live/forecastable-hosts').then((r) => r.data),
  hostTimeline: (hostId: string) => client.get<HostTimelineEntry[]>(`/host-timeline/${hostId}`).then((r) => r.data),
  forecast: (hostId: string, atWindowIdx?: number) =>
    client.get<ForecastResponse>(`/forecast/${hostId}`, { params: atWindowIdx != null ? { at_window_idx: atWindowIdx } : {} }).then((r) => r.data),
  branchingForecast: (hostId: string, atWindowIdx?: number, depth?: number, branchFactor?: number) =>
    client.get<BranchingForecastResponse>(`/forecast/${hostId}/branches`, {
      params: {
        ...(atWindowIdx != null ? { at_window_idx: atWindowIdx } : {}),
        ...(depth != null ? { depth } : {}),
        ...(branchFactor != null ? { branch_factor: branchFactor } : {}),
      },
    }).then((r) => r.data),
  track: (hostId: string, atWindowIdx?: number) =>
    client.get<TrackResponse>(`/track/${encodeURIComponent(hostId)}`, { params: atWindowIdx != null ? { at_window_idx: atWindowIdx } : {} }).then((r) => r.data),
  stepTrackingReport: () => client.get<StepTrackingReport>('/step-tracking-report').then((r) => r.data),
  thresholdCalibration: () => client.get<ThresholdCalibrationReport>('/threshold-calibration').then((r) => r.data),
  robustnessReport: () => client.get<RobustnessReport>('/robustness-report').then((r) => r.data),
  attackStageBreakdown: () => client.get<StageBreakdown>('/attack-stage-breakdown').then((r) => r.data),
  forecastLog: (limit = 25) => client.get<ForecastLogRow[]>('/forecast-log', { params: { limit } }).then((r) => r.data),
  attackMapping: () => client.get<AttackMapping[]>('/attack-mapping').then((r) => r.data),
  benchmark: () => client.get<BenchmarkReport>('/benchmark').then((r) => r.data),
  calibration: () => client.get<CalibrationReport>('/calibration').then((r) => r.data),
  leadTime: () => client.get<LeadTimeReport>('/lead-time').then((r) => r.data),
  falseAlarms: () => client.get<FalseAlarmExample[]>('/false-alarms').then((r) => r.data),
  stageClasses: () => client.get<string[]>('/stage-classes').then((r) => r.data),
  mitigations: () => client.get<MitigationInfo[]>('/mitigations').then((r) => r.data),
  counterfactual: (hostId: string, mitigationId: string, atWindowIdx?: number) =>
    client.get<CounterfactualResponse>(`/counterfactual/${hostId}`, {
      params: { mitigation_id: mitigationId, ...(atWindowIdx != null ? { at_window_idx: atWindowIdx } : {}) },
    }).then((r) => r.data),
  shap: (hostId: string, atWindowIdx?: number) =>
    client.get<ShapResponse>(`/shap/${encodeURIComponent(hostId)}`, { params: atWindowIdx != null ? { at_window_idx: atWindowIdx } : {} }).then((r) => r.data),
  defense: (hostId: string, atWindowIdx?: number) =>
    client.get<DefenseAdvice>(`/defense/${encodeURIComponent(hostId)}`, { params: atWindowIdx != null ? { at_window_idx: atWindowIdx } : {} }).then((r) => r.data),
  sandboxTest: (file: File) => {
    const form = new FormData();
    form.append('file', file);
    return client.post<SandboxTestResult>('/sandbox/test', form).then((r) => r.data);
  },
  ingest: (file: File) => {
    const form = new FormData();
    form.append('file', file);
    return client.post<ForecastResponse[]>('/ingest', form).then((r) => r.data);
  },
  liveInterfaces: () => client.get<LiveInterface[]>('/live/interfaces').then((r) => r.data),
  liveStatus: () => client.get<LiveStatus>('/live/status').then((r) => r.data),
  liveStart: (iface: string, localIp: string) =>
    client.post<LiveStatus>('/live/start', { iface, local_ip: localIp }).then((r) => r.data),
  liveStop: () => client.post<LiveStatus>('/live/stop').then((r) => r.data),
  liveRecent: (limit = 50) => client.get<LiveWindowEntry[]>('/live/recent', { params: { limit } }).then((r) => r.data),
  liveAlerts: (limit = 50) => client.get<TripwireAlert[]>('/live/alerts', { params: { limit } }).then((r) => r.data),
  livePackets: (remoteIp: string, limit = 100) =>
    client.get<LivePacket[]>(`/live/packets/${remoteIp}`, { params: { limit } }).then((r) => r.data),
};

// Colors follow the attack-progression: recon (amber) -> credential access
// (orange) -> lateral movement (coral) -> C2 (red) -> exfiltration (deep red)
export const STAGE_COLORS: Record<string, string> = {
  benign: '#3dd598',
  ambiguous_pre_attack: '#ffb84d',
  port_scan: '#f7b955',
  ssh_bruteforce: '#ff9f5b',
  rdp_bruteforce: '#ff8f5b',
  smb_bruteforce: '#ff7f5b',
  ssh_lateral_movement: '#ff7d6b',
  rdp_lateral_movement: '#ff6d6b',
  smb_lateral_movement: '#ff5d6b',
  c2_beacon: '#ff6b81',
  data_exfiltration: '#c94f6d',
};

export const STAGE_LABELS: Record<string, string> = {
  benign: 'Benign',
  ambiguous_pre_attack: 'Ambiguous Pre-Attack',
  port_scan: 'Port Scan',
  ssh_bruteforce: 'SSH Brute-Force',
  rdp_bruteforce: 'RDP Brute-Force',
  smb_bruteforce: 'SMB Brute-Force',
  ssh_lateral_movement: 'SSH Lateral Movement',
  rdp_lateral_movement: 'RDP Lateral Movement',
  smb_lateral_movement: 'SMB Lateral Movement',
  c2_beacon: 'C2 Beaconing',
  data_exfiltration: 'Data Exfiltration',
};

// compact labels for space-constrained diagram nodes
export const STAGE_SHORT_LABELS: Record<string, string> = {
  benign: 'Benign',
  ambiguous_pre_attack: 'Ambig.',
  port_scan: 'Scan',
  ssh_bruteforce: 'SSH-BF',
  rdp_bruteforce: 'RDP-BF',
  smb_bruteforce: 'SMB-BF',
  ssh_lateral_movement: 'SSH-Lat',
  rdp_lateral_movement: 'RDP-Lat',
  smb_lateral_movement: 'SMB-Lat',
  c2_beacon: 'C2',
  data_exfiltration: 'Exfil',
};
