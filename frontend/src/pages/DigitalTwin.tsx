import { useEffect, useState } from 'react';
import { api, STAGE_LABELS } from '../api';
import type { CounterfactualResponse, HostTimelineEntry, MitigationInfo } from '../types';
import CardHeader from '../components/CardHeader';
import CounterfactualChart, { type CounterfactualPoint } from '../components/CounterfactualChart';
import DefenseAdvisor from '../components/DefenseAdvisor';
import TrajectoryPathDiagram from '../components/TrajectoryPathDiagram';
import { NetworkIcon, ShieldIcon, AlertIcon } from '../icons';

export default function DigitalTwin({ selectedHost, onSelectHost }: { selectedHost: string | null; onSelectHost: (h: string) => void }) {
  const [hosts, setHosts] = useState<string[]>([]);
  const [mitigations, setMitigations] = useState<MitigationInfo[]>([]);
  const [mitigationId, setMitigationId] = useState('block_ssh');
  const [timeline, setTimeline] = useState<HostTimelineEntry[]>([]);
  const [windowIdx, setWindowIdx] = useState<number | null>(null);
  const [result, setResult] = useState<CounterfactualResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [liveHosts, setLiveHosts] = useState<string[]>([]);
  const isLive = selectedHost?.startsWith('live:') ?? false;

  useEffect(() => {
    api.hosts().then((h) => {
      const attackHosts = h.filter((x) => x.includes('attack'));
      setHosts(attackHosts.length ? attackHosts : h);
      if (!selectedHost && (attackHosts.length || h.length)) onSelectHost((attackHosts[0] ?? h[0]));
    }).catch(() => {});
    api.mitigations().then(setMitigations).catch(() => {});
    api.liveForecastableHosts().then(setLiveHosts).catch(() => {});
  }, []);

  useEffect(() => {
    if (!selectedHost || isLive) return;
    api.hostTimeline(selectedHost).then((t) => {
      setTimeline(t);
      // labelled traffic (sample data / labelled CSV): start just before the attack begins;
      // unlabelled traffic (pcap uploads): start from the latest window
      const firstNonBenign = t.find((r) => r.true_stage != null && r.true_stage !== 'benign');
      setWindowIdx(firstNonBenign ? Math.max(t[0].window_idx, firstNonBenign.window_idx - 1) : t[Math.max(0, t.length - 1)]?.window_idx ?? null);
    }).catch(() => {});
  }, [selectedHost, isLive]);

  async function runWhatIf() {
    if (!selectedHost) return;
    if (!isLive && windowIdx == null) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.counterfactual(selectedHost, mitigationId, isLive ? undefined : windowIdx ?? undefined);
      setResult(res);
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? String(e));
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (selectedHost && (isLive || windowIdx != null)) runWhatIf();
  }, [selectedHost, windowIdx, mitigationId, isLive]);

  // group the timeline into contiguous action segments for the "jump to" picker
  const segments: { action: string; startWindow: number; endWindow: number }[] = [];
  for (const row of timeline) {
    if (!row.true_stage) continue; // unlabelled (live / pcap) windows have no ground-truth segment
    const last = segments[segments.length - 1];
    if (last && last.action === row.true_stage) {
      last.endWindow = row.window_idx;
    } else {
      segments.push({ action: row.true_stage, startWindow: row.window_idx, endWindow: row.window_idx });
    }
  }

  const chartData: CounterfactualPoint[] = result
    ? result.without_mitigation.infiltration_probs.map((p, i) => ({
        step: `t+${i + 1}`,
        without: p,
        with_: result.with_mitigation.infiltration_probs[i],
      }))
    : [];

  const probDeltaPct = result
    ? Math.max(...result.without_mitigation.infiltration_probs.map((p, i) => (p - result.with_mitigation.infiltration_probs[i]) * 100))
    : 0;
  const probMoved = (result?.metrics.risk_reduction_pct ?? 0) > 5;
  const actionChanged = (result?.action_divergences.length ?? 0) > 0;

  let verdict: string | null = null;
  if (result) {
    if (probMoved && actionChanged) {
      verdict = `This mitigation lowered the predicted infiltration probability (by up to ${probDeltaPct.toFixed(1)} points) and altered the attacker's trajectory in the twin state.`;
    } else if (probMoved) {
      verdict = `This mitigation lowered predicted infiltration probability by up to ${probDeltaPct.toFixed(1)} points — the simulated state suppressed attack telemetry.`;
    } else if (actionChanged) {
      verdict = `This mitigation changed the expected next actions on the twin state, although overall risk remains elevated due to prior attack confirmation.`;
    } else {
      verdict = `This mitigation had minimal effect at this point — try applying it earlier in the timeline.`;
    }
  }

  const twinState = result?.cloned_twin_state;
  const attackerResp = result?.simulated_attacker_response;
  const pathPrediction = result?.path_prediction;

  return (
    <div className="flex flex-col gap-6">
      {/* Top Banner / Conceptual Flow */}
      <div className="card p-6">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
          <CardHeader
            title="Digital Twin — Safe Sandbox & Network Simulation"
            subtitle="Simulates network state & attacker interaction before applying defenses for real."
          />
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-[var(--color-good)]/10 text-[var(--color-good)] text-xs font-semibold border border-[var(--color-good)]/30">
            <ShieldIcon size={14} />
            SAFE SANDBOX (real_network_touched: false)
          </div>
        </div>

        {/* Conceptual Loop Visualization */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-2 text-center text-xs mb-5">
          <div className="p-3 rounded-xl bg-[var(--color-accent-soft)]/50 border border-[var(--color-accent-soft)]">
            <div className="font-semibold text-[var(--color-ink)] mb-1">1. Live / Observed</div>
            <div className="text-[var(--color-ink-faint)]">Observed Network State</div>
          </div>
          <div className="p-3 rounded-xl bg-[var(--color-accent-soft)]/50 border border-[var(--color-accent-soft)]">
            <div className="font-semibold text-[var(--color-ink)] mb-1">2. Digital Twin</div>
            <div className="text-[var(--color-ink-faint)]">Clone Simulated Twin</div>
          </div>
          <div className="p-3 rounded-xl bg-[var(--color-accent-soft)]/50 border border-[var(--color-accent-soft)]">
            <div className="font-semibold text-[var(--color-ink)] mb-1">3. Mitigation</div>
            <div className="text-[var(--color-ink-faint)]">Apply State Mitigation</div>
          </div>
          <div className="p-3 rounded-xl bg-[var(--color-accent-soft)]/50 border border-[var(--color-accent-soft)]">
            <div className="font-semibold text-[var(--color-ink)] mb-1">4. Sim Response</div>
            <div className="text-[var(--color-ink-faint)]">Recalculate Paths</div>
          </div>
          <div className="p-3 rounded-xl bg-[var(--color-accent-soft)]/50 border border-[var(--color-accent-soft)]">
            <div className="font-semibold text-[var(--color-ink)] mb-1">5. World Model</div>
            <div className="text-[var(--color-ink-faint)]">LSTM Trajectory Forecast</div>
          </div>
        </div>

        {/* Picker Controls */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="text-xs text-[var(--color-ink-faint)] mb-1.5 block font-medium">Target Host</label>
            <select
              value={selectedHost ?? ''}
              onChange={(e) => onSelectHost(e.target.value)}
              className="w-full px-3 py-2 rounded-xl border border-[var(--color-accent-soft)] text-sm bg-white text-[var(--color-ink)]"
            >
              <optgroup label="Demo dataset (CSV-replay)">
                {hosts.map((h) => <option key={h} value={h}>{h}</option>)}
              </optgroup>
              {liveHosts.length > 0 && (
                <optgroup label="Live capture">
                  {liveHosts.map((h) => <option key={h} value={h}>{h.replace('live:', '')}</option>)}
                </optgroup>
              )}
            </select>
          </div>
          <div>
            <label className="text-xs text-[var(--color-ink-faint)] mb-1.5 block font-medium">Apply Mitigation Starting At</label>
            {isLive ? (
              <div className="w-full px-3 py-2 rounded-xl border border-[var(--color-accent-soft)] text-sm bg-[var(--color-accent-soft)]/40 text-[var(--color-ink-dim)]">
                now (live host window)
              </div>
            ) : (
              <select
                value={windowIdx ?? ''}
                onChange={(e) => setWindowIdx(Number(e.target.value))}
                className="w-full px-3 py-2 rounded-xl border border-[var(--color-accent-soft)] text-sm bg-white text-[var(--color-ink)]"
              >
                {segments.map((s) => (
                  <option key={s.startWindow} value={s.endWindow}>
                    end of {STAGE_LABELS[s.action] ?? s.action} (window #{s.endWindow})
                  </option>
                ))}
              </select>
            )}
          </div>
          <div>
            <label className="text-xs text-[var(--color-ink-faint)] mb-1.5 block font-medium">Defensive Mitigation</label>
            <select
              value={mitigationId}
              onChange={(e) => setMitigationId(e.target.value)}
              className="w-full px-3 py-2 rounded-xl border border-[var(--color-accent-soft)] text-sm bg-white text-[var(--color-ink)]"
            >
              {mitigations.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
            </select>
          </div>
        </div>

        {result && (
          <p className="text-xs text-[var(--color-ink-faint)] mt-3">
            <strong>Selected Mitigation:</strong> {mitigations.find((m) => m.id === mitigationId)?.description}
          </p>
        )}
      </div>

      {loading && <div className="card p-10 text-center text-sm text-[var(--color-ink-faint)]">running safe digital twin sandbox simulation…</div>}
      {error && <div className="card p-6 text-sm text-[var(--color-bad)]">{error}</div>}

      {result && !loading && (
        <>
          {/* Simulated Attacker Response & Twin State Details */}
          {attackerResp && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Attacker Response Card */}
              <div className="card p-6 flex flex-col justify-between">
                <div>
                  <CardHeader
                    title="Simulated Attacker Response"
                    subtitle="How the attacker's progression reacts to the mitigation on the cloned twin state."
                  />
                  <div className="flex items-center gap-3 my-4">
                    <span className={`text-xs font-bold px-3 py-1.5 rounded-full uppercase tracking-wider ${
                      attackerResp.outcome === 'BLOCKED' ? 'bg-[var(--color-good)]/15 text-[var(--color-good)] border border-[var(--color-good)]/30' :
                      attackerResp.outcome === 'THROTTLED' ? 'bg-[var(--color-warn)]/15 text-[var(--color-warn)] border border-[var(--color-warn)]/30' :
                      'bg-[var(--color-bad)]/15 text-[var(--color-bad)] border border-[var(--color-bad)]/30'
                    }`}>
                      {attackerResp.outcome === 'BLOCKED' && <ShieldIcon size={14} className="inline mr-1" />}
                      {attackerResp.outcome === 'THROTTLED' && <AlertIcon size={14} className="inline mr-1" />}
                      ATTACK OUTCOME: {attackerResp.outcome}
                    </span>
                  </div>
                  <p className="text-sm text-[var(--color-ink)] leading-relaxed bg-[var(--color-accent-soft)]/40 p-4 rounded-xl border border-[var(--color-accent-soft)]">
                    {attackerResp.reason}
                  </p>
                </div>

                <div className="mt-4 pt-3 border-t border-[var(--color-accent-soft)] flex items-center justify-between text-xs text-[var(--color-ink-faint)]">
                  <span>Target Asset: <strong>{attackerResp.target_host_id}</strong></span>
                  <span>Port Affected: <strong>{attackerResp.affected_port ? `Port ${attackerResp.affected_port}` : 'All / ACL'}</strong></span>
                </div>
              </div>

              {/* Digital Twin Network State Card */}
              <div className="card p-6 flex flex-col justify-between">
                <div>
                  <CardHeader
                    title="Digital Twin Network State"
                    subtitle="Logical state representation of hosts, services, and active firewall rules."
                  />
                  {twinState && (
                    <div className="flex flex-col gap-3 my-3">
                      <div className="text-xs bg-[var(--color-accent-soft)] p-3 rounded-xl">
                        <div className="font-semibold text-[var(--color-ink)] mb-1">Simulated Hosts & Services</div>
                        <div className="grid grid-cols-2 gap-2 text-[var(--color-ink-faint)]">
                          {Object.values(twinState.hosts).map((h) => (
                            <div key={h.host_id} className="p-2 bg-white rounded-lg border border-[var(--color-accent-soft)]">
                              <div className="font-medium text-[var(--color-ink)]">{h.host_id}</div>
                              <div>IP: {h.ip_address}</div>
                              <div>Isolation: {h.is_isolated ? <span className="text-[var(--color-good)] font-bold">QUARANTINED</span> : 'Active'}</div>
                            </div>
                          ))}
                        </div>
                      </div>

                      {twinState.firewall_rules.length > 0 && (
                        <div className="text-xs bg-[var(--color-accent-soft)]/40 p-3 rounded-xl border border-[var(--color-accent-soft)]">
                          <div className="font-semibold text-[var(--color-ink)] mb-1">Active Firewall Rules ({twinState.firewall_rules.length})</div>
                          <div className="text-[var(--color-ink-faint)] flex flex-col gap-1 max-h-24 overflow-y-auto">
                            {twinState.firewall_rules.map((r, i) => (
                              <div key={i} className="font-mono text-[11px] bg-white px-2 py-1 rounded border border-[var(--color-accent-soft)]">
                                [{r.action}] {r.description}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                <div className="text-xs text-[var(--color-ink-faint)] pt-2 border-t border-[var(--color-accent-soft)] flex justify-between">
                  <span>Data Mode: {isLive ? 'Live Packet Telemetry' : 'Synthetic Data Generator'}</span>
                  <span>Extensible Adapters: Ready</span>
                </div>
              </div>
            </div>
          )}

          {/* Topology Path Prediction Card */}
          {pathPrediction && (
            <div className="card p-6">
              <CardHeader
                title="Topology Path Prediction & Graph Reachability"
                subtitle="Calculated attack transition graph based on host reachability and firewall policies."
              />

              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 my-4">
                <div className="p-3 rounded-xl bg-[var(--color-accent-soft)]/50 text-center">
                  <div className="text-xs text-[var(--color-ink-faint)]">Total Potential Paths</div>
                  <div className="text-lg font-bold text-[var(--color-ink)]">{pathPrediction.metrics.total_paths}</div>
                </div>
                <div className="p-3 rounded-xl bg-[var(--color-good)]/10 border border-[var(--color-good)]/30 text-center">
                  <div className="text-xs text-[var(--color-good)]">Blocked Paths</div>
                  <div className="text-lg font-bold text-[var(--color-good)]">{pathPrediction.metrics.blocked_paths_count}</div>
                </div>
                <div className="p-3 rounded-xl bg-[var(--color-warn)]/10 border border-[var(--color-warn)]/30 text-center">
                  <div className="text-xs text-[var(--color-warn)]">Remaining Paths</div>
                  <div className="text-lg font-bold text-[var(--color-warn)]">{pathPrediction.metrics.remaining_paths_count}</div>
                </div>
                <div className="p-3 rounded-xl bg-[var(--color-accent-soft)]/50 text-center">
                  <div className="text-xs text-[var(--color-ink-faint)]">Path Reduction</div>
                  <div className="text-lg font-bold text-[var(--color-accent)]">{pathPrediction.metrics.reduction_pct}%</div>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                {/* Blocked Paths */}
                <div className="p-4 rounded-xl bg-[var(--color-good)]/5 border border-[var(--color-good)]/20">
                  <div className="font-semibold text-[var(--color-good)] mb-2 flex items-center gap-1.5">
                    <ShieldIcon size={14} />
                    Blocked Attack Paths ({pathPrediction.blocked_paths.length})
                  </div>
                  {pathPrediction.blocked_paths.length > 0 ? (
                    <div className="flex flex-col gap-1.5 max-h-48 overflow-y-auto">
                      {pathPrediction.blocked_paths.map((p, i) => (
                        <div key={i} className="p-2 rounded-lg bg-white border border-[var(--color-good)]/30 text-[var(--color-ink)]">
                          <div className="font-medium text-[var(--color-good)]">{STAGE_LABELS[p.stage] ?? p.stage} ({p.tactic})</div>
                          <div className="text-[var(--color-ink-faint)] text-[11px]">{p.reason}</div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-[var(--color-ink-faint)]">No paths blocked by current state.</div>
                  )}
                </div>

                {/* Remaining Open Paths */}
                <div className="p-4 rounded-xl bg-[var(--color-warn)]/5 border border-[var(--color-warn)]/20">
                  <div className="font-semibold text-[var(--color-warn)] mb-2 flex items-center gap-1.5">
                    <AlertIcon size={14} />
                    Remaining Open Paths ({pathPrediction.remaining_paths.length})
                  </div>
                  {pathPrediction.remaining_paths.length > 0 ? (
                    <div className="flex flex-col gap-1.5 max-h-48 overflow-y-auto">
                      {pathPrediction.remaining_paths.map((p, i) => (
                        <div key={i} className="p-2 rounded-lg bg-white border border-[var(--color-warn)]/30 text-[var(--color-ink)]">
                          <div className="font-medium text-[var(--color-warn)]">{STAGE_LABELS[p.stage] ?? p.stage} ({p.tactic})</div>
                          <div className="text-[var(--color-ink-faint)] text-[11px]">{p.reason}</div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-[var(--color-good)] font-medium">All attack paths successfully blocked!</div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* LSTM Forecast Chart */}
          <div className="card p-6">
            <CardHeader
              title="Predicted Infiltration Probability — LSTM World Model Forecast"
              subtitle={`${result.host_id}, mitigation applied on twin state from window #${result.window_idx} onward (observed stage: ${STAGE_LABELS[result.true_stage ?? 'benign'] ?? result.true_stage})`}
            />
            <CounterfactualChart data={chartData} />
          </div>

          {verdict && (
            <div className={`card p-5 border-l-4 ${probMoved || actionChanged ? 'border-l-[var(--color-accent)]' : 'border-l-[var(--color-ink-faint)]'}`}>
              <div className="flex items-center justify-between gap-3 mb-1.5">
                <div className="text-xs font-semibold uppercase tracking-wide text-[var(--color-ink-faint)]">Simulation Verdict</div>
                <span className="text-xs font-medium px-2.5 py-1 rounded-full bg-[var(--color-accent-soft)] text-[var(--color-accent)] whitespace-nowrap">
                  {result.metrics.verdict} · {result.metrics.risk_reduction_pct.toFixed(1)}% risk reduction
                </span>
              </div>
              <p className="text-sm text-[var(--color-ink)] leading-relaxed">{verdict}</p>
            </div>
          )}

          {/* Action Divergence / Trajectory Path */}
          <div className="card p-6">
            <CardHeader
              title="What Changed in Predicted Trajectory"
              subtitle="Points where the predicted next action differs between the baseline and mitigated state."
            />
            {result.action_divergences.length ? (
              <div className="flex flex-col gap-2">
                {result.action_divergences.map((d, i) => (
                  <div key={i} className="flex items-center gap-2 text-sm bg-[var(--color-accent-soft)] rounded-xl px-4 py-3">
                    <span className="font-medium text-[var(--color-ink)]">t+{d.horizon}:</span>
                    <span className="text-[var(--color-bad)]">{STAGE_LABELS[d.without_mitigation_action] ?? d.without_mitigation_action}</span>
                    <span className="text-[var(--color-ink-faint)]">→</span>
                    <span className="text-[var(--color-good)] font-medium">{STAGE_LABELS[d.with_mitigation_action] ?? d.with_mitigation_action}</span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-sm text-[var(--color-ink-faint)] py-4 text-center flex items-center justify-center gap-2">
                <NetworkIcon size={16} />
                No predicted-action change — mitigation had minimal trajectory shift at this window.
              </div>
            )}
          </div>

          <div className="card p-6">
            <CardHeader
              title="Predicted Action Path — Without vs. With Mitigation"
              subtitle="Comparing the LSTM world model's most-likely action sequence step-by-step."
            />
            <TrajectoryPathDiagram
              withoutActions={result.without_mitigation.predicted_stage_per_horizon}
              withActions={result.with_mitigation.predicted_stage_per_horizon}
            />
          </div>
        </>
      )}

      <DefenseAdvisor hostId={selectedHost} atWindowIdx={windowIdx ?? undefined} title="Defense Advisor — Which Mitigation Should We Pick?" />
    </div>
  );
}
