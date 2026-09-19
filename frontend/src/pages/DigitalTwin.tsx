import { useEffect, useState } from 'react';
import { api, STAGE_COLORS, STAGE_LABELS } from '../api';
import type { CounterfactualResponse, HostTimelineEntry, MitigationInfo } from '../types';
import CardHeader from '../components/CardHeader';
import CounterfactualChart, { type CounterfactualPoint } from '../components/CounterfactualChart';
import { NetworkIcon, ShieldIcon, ArrowDownIcon, AlertIcon, ClockIcon } from '../icons';

export default function DigitalTwin({ selectedHost, onSelectHost }: { selectedHost: string | null; onSelectHost: (h: string) => void }) {
  const [hosts, setHosts] = useState<string[]>([]);
  const [mitigations, setMitigations] = useState<MitigationInfo[]>([]);
  const [mitigationId, setMitigationId] = useState('isolate_host');
  const [timeline, setTimeline] = useState<HostTimelineEntry[]>([]);
  const [windowIdx, setWindowIdx] = useState<number | null>(null);
  const [result, setResult] = useState<CounterfactualResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [categoryFilter, setCategoryFilter] = useState<string>('all');

  useEffect(() => {
    api.hosts().then((h) => {
      const attackHosts = h.filter((x) => x.startsWith('attack-host'));
      setHosts(attackHosts.length ? attackHosts : h);
      if (!selectedHost && (attackHosts.length || h.length)) onSelectHost((attackHosts[0] ?? h[0]));
    }).catch(() => {});
    api.mitigations().then((m: MitigationInfo[]) => {
      setMitigations(m);
    }).catch(() => {});
  }, []);

  useEffect(() => {
    if (!selectedHost) return;
    api.hostTimeline(selectedHost).then((t) => {
      setTimeline(t);
      // default to the first non-benign window where the attack actively starts
      const firstNonBenign = t.find((r) => r.true_stage !== 'benign');
      setWindowIdx(firstNonBenign ? firstNonBenign.window_idx : t[Math.max(0, t.length - 1)]?.window_idx ?? null);
    }).catch(() => {});
  }, [selectedHost]);

  async function runWhatIf() {
    if (!selectedHost || windowIdx == null) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.counterfactual(selectedHost, mitigationId, windowIdx);
      setResult(res);
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? String(e));
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (selectedHost && windowIdx != null) runWhatIf();
  }, [selectedHost, windowIdx, mitigationId]);

  // Group timeline into contiguous attack stage segments
  const segments: { action: string; startWindow: number; endWindow: number }[] = [];
  for (const row of timeline) {
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

  const categories = ['all', ...Array.from(new Set(mitigations.map((m) => m.category || 'General').filter(Boolean)))];
  const filteredMitigations = categoryFilter === 'all'
    ? mitigations
    : mitigations.filter((m) => (m.category || 'General') === categoryFilter);

  const riskRedPct = result?.risk_reduction_pct ?? 0;
  const currentHostStage = timeline.find((r) => r.window_idx === windowIdx)?.true_stage ?? result?.true_stage ?? 'unknown';

  return (
    <div className="flex flex-col gap-6">
      {/* Control Panel */}
      <div className="card p-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <ShieldIcon size={20} className="text-[var(--color-accent)]" />
              <h2 className="text-xl font-bold text-[var(--color-ink)]">Digital Twin — Counterfactual Simulator</h2>
              <span className="text-xs px-2.5 py-0.5 rounded-full bg-[var(--color-accent-soft)] text-[var(--color-accent)] font-semibold">
                Autonomous Prototype
              </span>
            </div>
            <p className="text-xs text-[var(--color-ink-faint)] max-w-3xl">
              Model-based "what-if" engine: replays defender interventions against the LSTM world model to forecast trajectory divergence, quantify infiltration risk reduction, and verify attack containment.
            </p>
          </div>
        </div>

        {/* Filters and Selectors */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          <div>
            <label className="text-xs font-medium text-[var(--color-ink-faint)] mb-1.5 flex items-center justify-between">
              <span>Target Network Host</span>
              <span className="text-[10px] text-[var(--color-accent)] font-mono">{hosts.length} simulated hosts</span>
            </label>
            <select
              value={selectedHost ?? ''}
              onChange={(e) => onSelectHost(e.target.value)}
              className="w-full px-3 py-2 rounded-xl border border-[var(--color-accent-soft)] text-sm bg-white text-[var(--color-ink)] focus:outline-none focus:ring-2 focus:ring-[var(--color-accent)]"
            >
              {hosts.map((h) => (
                <option key={h} value={h}>{h}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="text-xs font-medium text-[var(--color-ink-faint)] mb-1.5 flex items-center justify-between">
              <span>Intervention Point (Timeline)</span>
              <span className="text-[10px] px-1.5 py-0.2 rounded" style={{ background: STAGE_COLORS[currentHostStage] + '25', color: STAGE_COLORS[currentHostStage] }}>
                {STAGE_LABELS[currentHostStage] ?? currentHostStage}
              </span>
            </label>
            <select
              value={windowIdx ?? ''}
              onChange={(e) => setWindowIdx(Number(e.target.value))}
              className="w-full px-3 py-2 rounded-xl border border-[var(--color-accent-soft)] text-sm bg-white text-[var(--color-ink)] focus:outline-none focus:ring-2 focus:ring-[var(--color-accent)]"
            >
              {segments.map((s) => (
                <option key={s.startWindow} value={s.startWindow}>
                  Window #{s.startWindow} ({STAGE_LABELS[s.action] ?? s.action})
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="text-xs font-medium text-[var(--color-ink-faint)] mb-1.5 flex items-center justify-between">
              <span>Defender Mitigation Policy</span>
              <span className="text-[10px] text-[var(--color-ink-faint)]">Category: {mitigations.find((m) => m.id === mitigationId)?.category ?? 'General'}</span>
            </label>
            <select
              value={mitigationId}
              onChange={(e) => setMitigationId(e.target.value)}
              className="w-full px-3 py-2 rounded-xl border border-[var(--color-accent-soft)] text-sm bg-white text-[var(--color-ink)] focus:outline-none focus:ring-2 focus:ring-[var(--color-accent)]"
            >
              {filteredMitigations.map((m) => (
                <option key={m.id} value={m.id}>{m.label}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Category Pill Bar */}
        <div className="flex items-center gap-1.5 flex-wrap border-t border-[var(--color-accent-soft)] pt-3 mt-1">
          <span className="text-xs text-[var(--color-ink-faint)] mr-2">Category:</span>
          {categories.map((cat) => (
            <button
              key={cat}
              onClick={() => setCategoryFilter(cat)}
              className={`text-xs px-3 py-1 rounded-full capitalize transition font-medium ${
                categoryFilter === cat
                  ? 'bg-[var(--color-accent)] text-white shadow-sm'
                  : 'bg-[var(--color-page)] text-[var(--color-ink-dim)] hover:bg-[var(--color-accent-soft)]'
              }`}
            >
              {cat}
            </button>
          ))}
        </div>

        {result && (
          <div className="mt-3 p-3 bg-[var(--color-page)] rounded-xl border border-[var(--color-accent-soft)] text-xs text-[var(--color-ink-dim)] flex items-start gap-2">
            <ShieldIcon size={16} className="text-[var(--color-accent)] shrink-0 mt-0.5" />
            <div>
              <span className="font-semibold text-[var(--color-ink)]">{mitigations.find((m) => m.id === mitigationId)?.label}: </span>
              {mitigations.find((m) => m.id === mitigationId)?.description}
            </div>
          </div>
        )}
      </div>

      {loading && (
        <div className="card p-12 text-center text-sm text-[var(--color-ink-faint)] flex flex-col items-center justify-center gap-3">
          <div className="w-8 h-8 border-3 border-[var(--color-accent)] border-t-transparent rounded-full animate-spin" />
          <span>Simulating counterfactual dynamics with LSTM world model…</span>
        </div>
      )}

      {error && <div className="card p-6 text-sm text-[var(--color-bad)] bg-red-50 border border-red-200">{error}</div>}

      {result && !loading && (
        <>
          {/* Simulation Outcome KPI Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="card p-4 flex flex-col">
              <span className="text-xs text-[var(--color-ink-faint)] mb-1">Infiltration Risk Reduction</span>
              <div className="flex items-center gap-2">
                <span className={`text-2xl font-extrabold ${riskRedPct > 30 ? 'text-[var(--color-good)]' : 'text-[var(--color-ink)]'}`}>
                  {riskRedPct > 0 ? `-${riskRedPct.toFixed(1)}%` : '0.0%'}
                </span>
                {riskRedPct > 0 && <ArrowDownIcon size={18} className="text-[var(--color-good)]" />}
              </div>
              <span className="text-[11px] text-[var(--color-ink-faint)] mt-1">Mean probability delta over rollout</span>
            </div>

            <div className="card p-4 flex flex-col">
              <span className="text-xs text-[var(--color-ink-faint)] mb-1">Counterfactual Verdict</span>
              <div className="flex items-center gap-1.5">
                <span className={`text-sm font-bold ${
                  result.verdict?.includes('High') ? 'text-[var(--color-good)]' :
                  result.verdict?.includes('Moderate') ? 'text-[var(--color-accent)]' : 'text-[var(--color-ink-dim)]'
                }`}>
                  {result.verdict ?? 'Evaluated'}
                </span>
              </div>
              <span className="text-[11px] text-[var(--color-ink-faint)] mt-1">World model stability assessment</span>
            </div>

            <div className="card p-4 flex flex-col">
              <span className="text-xs text-[var(--color-ink-faint)] mb-1">Peak Risk Comparison</span>
              <div className="flex items-baseline gap-2">
                <span className="text-lg font-bold text-[var(--color-bad)]">
                  {Math.round(Math.max(...result.without_mitigation.infiltration_probs) * 100)}%
                </span>
                <span className="text-xs text-[var(--color-ink-faint)]">→</span>
                <span className="text-lg font-bold text-[var(--color-good)]">
                  {Math.round(Math.max(...result.with_mitigation.infiltration_probs) * 100)}%
                </span>
              </div>
              <span className="text-[11px] text-[var(--color-ink-faint)] mt-1">Max probability across 6 steps</span>
            </div>

            <div className="card p-4 flex flex-col">
              <span className="text-xs text-[var(--color-ink-faint)] mb-1">Prevented Attack Horizons</span>
              <div className="flex items-center gap-2">
                <span className="text-2xl font-extrabold text-[var(--color-accent)]">
                  {result.action_divergences.length} / {result.horizon_windows}
                </span>
                <ClockIcon size={16} className="text-[var(--color-accent)]" />
              </div>
              <span className="text-[11px] text-[var(--color-ink-faint)] mt-1">Divergent next-action steps</span>
            </div>
          </div>

          {/* Probability Trajectory Chart */}
          <div className="card p-6">
            <CardHeader
              title="Predicted Infiltration Probability — With vs. Without Mitigation"
              subtitle={`${result.host_id} (Active Window #${result.window_idx}, Ground Truth: ${STAGE_LABELS[result.true_stage ?? 'benign'] ?? result.true_stage}) — Autoregressive 6-Step Rollout`}
            />
            <CounterfactualChart data={chartData} />
          </div>

          {/* Action Trajectory Comparison Side-by-Side */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="card p-6">
              <CardHeader
                title="Predicted Trajectory — Without Mitigation"
                subtitle="Autonomous escalation forecast without defender intervention"
              />
              <div className="flex flex-col gap-2 mt-2">
                {result.without_mitigation.predicted_stage_per_horizon.map((a, i) => (
                  <div key={i} className="flex items-center justify-between p-2.5 rounded-xl bg-[var(--color-page)] text-sm border border-[var(--color-accent-soft)]/50">
                    <div className="flex items-center gap-2.5">
                      <span className="text-xs font-mono font-semibold text-[var(--color-ink-faint)] w-9">t+{i + 1}</span>
                      <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: STAGE_COLORS[a] }} />
                      <span className="font-medium text-[var(--color-ink)]">{STAGE_LABELS[a] ?? a}</span>
                    </div>
                    <span className="text-xs font-mono font-semibold text-[var(--color-bad)]">
                      {(result.without_mitigation.infiltration_probs[i] * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>

            <div className="card p-6">
              <CardHeader
                title="Predicted Trajectory — Under Sustained Mitigation"
                subtitle={`Simulated trajectory under "${result.mitigation.label}"`}
              />
              <div className="flex flex-col gap-2 mt-2">
                {result.with_mitigation.predicted_stage_per_horizon.map((a, i) => {
                  const diverged = result.action_divergences.some((d) => d.horizon === i + 1);
                  const isBenign = a === 'benign';
                  return (
                    <div
                      key={i}
                      className={`flex items-center justify-between p-2.5 rounded-xl text-sm border transition ${
                        diverged
                          ? 'bg-emerald-50/70 border-emerald-200'
                          : 'bg-[var(--color-page)] border-[var(--color-accent-soft)]/50'
                      }`}
                    >
                      <div className="flex items-center gap-2.5">
                        <span className="text-xs font-mono font-semibold text-[var(--color-ink-faint)] w-9">t+{i + 1}</span>
                        <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: STAGE_COLORS[a] }} />
                        <span className={`font-medium ${diverged ? 'text-[var(--color-good)] font-semibold' : 'text-[var(--color-ink)]'}`}>
                          {STAGE_LABELS[a] ?? a}
                        </span>
                        {diverged && (
                          <span className="text-[10px] font-bold text-emerald-700 bg-emerald-100 px-2 py-0.5 rounded-full">
                            THWARTED
                          </span>
                        )}
                      </div>
                      <span className={`text-xs font-mono font-semibold ${isBenign ? 'text-[var(--color-good)]' : 'text-[var(--color-ink-dim)]'}`}>
                        {(result.with_mitigation.infiltration_probs[i] * 100).toFixed(1)}%
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Action Divergence Impact Table */}
          <div className="card p-6">
            <CardHeader
              title="Intervention Proof — Prevented Escalations"
              subtitle="Step-by-step verification of attack stages averted by the digital twin mitigation"
            />
            {result.action_divergences.length ? (
              <div className="flex flex-col gap-2.5 mt-2">
                {result.action_divergences.map((d, i) => (
                  <div key={i} className="flex items-center justify-between p-3.5 bg-gradient-to-r from-emerald-50/60 to-white rounded-xl border border-emerald-100 text-sm">
                    <div className="flex items-center gap-3">
                      <span className="font-mono font-bold text-xs bg-emerald-100 text-emerald-800 px-2.5 py-1 rounded-md">
                        Step t+{d.horizon}
                      </span>
                      <span className="line-through text-[var(--color-bad)] font-medium">
                        {STAGE_LABELS[d.without_mitigation_action] ?? d.without_mitigation_action}
                      </span>
                      <span className="text-[var(--color-ink-faint)]">→</span>
                      <span className="text-[var(--color-good)] font-bold">
                        {STAGE_LABELS[d.with_mitigation_action] ?? d.with_mitigation_action}
                      </span>
                    </div>
                    <span className="text-xs font-semibold text-emerald-700 bg-emerald-100/80 px-2.5 py-0.5 rounded-full">
                      Attack Averted
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-sm text-[var(--color-ink-faint)] py-6 text-center flex flex-col items-center justify-center gap-2">
                <NetworkIcon size={24} className="text-[var(--color-ink-faint)]" />
                <span>The model's predicted most likely action remains unchanged at this horizon for this mitigation.</span>
                <span className="text-xs">Overall infiltration confidence may still experience subtle probability shifts.</span>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}

