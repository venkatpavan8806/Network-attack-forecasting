import { useEffect, useState } from 'react';
import { api, STAGE_COLORS, STAGE_LABELS } from '../api';
import type { DefenseAdvice } from '../types';
import CardHeader from './CardHeader';
import CounterfactualChart, { type CounterfactualPoint } from './CounterfactualChart';

const LIVE_REFRESH_MS = 10000;

const LEVEL_STYLE: Record<DefenseAdvice['risk_level'], { label: string; color: string }> = {
  act_now: { label: 'Act now', color: 'var(--color-bad)' },
  watch: { label: 'Watch closely', color: 'var(--color-warn)' },
  monitor: { label: 'Keep monitoring', color: 'var(--color-good)' },
};

const DISRUPTION_COLOR: Record<string, string> = {
  low: 'var(--color-good)',
  medium: 'var(--color-warn)',
  high: 'var(--color-bad)',
};

export default function DefenseAdvisor({
  hostId, atWindowIdx, title = 'Defense Advisor',
}: { hostId: string | null; atWindowIdx?: number; title?: string }) {
  const [advice, setAdvice] = useState<DefenseAdvice | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!hostId) return;
    let cancelled = false;
    const isLive = hostId.startsWith('live:');

    const load = (showSpinner: boolean) => {
      if (showSpinner) setLoading(true);
      api.defense(hostId, atWindowIdx)
        .then((r) => { if (!cancelled) { setAdvice(r); setError(null); } })
        .catch((e) => { if (!cancelled) { setAdvice(null); setError(e?.response?.data?.detail ?? String(e)); } })
        .finally(() => { if (!cancelled && showSpinner) setLoading(false); });
    };

    load(true);
    // live hosts: re-run the counterfactual ranking as new real windows arrive
    const timer = isLive ? window.setInterval(() => load(false), LIVE_REFRESH_MS) : null;
    return () => { cancelled = true; if (timer) window.clearInterval(timer); };
  }, [hostId, atWindowIdx]);

  if (!hostId) {
    return (
      <div className="card p-6">
        <CardHeader title={title} subtitle="select a host" />
        <div className="text-sm text-[var(--color-ink-faint)] py-6 text-center">not yet available</div>
      </div>
    );
  }

  const chartData: CounterfactualPoint[] = advice
    ? advice.trajectory_without.map((p, i) => ({
        step: `t+${i + 1}`,
        without: p,
        with_: advice.trajectory_with_recommended ? advice.trajectory_with_recommended[i] : null,
      }))
    : [];

  const level = advice ? LEVEL_STYLE[advice.risk_level] : null;

  return (
    <div className="card p-6">
      <CardHeader
        title={title}
        subtitle="Every mitigation is replayed through the trained world model and ranked by how much it lowers the predicted infiltration curve. Decision support only — nothing is applied to a network."
      />

      {loading && <div className="text-sm text-[var(--color-ink-faint)] py-8 text-center">ranking mitigations with the world model…</div>}
      {error && !loading && <div className="text-sm text-[var(--color-bad)] py-4">{error}</div>}

      {advice && level && !loading && (
        <div className="flex flex-col gap-6">
          <div className="flex flex-wrap items-center gap-3 rounded-xl px-4 py-3" style={{ background: `color-mix(in srgb, ${level.color} 14%, white)` }}>
            <span className="text-sm font-bold" style={{ color: level.color }}>{level.label}</span>
            <span className="text-sm text-[var(--color-ink)]">
              peak predicted infiltration <strong>{(advice.peak_probability * 100).toFixed(1)}%</strong> within {advice.horizon_windows} windows
            </span>
            {advice.expected_stage && (
              <span className="flex items-center gap-1.5 text-sm text-[var(--color-ink)]">
                · furthest expected stage:
                <span className="w-2 h-2 rounded-full" style={{ background: STAGE_COLORS[advice.expected_stage] ?? '#999' }} />
                <strong>{STAGE_LABELS[advice.expected_stage] ?? advice.expected_stage}</strong>
              </span>
            )}
            <span className="text-xs text-[var(--color-ink-dim)] ml-auto">{advice.host_id} · window #{advice.window_idx}</span>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="flex flex-col gap-3">
              <div className="text-xs text-[var(--color-ink-faint)]">Recommended action</div>
              {advice.recommended ? (
                <div className="rounded-xl border border-[var(--color-accent-soft)] p-4">
                  <div className="flex flex-wrap items-center gap-2 mb-1.5">
                    <span className="font-semibold text-[var(--color-ink)]">{advice.recommended.label}</span>
                    <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full" style={{ color: DISRUPTION_COLOR[advice.recommended.disruption], background: `color-mix(in srgb, ${DISRUPTION_COLOR[advice.recommended.disruption]} 15%, white)` }}>
                      {advice.recommended.disruption} disruption
                    </span>
                    <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-[var(--color-accent-soft)] text-[var(--color-accent)]">
                      −{(advice.recommended.mean_reduction * 100).toFixed(1)} pts mean probability
                    </span>
                  </div>
                  <p className="text-sm text-[var(--color-ink-dim)]">{advice.recommendation_reason}</p>
                </div>
              ) : (
                <div className="rounded-xl bg-[var(--color-accent-soft)]/50 p-4 text-sm text-[var(--color-ink-dim)]">
                  No mitigation recommended. {advice.recommendation_reason}
                </div>
              )}
              {advice.trajectory_with_recommended && (
                <div>
                  <div className="text-xs text-[var(--color-ink-faint)] mb-1">Predicted trajectory with vs. without the recommended action</div>
                  <CounterfactualChart data={chartData} />
                </div>
              )}
            </div>

            <div className="flex flex-col gap-4">
              <div>
                <div className="text-xs text-[var(--color-ink-faint)] mb-1">
                  Manual playbook{advice.playbook.attack_mapping?.technique_id ? ` — ATT&CK ${advice.playbook.attack_mapping.technique_id}` : ''}
                </div>
                <p className="text-sm text-[var(--color-ink)]">{advice.playbook.summary}</p>
              </div>
              {advice.playbook.mitigations.length > 0 && (
                <div className="flex flex-col gap-1.5">
                  {advice.playbook.mitigations.map((m, i) => (
                    <div key={i} className="text-xs flex gap-2">
                      <span className="shrink-0 font-semibold text-[var(--color-accent)] w-12">{m.id ?? 'practice'}</span>
                      <span className="text-[var(--color-ink)]"><strong>{m.name}</strong> — <span className="text-[var(--color-ink-dim)]">{m.action}</span></span>
                    </div>
                  ))}
                </div>
              )}
              <ol className="list-decimal pl-5 text-sm text-[var(--color-ink)] flex flex-col gap-1">
                {advice.playbook.analyst_steps.map((s, i) => <li key={i}>{s}</li>)}
              </ol>
            </div>
          </div>

          <div className="overflow-x-auto">
            <div className="text-xs text-[var(--color-ink-faint)] mb-2">Evidence — every mitigation the world model tried, best first</div>
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-[var(--color-ink-faint)] text-xs uppercase tracking-wide">
                  <th className="py-2 pr-4 font-medium">Mitigation</th>
                  <th className="py-2 pr-4 font-medium">Mean prob. before → after</th>
                  <th className="py-2 pr-4 font-medium">Reduction</th>
                  <th className="py-2 pr-4 font-medium">Peak reduction</th>
                  <th className="py-2 pr-4 font-medium">Stages changed</th>
                  <th className="py-2 pr-4 font-medium">Disruption</th>
                </tr>
              </thead>
              <tbody>
                {advice.evidence.map((e) => (
                  <tr key={e.id} className={`border-t border-[var(--color-accent-soft)] ${advice.recommended?.id === e.id ? 'bg-[var(--color-accent-soft)]/50' : ''}`}>
                    <td className="py-2.5 pr-4 font-medium text-[var(--color-ink)]">
                      {e.label}{advice.recommended?.id === e.id && <span className="ml-2 text-[10px] text-[var(--color-accent)]">recommended</span>}
                    </td>
                    <td className="py-2.5 pr-4 text-[var(--color-ink-dim)]">{(e.mean_probability_without * 100).toFixed(1)}% → {(e.mean_probability_with * 100).toFixed(1)}%</td>
                    <td className={`py-2.5 pr-4 font-semibold ${e.effective ? 'text-[var(--color-good)]' : 'text-[var(--color-ink-faint)]'}`}>
                      {e.mean_reduction >= 0 ? '−' : '+'}{Math.abs(e.mean_reduction * 100).toFixed(1)} pts
                    </td>
                    <td className="py-2.5 pr-4 text-[var(--color-ink-dim)]">{(e.peak_reduction * 100).toFixed(1)} pts</td>
                    <td className="py-2.5 pr-4 text-[var(--color-ink-dim)]">{e.stages_changed}/{advice.horizon_windows}</td>
                    <td className="py-2.5 pr-4" style={{ color: DISRUPTION_COLOR[e.disruption] }}>{e.disruption}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <p className="text-[11px] text-[var(--color-ink-faint)]">
            {advice.method.limits} Playbook text and disruption tiers are fixed reference content; the probabilities above are computed live.
            {advice.host_id.startsWith('live:') && ' Live host: re-ranked every 10 s.'}
          </p>
        </div>
      )}
    </div>
  );
}
