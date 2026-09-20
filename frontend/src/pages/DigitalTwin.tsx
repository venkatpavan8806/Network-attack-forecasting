import { useEffect, useState } from 'react';
import { api, STAGE_COLORS, STAGE_LABELS } from '../api';
import type { CounterfactualResponse, HostTimelineEntry, MitigationInfo } from '../types';
import CardHeader from '../components/CardHeader';
import CounterfactualChart, { type CounterfactualPoint } from '../components/CounterfactualChart';
import DefenseAdvisor from '../components/DefenseAdvisor';
import { NetworkIcon } from '../icons';

export default function DigitalTwin({ selectedHost, onSelectHost }: { selectedHost: string | null; onSelectHost: (h: string) => void }) {
  const [hosts, setHosts] = useState<string[]>([]);
  const [mitigations, setMitigations] = useState<MitigationInfo[]>([]);
  const [mitigationId, setMitigationId] = useState('block_ssh');
  const [timeline, setTimeline] = useState<HostTimelineEntry[]>([]);
  const [windowIdx, setWindowIdx] = useState<number | null>(null);
  const [result, setResult] = useState<CounterfactualResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.hosts().then((h) => {
      const attackHosts = h.filter((x) => x.startsWith('attack-host'));
      setHosts(attackHosts.length ? attackHosts : h);
      if (!selectedHost && (attackHosts.length || h.length)) onSelectHost((attackHosts[0] ?? h[0]));
    }).catch(() => {});
    api.mitigations().then(setMitigations).catch(() => {});
  }, []);

  useEffect(() => {
    if (!selectedHost) return;
    api.hostTimeline(selectedHost).then((t) => {
      setTimeline(t);
      // default to the last window of the first non-benign action run, so the
      // what-if starts right when something interesting was first observed
      const firstNonBenign = t.find((r) => r.true_stage !== 'benign');
      setWindowIdx(firstNonBenign ? firstNonBenign.window_idx - 1 : t[Math.max(0, t.length - 1)]?.window_idx ?? null);
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

  // group the timeline into contiguous action segments for the "jump to" picker
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

  return (
    <div className="flex flex-col gap-6">
      <div className="card p-6">
        <CardHeader
          title="Digital Twin — Model-Based What-If"
          subtitle="Not a live simulated network. This mutates the real observed feature window the way a mitigation would change observable traffic, then replays it through the trained world model. Decision support only — nothing here touches a real network."
        />

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-2">
          <div>
            <label className="text-xs text-[var(--color-ink-faint)] mb-1.5 block">Host</label>
            <select
              value={selectedHost ?? ''}
              onChange={(e) => onSelectHost(e.target.value)}
              className="w-full px-3 py-2 rounded-xl border border-[var(--color-accent-soft)] text-sm bg-white text-[var(--color-ink)]"
            >
              {hosts.map((h) => <option key={h} value={h}>{h}</option>)}
            </select>
          </div>
          <div>
            <label className="text-xs text-[var(--color-ink-faint)] mb-1.5 block">Apply mitigation starting at</label>
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
          </div>
          <div>
            <label className="text-xs text-[var(--color-ink-faint)] mb-1.5 block">Mitigation</label>
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
          <p className="text-xs text-[var(--color-ink-faint)] mt-2">
            {mitigations.find((m) => m.id === mitigationId)?.description}
          </p>
        )}
      </div>

      {loading && <div className="card p-10 text-center text-sm text-[var(--color-ink-faint)]">running counterfactual rollout…</div>}
      {error && <div className="card p-6 text-sm text-[var(--color-bad)]">{error}</div>}

      {result && !loading && (
        <>
          <div className="card p-6">
            <CardHeader
              title="Predicted Infiltration Probability — With vs. Without Mitigation"
              subtitle={`${result.host_id}, mitigation applied from window #${result.window_idx} onward (real ground truth at that point: ${STAGE_LABELS[result.true_stage ?? 'benign'] ?? result.true_stage})`}
            />
            <CounterfactualChart data={chartData} />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="card p-6">
              <CardHeader title="Predicted Action Trajectory — Without Mitigation" />
              <div className="flex flex-col gap-2">
                {result.without_mitigation.predicted_stage_per_horizon.map((a, i) => (
                  <div key={i} className="flex items-center gap-2 text-sm">
                    <span className="text-xs text-[var(--color-ink-faint)] w-10 shrink-0">t+{i + 1}</span>
                    <span className="w-2 h-2 rounded-full shrink-0" style={{ background: STAGE_COLORS[a] }} />
                    <span className="text-[var(--color-ink)]">{STAGE_LABELS[a] ?? a}</span>
                  </div>
                ))}
              </div>
            </div>
            <div className="card p-6">
              <CardHeader title="Predicted Action Trajectory — With Mitigation" />
              <div className="flex flex-col gap-2">
                {result.with_mitigation.predicted_stage_per_horizon.map((a, i) => {
                  const diverged = result.action_divergences.some((d) => d.horizon === i + 1);
                  return (
                    <div key={i} className="flex items-center gap-2 text-sm">
                      <span className="text-xs text-[var(--color-ink-faint)] w-10 shrink-0">t+{i + 1}</span>
                      <span className="w-2 h-2 rounded-full shrink-0" style={{ background: STAGE_COLORS[a] }} />
                      <span className={diverged ? 'font-semibold text-[var(--color-accent)]' : 'text-[var(--color-ink)]'}>{STAGE_LABELS[a] ?? a}</span>
                      {diverged && <span className="text-[10px] text-[var(--color-accent)] bg-[var(--color-accent-soft)] px-1.5 py-0.5 rounded-full">changed</span>}
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          <div className="card p-6">
            <CardHeader
              title="What Changed"
              subtitle="Points where the predicted next action differs between the two rollouts"
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
                no predicted-action change at this horizon for this mitigation — the model's aggregate infiltration
                probability may still shift slightly, but the most-likely next action stays the same
              </div>
            )}
          </div>
        </>
      )}

      <DefenseAdvisor hostId={selectedHost} atWindowIdx={windowIdx ?? undefined} title="Defense Advisor — Which Mitigation Should We Pick?" />
    </div>
  );
}
