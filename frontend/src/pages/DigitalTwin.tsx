import { useEffect, useState } from 'react';
import { api, STAGE_LABELS } from '../api';
import type { CounterfactualResponse, HostTimelineEntry, MitigationInfo } from '../types';
import CardHeader from '../components/CardHeader';
import CounterfactualChart, { type CounterfactualPoint } from '../components/CounterfactualChart';
import DefenseAdvisor from '../components/DefenseAdvisor';
import TrajectoryPathDiagram from '../components/TrajectoryPathDiagram';
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

  const [liveHosts, setLiveHosts] = useState<string[]>([]);
  const isLive = selectedHost?.startsWith('live:') ?? false;

  useEffect(() => {
    api.hosts().then((h) => {
      const attackHosts = h.filter((x) => x.startsWith('attack-host'));
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
      // default to the last window of the first non-benign action run, so the
      // what-if starts right when something interesting was first observed
      const firstNonBenign = t.find((r) => r.true_stage !== 'benign');
      setWindowIdx(firstNonBenign ? firstNonBenign.window_idx - 1 : t[Math.max(0, t.length - 1)]?.window_idx ?? null);
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

  // plain-language verdict: what actually changed, in the two ways it can.
  // probMoved is driven by the same mean-based risk_reduction_pct the
  // backend's own verdict badge uses (>5%, its "Neutral" cutoff) so the two
  // never read as disagreeing about the same result.
  const probDeltaPct = result
    ? Math.max(...result.without_mitigation.infiltration_probs.map((p, i) => (p - result.with_mitigation.infiltration_probs[i]) * 100))
    : 0;
  const probMoved = (result?.metrics.risk_reduction_pct ?? 0) > 5;
  const actionChanged = (result?.action_divergences.length ?? 0) > 0;

  let verdict: string | null = null;
  if (result) {
    if (probMoved && actionChanged) {
      verdict = `This mitigation both lowered the predicted infiltration probability (by up to ${probDeltaPct.toFixed(1)} points) and changed what the model expects to happen next.`;
    } else if (probMoved) {
      verdict = `This mitigation lowered the predicted infiltration probability by up to ${probDeltaPct.toFixed(1)} points — the model is measurably less alarmed, though it still expects the same next action.`;
    } else if (actionChanged) {
      verdict = `This mitigation barely moved the overall alarm level — the model is already confident something is wrong at this point in the attack, and one mitigation on its own doesn't undo that. What it DID change is which action the model expects next (see below). That's often the more useful signal once an attack is already this far along.`;
    } else {
      verdict = `This mitigation had no measurable effect here — the model's forecast is the same with or without it. This usually means the attack was already too far along, or too broadly confirmed by other signals, for one targeted mitigation to change the picture. Try applying it earlier (pick an earlier "apply mitigation starting at" option above) to see a bigger effect.`;
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="card p-6">
        <CardHeader
          title="Digital Twin — Model-Based What-If"
          subtitle="A safe sandbox to test a defense before applying it for real — not a live simulated network."
        />
        <div className="rounded-xl bg-[var(--color-accent-soft)] p-4 mb-4 text-xs text-[var(--color-ink)] leading-relaxed">
          <strong>World model = the brain.</strong> The LSTM (rollout) predicts cause-and-effect over time — given port 22 is open, what's the probability the attacker brute-forces SSH next, versus something else?
          <br />
          <strong>Digital twin = the sandbox.</strong> This page asks that same model "what if we acted now?" — e.g. an attacker finds SSH open and starts brute-forcing it; instead of letting every attempt through, we rate-limit or block the port (concretely: cutting allowed connection attempts down to a handful, same idea as your reviewer's "reduce password attempts to 2 or 3") and see how the forecast changes, <em>before</em> touching the real network.
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-2">
          <div>
            <label className="text-xs text-[var(--color-ink-faint)] mb-1.5 block">Host</label>
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
            <label className="text-xs text-[var(--color-ink-faint)] mb-1.5 block">Apply mitigation starting at</label>
            {isLive ? (
              <div className="w-full px-3 py-2 rounded-xl border border-[var(--color-accent-soft)] text-sm bg-[var(--color-accent-soft)]/40 text-[var(--color-ink-dim)]">
                now (live hosts only keep their most recent 8 windows)
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
        <p className="text-xs text-[var(--color-ink-faint)] mt-2">
          Tip: applying a mitigation earlier in the attack (e.g. right after Port Scan, before Brute-Force starts) tends to shift both the probability and the predicted next action. Applying it very late (e.g. during C2 or Exfiltration) mostly only shifts which action comes next, since the model is already highly confident something is wrong by then.
        </p>
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

          {verdict && (
            <div className={`card p-5 border-l-4 ${probMoved || actionChanged ? 'border-l-[var(--color-accent)]' : 'border-l-[var(--color-ink-faint)]'}`}>
              <div className="flex items-center justify-between gap-3 mb-1.5">
                <div className="text-xs font-semibold uppercase tracking-wide text-[var(--color-ink-faint)]">In plain terms</div>
                <span className="text-xs font-medium px-2.5 py-1 rounded-full bg-[var(--color-accent-soft)] text-[var(--color-accent)] whitespace-nowrap">
                  {result.metrics.verdict} · {result.metrics.risk_reduction_pct.toFixed(1)}% mean risk reduction
                </span>
              </div>
              <p className="text-sm text-[var(--color-ink)] leading-relaxed">{verdict}</p>
            </div>
          )}

          <div className="card p-6">
            <CardHeader
              title="What Changed"
              subtitle="Points where the predicted next action differs between the two rollouts — this is the main thing to look at when the probability lines above overlap"
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
                no predicted-action change either — this mitigation made no measurable difference at this point
              </div>
            )}
          </div>

          <div className="card p-6">
            <CardHeader
              title="Predicted Action Path — Without vs. With Mitigation"
              subtitle="Each node is the model's most-likely next action at that step. Where the two paths agree, the connector is a faint dashed line; where they split, it's solid purple — that's exactly where the mitigation changed the outcome."
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
