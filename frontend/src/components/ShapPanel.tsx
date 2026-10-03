import { useEffect, useState } from 'react';
import { api } from '../api';
import type { ShapResponse } from '../types';
import CardHeader from './CardHeader';

const LIVE_REFRESH_MS = 10000;

function fmtValue(v: number): string {
  const a = Math.abs(v);
  if (a >= 1000) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  return v.toFixed(3);
}

export default function ShapPanel({ hostId, atWindowIdx }: { hostId: string | null; atWindowIdx?: number }) {
  const [data, setData] = useState<ShapResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!hostId) return;
    let cancelled = false;
    const isLive = hostId.startsWith('live:');

    const load = (showSpinner: boolean) => {
      if (showSpinner) setLoading(true);
      api.shap(hostId, atWindowIdx)
        .then((r) => { if (!cancelled) { setData(r); setError(null); } })
        .catch((e) => { if (!cancelled) { setData(null); setError(e?.response?.data?.detail ?? String(e)); } })
        .finally(() => { if (!cancelled && showSpinner) setLoading(false); });
    };

    load(true);
    // live-capture hosts change every window, so keep the explanation fresh
    const timer = isLive ? window.setInterval(() => load(false), LIVE_REFRESH_MS) : null;
    return () => { cancelled = true; if (timer) window.clearInterval(timer); };
  }, [hostId, atWindowIdx]);

  if (!hostId) {
    return (
      <div className="card p-6">
        <CardHeader title="SHAP vs. Attention — Two Explanations, One Moment" subtitle="select a host" />
        <div className="text-sm text-[var(--color-ink-faint)] py-8 text-center">not yet available — pick a host in Forecasts</div>
      </div>
    );
  }

  const maxAbsShap = data ? Math.max(...data.shap.contributions.map((c) => Math.abs(c.shap_value)), 1e-6) : 1;
  const maxAbsLstm = data ? Math.max(...data.lstm.top_contributors.map((c) => Math.abs(c.contribution_score)), 1e-9) : 1;
  const sharedSet = new Set(data?.agreement.shared_features ?? []);

  return (
    <div className="card p-6">
      <CardHeader
        title="SHAP vs. Attention — Two Explanations, One Moment"
        subtitle={data
          ? `${data.host_id} — window ${data.window_idx}. SHAP explains the simple baseline (current window only); attention + saliency explains the LSTM (last 8 windows).`
          : 'computing…'}
      />

      {loading && <div className="text-sm text-[var(--color-ink-faint)] py-8 text-center">computing SHAP values…</div>}
      {error && !loading && <div className="text-sm text-[var(--color-bad)] py-4">{error}</div>}

      {data && !loading && (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
            <div className="bg-[var(--color-accent-soft)] rounded-xl p-3">
              <div className="text-xs text-[var(--color-ink-dim)]">Baseline says (current window)</div>
              <div className="text-lg font-bold text-[var(--color-ink)]">{(data.shap.model_probability * 100).toFixed(1)}% malicious</div>
              <div className="text-[10px] text-[var(--color-ink-faint)] mt-0.5">
                SHAP re-adds to {(data.shap.baseline_probability * 100).toFixed(1)}% — {Math.abs(data.shap.baseline_probability - data.shap.model_probability) < 0.005 ? 'matches the model ✓' : 'mismatch'}
              </div>
            </div>
            <div className="bg-[var(--color-accent-soft)] rounded-xl p-3">
              <div className="text-xs text-[var(--color-ink-dim)]">LSTM says (next window)</div>
              <div className="text-lg font-bold text-[var(--color-ink)]">{(data.lstm.infiltration_probability * 100).toFixed(1)}% infiltration</div>
              <div className="text-[10px] text-[var(--color-ink-faint)] mt-0.5">from attention-pooled history</div>
            </div>
            <div className="bg-[var(--color-accent-soft)] rounded-xl p-3">
              <div className="text-xs text-[var(--color-ink-dim)]">Do they blame the same features?</div>
              <div className="text-lg font-bold text-[var(--color-ink)]">{data.agreement.shared_features.length} shared · overlap {(data.agreement.jaccard * 100).toFixed(0)}%</div>
              <div className="text-[10px] text-[var(--color-ink-faint)] mt-0.5 truncate">
                {data.agreement.shared_features.length ? data.agreement.shared_features.join(', ') : 'no shared top features'}
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div>
              <div className="text-sm font-semibold text-[var(--color-ink)] mb-1">SHAP — baseline logistic regression</div>
              <p className="text-xs text-[var(--color-ink-dim)] mb-3">
                Each bar = how far that feature pushed the score away from normal traffic (log-odds).
                <span className="text-[var(--color-bad)]"> Red raises risk</span>,
                <span className="text-[var(--color-good)]"> green lowers it</span>.
              </p>
              <div className="flex flex-col gap-2.5">
                {data.shap.contributions.map((c) => (
                  <div key={c.feature}>
                    <div className="flex items-center justify-between text-xs mb-1">
                      <span className="font-medium text-[var(--color-ink)]">
                        {c.feature}{sharedSet.has(c.feature) && <span className="ml-1.5 text-[10px] text-[var(--color-accent)] bg-[var(--color-accent-soft)] px-1.5 py-0.5 rounded-full">also in LSTM</span>}
                      </span>
                      <span className="text-[var(--color-ink-dim)]">
                        value {fmtValue(c.raw_value)} ({c.scaled_value >= 0 ? '+' : ''}{c.scaled_value.toFixed(1)}σ) · {c.shap_value >= 0 ? '+' : ''}{c.shap_value.toFixed(2)}
                      </span>
                    </div>
                    <div className="h-2 rounded-full bg-[var(--color-accent-soft)] overflow-hidden">
                      <div
                        className="h-full rounded-full"
                        style={{
                          width: `${Math.max(3, (Math.abs(c.shap_value) / maxAbsShap) * 100)}%`,
                          background: c.direction === 'raises_risk' ? 'var(--color-bad)' : 'var(--color-good)',
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <div className="text-sm font-semibold text-[var(--color-ink)] mb-1">Attention × saliency — LSTM world model</div>
              <p className="text-xs text-[var(--color-ink-dim)] mb-3">
                Each bar = a (window, feature) pair the LSTM leaned on, weighted by how much attention that window received.
                "t0" is the most recent window.
              </p>
              <div className="flex flex-col gap-2.5">
                {data.lstm.top_contributors.map((c, i) => (
                  <div key={i}>
                    <div className="flex items-center justify-between text-xs mb-1">
                      <span className="font-medium text-[var(--color-ink)]">
                        {c.feature}<span className="text-[var(--color-ink-faint)] font-normal"> · t{c.timestep_offset} · attn {(c.attention_weight * 100).toFixed(0)}%</span>
                        {sharedSet.has(c.feature) && <span className="ml-1.5 text-[10px] text-[var(--color-accent)] bg-[var(--color-accent-soft)] px-1.5 py-0.5 rounded-full">also in SHAP</span>}
                      </span>
                      <span className="text-[var(--color-ink-dim)]">{c.contribution_score >= 0 ? '+' : ''}{c.contribution_score.toFixed(4)}</span>
                    </div>
                    <div className="h-2 rounded-full bg-[var(--color-accent-soft)] overflow-hidden">
                      <div
                        className="h-full rounded-full"
                        style={{
                          width: `${Math.max(3, (Math.abs(c.contribution_score) / maxAbsLstm) * 100)}%`,
                          background: c.contribution_score >= 0 ? 'var(--color-bad)' : 'var(--color-good)',
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <p className="text-[11px] text-[var(--color-ink-faint)] mt-5">
            The two methods look at different inputs (one window vs. eight), so partial disagreement is expected and is not an error.
            SHAP is applied to the baseline only — SHAP on a recurrent model is unreliable, which is why the LSTM uses its own attention and gradient saliency.
            {data.host_id.startsWith('live:') && ' Live host: refreshes every 10 s.'}
          </p>
        </>
      )}
    </div>
  );
}
