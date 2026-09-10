import { useEffect, useRef, useState } from 'react';
import { api, STAGE_COLORS, STAGE_LABELS } from '../api';
import type { ForecastResponse, SandboxTestResult } from '../types';
import CardHeader from '../components/CardHeader';
import TrajectoryChart, { type TrajectoryPoint } from '../components/TrajectoryChart';
import LiveCapturePanel from '../components/LiveCapturePanel';
import { UploadIcon } from '../icons';

export default function Forecasts({ selectedHost, onSelectHost }: { selectedHost: string | null; onSelectHost: (h: string) => void }) {
  const [hosts, setHosts] = useState<string[]>([]);
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [uploadResult, setUploadResult] = useState<SandboxTestResult | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api.hosts().then((h) => {
      setHosts(h);
      if (!selectedHost && h.length) onSelectHost(h[0]);
    }).catch(() => {});
  }, []);

  useEffect(() => {
    if (!selectedHost) return;
    setLoading(true);
    api.forecast(selectedHost).then(setForecast).finally(() => setLoading(false)).catch(() => setLoading(false));
  }, [selectedHost]);

  const trajectory: TrajectoryPoint[] = forecast
    ? [
        { step: 'now', worldModel: forecast.infiltration_probability_world_model, baseline: forecast.infiltration_probability_baseline },
        ...forecast.rollout.infiltration_probs_world_model.map((p, i) => ({ step: `t+${i + 1}`, worldModel: p, baseline: null })),
      ]
    : [];

  async function handleFile(mode: 'test' | 'ingest') {
    const file = fileRef.current?.files?.[0];
    if (!file) return;
    setUploadError(null);
    setUploadResult(null);
    try {
      if (mode === 'test') {
        const res = await api.sandboxTest(file);
        setUploadResult(res);
      } else {
        const res = await api.ingest(file);
        setUploadResult({ outcome: 'success', errors: [], rows: res.length, hosts: new Set(res.map((r) => r.host_id)).size });
        if (res.length) onSelectHost(res[0].host_id);
      }
    } catch (e: any) {
      setUploadError(e?.response?.data?.detail ?? String(e));
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <LiveCapturePanel />

      <div className="card p-6">
        <CardHeader title="CSV Ingestion" subtitle="Upload a synthetic-telemetry-shaped CSV to run real inference offline. /sandbox/test validates structure only; /ingest also runs the models." />
        <div className="flex flex-wrap items-center gap-3">
          <input ref={fileRef} type="file" accept=".csv" className="text-sm text-[var(--color-ink-dim)]" />
          <button onClick={() => handleFile('test')} className="px-4 py-2 rounded-full bg-[var(--color-accent-soft)] text-[var(--color-accent)] text-sm font-medium flex items-center gap-2">
            <UploadIcon size={14} /> Validate only
          </button>
          <button onClick={() => handleFile('ingest')} className="px-4 py-2 rounded-full bg-[var(--color-accent)] text-white text-sm font-medium flex items-center gap-2">
            <UploadIcon size={14} /> Ingest &amp; forecast
          </button>
        </div>
        {uploadResult && (
          <div className={`mt-4 text-sm rounded-xl p-3 ${uploadResult.outcome === 'success' ? 'bg-[var(--color-good)]/10 text-[var(--color-good)]' : 'bg-[var(--color-bad)]/10 text-[var(--color-bad)]'}`}>
            outcome: <strong>{uploadResult.outcome}</strong> — {uploadResult.rows} rows, {uploadResult.hosts} host(s)
            {uploadResult.errors.length > 0 && (
              <ul className="list-disc pl-5 mt-1">
                {uploadResult.errors.map((e, i) => <li key={i}>{e}</li>)}
              </ul>
            )}
          </div>
        )}
        {uploadError && <div className="mt-4 text-sm rounded-xl p-3 bg-[var(--color-bad)]/10 text-[var(--color-bad)]">{uploadError}</div>}
      </div>

      <div className="card p-6">
        <div className="flex items-center justify-between mb-5 flex-wrap gap-3">
          <CardHeader title="Host Forecast Explorer" subtitle="Select a host to run a real one-step forecast + K-step rollout" />
          <select
            value={selectedHost ?? ''}
            onChange={(e) => onSelectHost(e.target.value)}
            className="px-3 py-2 rounded-xl border border-[var(--color-accent-soft)] text-sm bg-white text-[var(--color-ink)]"
          >
            {hosts.map((h) => <option key={h} value={h}>{h}</option>)}
          </select>
        </div>

        {loading && <div className="text-sm text-[var(--color-ink-faint)] py-8 text-center">running inference…</div>}

        {!loading && forecast && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <TrajectoryChart data={trajectory} />
            </div>
            <div className="flex flex-col gap-4">
              <div>
                <div className="text-xs text-[var(--color-ink-faint)] mb-2">Stage probabilities (window {forecast.window_idx})</div>
                <div className="flex flex-col gap-2">
                  {Object.entries(forecast.stage_probabilities).sort((a, b) => b[1] - a[1]).map(([stage, p]) => (
                    <div key={stage} className="flex items-center gap-2 text-xs">
                      <span className="w-2 h-2 rounded-full shrink-0" style={{ background: STAGE_COLORS[stage] }} />
                      <span className="flex-1 truncate text-[var(--color-ink-dim)]">{STAGE_LABELS[stage]}</span>
                      <span className="font-semibold text-[var(--color-ink)]">{(p * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </div>
              {forecast.attack_mapping.technique_id && (
                <div className="text-xs bg-[var(--color-accent-soft)] rounded-xl p-3 text-[var(--color-ink)]">
                  <div className="font-semibold">{forecast.attack_mapping.technique_id} — {forecast.attack_mapping.technique_name}</div>
                  <div className="text-[var(--color-ink-dim)] mt-0.5">{forecast.attack_mapping.tactic}</div>
                </div>
              )}
              {forecast.true_stage && (
                <div className="text-xs text-[var(--color-ink-faint)]">
                  ground truth: <span className="font-medium text-[var(--color-ink-dim)]">{forecast.true_stage}</span>
                  {forecast.state_label && forecast.state_label !== forecast.true_stage && (
                    <> (state-labeling engine: <span className="font-medium text-[var(--color-ink-dim)]">{forecast.state_label}</span>)</>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
