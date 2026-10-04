import { useEffect, useRef, useState } from 'react';
import { api, STAGE_COLORS, STAGE_LABELS } from '../api';
import type { AttackMapping, ForecastResponse, BranchingForecastResponse, SandboxTestResult } from '../types';
import { pickInterestingWindowIdx } from '../hostWindow';
import CardHeader from '../components/CardHeader';
import TrajectoryChart, { type TrajectoryPoint } from '../components/TrajectoryChart';
import BranchingForecastTree, { PathSummaryList } from '../components/BranchingForecastTree';
import LiveCapturePanel from '../components/LiveCapturePanel';
import AttackerStepTracker from '../components/AttackerStepTracker';
import MitreForecastGraph from '../components/MitreForecastGraph';
import AttackForecastDetails from '../components/AttackForecastDetails';
import { downloadForecastPdf } from '../pdfReport';
import { UploadIcon } from '../icons';

export default function Forecasts({ selectedHost, onSelectHost }: { selectedHost: string | null; onSelectHost: (h: string) => void }) {
  const [hosts, setHosts] = useState<string[]>([]);
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [branching, setBranching] = useState<BranchingForecastResponse | null>(null);
  const [branchingLoading, setBranchingLoading] = useState(false);
  const [loading, setLoading] = useState(false);
  const [uploadResult, setUploadResult] = useState<SandboxTestResult | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [attackMapping, setAttackMapping] = useState<AttackMapping[]>([]);
  const [forecastError, setForecastError] = useState<string | null>(null);
  const [liveUpdatedAt, setLiveUpdatedAt] = useState<number | null>(null);
  const [, setClock] = useState(0);
  const fileRef = useRef<HTMLInputElement>(null);

  // the host list includes live-captured hosts (live:<ip>), which appear as
  // the capture runs -- so keep it fresh
  useEffect(() => {
    const loadHosts = () => api.hosts().then((h) => {
      setHosts(h);
      if (!selectedHost && h.length) onSelectHost(h[0]);
    }).catch(() => {});
    loadHosts();
    api.attackMapping().then(setAttackMapping).catch(() => {});
    const t = window.setInterval(loadHosts, 10000);
    return () => window.clearInterval(t);
  }, [selectedHost]);

  useEffect(() => {
    if (!selectedHost) return;
    const isLive = selectedHost.startsWith('live:');
    setLoading(true);
    setBranchingLoading(true);

    setForecastError(null);
    setLiveUpdatedAt(null);

    const run = (atWindowIdx: number | undefined) => {
      api.forecast(selectedHost, atWindowIdx)
        .then((f) => { setForecast(f); setForecastError(null); if (isLive) setLiveUpdatedAt(Date.now()); })
        .catch((e) => { setForecast(null); setForecastError(e?.response?.data?.detail ?? String(e)); })
        .finally(() => setLoading(false));
      api.branchingForecast(selectedHost, atWindowIdx).then(setBranching).catch(() => setBranching(null))
        .finally(() => setBranchingLoading(false));
    };

    if (isLive) {
      // a live host gains a new window every 30 s while its capture runs:
      // re-run the forecast every 10 s so the page follows it as it happens
      run(undefined);
      const t = window.setInterval(() => run(undefined), 10000);
      const tick = window.setInterval(() => setClock((c) => c + 1), 1000);
      return () => { window.clearInterval(t); window.clearInterval(tick); };
    }
    // Default to the end of the host's first non-benign segment instead of
    // its last window -- a demo host's timeline always ends in a benign
    // tail, so "last window" would show nothing interesting.
    api.hostTimeline(selectedHost).then((t) => run(pickInterestingWindowIdx(t))).catch(() => run(undefined));
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
        setHosts(await api.hosts()); // uploaded hosts are saved to this user's account
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
          <div className="flex items-center gap-2">
            <select
              value={selectedHost ?? ''}
              onChange={(e) => onSelectHost(e.target.value)}
              className="px-3 py-2 rounded-xl border border-[var(--color-accent-soft)] text-sm bg-white text-[var(--color-ink)]"
            >
              {hosts.map((h) => <option key={h} value={h}>{h}</option>)}
            </select>
            {forecast && (
              <button
                onClick={() => downloadForecastPdf(forecast, attackMapping)}
                className="px-4 py-2 rounded-full bg-[var(--color-accent)] text-white text-sm font-medium flex items-center gap-2"
              >
                <UploadIcon size={14} className="rotate-180" /> Download PDF
              </button>
            )}
          </div>
        </div>

        {selectedHost?.startsWith('live:') && (
          <div className="flex items-center gap-2 text-xs mb-3">
            <span className="flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-[var(--color-good)]/15 text-[#1f8a5c] font-semibold">
              <span className="w-2 h-2 rounded-full bg-[var(--color-good)] animate-pulse" /> LIVE
            </span>
            <span className="text-[var(--color-ink-dim)]">
              {liveUpdatedAt ? `updated ${Math.max(0, Math.round((Date.now() - liveUpdatedAt) / 1000))}s ago` : 'waiting for data'}
              {forecast ? ` · window ${forecast.window_idx}` : ''} · refreshes as new 30-second windows are captured
            </span>
          </div>
        )}
        {!loading && forecastError && (
          <div className="text-sm text-[var(--color-ink-dim)] py-6 text-center">{forecastError}</div>
        )}
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

        {!loading && forecast && (
          <div className="mt-6 pt-5 border-t border-[var(--color-accent-soft)]">
            <div className="mb-3">
              <h4 className="text-sm font-semibold text-[var(--color-ink)]">Attack Type, Tools &amp; Likely System State</h4>
              <p className="text-xs text-[var(--color-ink-dim)] mt-0.5">
                What the model predicts, what tools are typically used for it, and what happens to the system if it succeeds.
              </p>
            </div>
            <AttackForecastDetails action={forecast.predicted_stage} mapping={forecast.attack_mapping} />
          </div>
        )}

        {!loading && forecast && attackMapping.length > 0 && (
          <div className="mt-6 pt-5 border-t border-[var(--color-accent-soft)]">
            <div className="mb-3">
              <h4 className="text-sm font-semibold text-[var(--color-ink)]">MITRE ATT&CK Forecast — Branching Next Actions</h4>
              <p className="text-xs text-[var(--color-ink-dim)] mt-0.5">
                Positioned by real MITRE tactic (Reconnaissance → Credential Access → Lateral Movement → C2 → Exfiltration). Hover a node for its technique ID.
              </p>
            </div>
            <MitreForecastGraph
              stageProbabilitiesNow={forecast.stage_probabilities}
              branchingForecast={forecast.rollout.branching_forecast}
              attackMapping={attackMapping}
            />
          </div>
        )}
      </div>

      <AttackerStepTracker hostId={selectedHost} />

      <div className="card p-6">
        <CardHeader
          title="Branching Attack-Path Forecast"
          subtitle={branching ? `K-step forecast forked into the ${branching.branch_factor} most probable next actions at each of ${branching.depth} steps, each node MITRE ATT&CK-mapped` : 'K-step + branching + MITRE ATT&CK mapping'}
        />
        {branchingLoading && <div className="text-sm text-[var(--color-ink-faint)] py-8 text-center">building attack-path tree…</div>}
        {!branchingLoading && branching && (
          <div className="flex flex-col gap-6">
            <BranchingForecastTree tree={branching.tree} />
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2 border-t border-[var(--color-accent-soft)]">
              {branching.most_likely_path && <PathSummaryList title="Most likely continuation" paths={[branching.most_likely_path]} />}
              {branching.highest_risk_path && <PathSummaryList title="Highest-risk continuation" paths={[branching.highest_risk_path]} />}
            </div>
          </div>
        )}
        {!branchingLoading && !branching && (
          <div className="text-sm text-[var(--color-ink-faint)] py-8 text-center">no branching forecast available for this host</div>
        )}
      </div>
    </div>
  );
}
