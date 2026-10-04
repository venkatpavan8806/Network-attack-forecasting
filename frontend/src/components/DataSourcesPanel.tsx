import { useEffect, useRef, useState } from 'react';
import { api } from '../api';
import type { HostDetail, PcapUploadResult, SampleDataResult, SandboxTestResult } from '../types';
import CardHeader from './CardHeader';
import { UploadIcon } from '../icons';

const SOURCE_LABELS: Record<string, string> = {
  agent: 'live sensor', pcap: 'pcap upload', csv: 'CSV upload', sample: 'sample data',
};

const errText = (e: any) => e?.response?.data?.detail ?? e?.message ?? String(e);

export default function DataSourcesPanel({ onSelectHost, onChanged }: { onSelectHost: (h: string) => void; onChanged?: () => void }) {
  const pcapRef = useRef<HTMLInputElement>(null);
  const csvRef = useRef<HTMLInputElement>(null);
  const [localIp, setLocalIp] = useState('');
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pcapResult, setPcapResult] = useState<PcapUploadResult | null>(null);
  const [csvResult, setCsvResult] = useState<SandboxTestResult | null>(null);
  const [sampleResult, setSampleResult] = useState<SampleDataResult | null>(null);
  const [hosts, setHosts] = useState<HostDetail[]>([]);

  const loadHosts = () => api.hostsDetails().then(setHosts).catch(() => {});
  useEffect(() => { loadHosts(); }, []);
  const changed = () => { loadHosts(); onChanged?.(); };

  async function run<T>(label: string, fn: () => Promise<T>): Promise<T | undefined> {
    setBusy(label);
    setError(null);
    try {
      return await fn();
    } catch (e) {
      setError(errText(e));
      return undefined;
    } finally {
      setBusy(null);
    }
  }

  async function uploadPcap() {
    const file = pcapRef.current?.files?.[0];
    if (!file) { setError('choose a .pcap or .pcapng file first'); return; }
    const res = await run('pcap', () => api.uploadPcap(file, localIp.trim() || undefined));
    if (res) { setPcapResult(res); changed(); if (res.hosts[0]) onSelectHost(res.hosts[0].host_id); }
  }

  async function csv(mode: 'test' | 'ingest') {
    const file = csvRef.current?.files?.[0];
    if (!file) { setError('choose a CSV file first'); return; }
    if (mode === 'test') {
      const res = await run('csv', () => api.sandboxTest(file));
      if (res) setCsvResult(res);
    } else {
      const res = await run('csv', () => api.ingest(file));
      if (res) {
        setCsvResult({ outcome: 'success', errors: [], rows: res.length, hosts: new Set(res.map((r) => r.host_id)).size });
        changed();
        if (res[0]) onSelectHost(res[0].host_id);
      }
    }
  }

  async function sample() {
    const res = await run('sample', () => api.generateSample(2, 2));
    if (res) { setSampleResult(res); changed(); const a = res.hosts.find((h) => h.host_id.includes('attack')); if (a) onSelectHost(a.host_id); }
  }

  async function removeHost(h: string) {
    await run('delete', () => api.deleteHost(h));
    changed();
  }

  async function reset() {
    if (!window.confirm('Delete ALL traffic, forecasts, alerts and packets in your workspace? Sensors are kept.')) return;
    await run('reset', () => api.resetWorkspace(false));
    setPcapResult(null); setCsvResult(null); setSampleResult(null);
    changed();
  }

  const btn = 'px-4 py-2 rounded-full text-sm font-medium flex items-center gap-2 disabled:opacity-60';

  return (
    <div className="card p-6">
      <CardHeader title="Other Data Sources" subtitle="Analyse traffic you already captured, or generate fresh sample traffic to try the dashboard." />
      {error && <div className="mb-4 text-sm rounded-xl p-3 bg-[var(--color-bad)]/10 text-[var(--color-bad)]">{error}</div>}

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <div className="flex flex-col gap-2">
          <div className="text-sm font-semibold text-[var(--color-ink)]">Upload a packet capture</div>
          <p className="text-xs text-[var(--color-ink-dim)]">
            A <code>.pcap</code>/<code>.pcapng</code> from Wireshark or tcpdump, replayed through the same feature extraction and tripwire as the live agent (one prediction per 30 s of capture).
          </p>
          <input ref={pcapRef} type="file" accept=".pcap,.pcapng,.cap" className="text-sm text-[var(--color-ink-dim)]" />
          <input value={localIp} onChange={(e) => setLocalIp(e.target.value)} placeholder="monitored host IP (optional — auto-detected)"
            className="px-3 py-2 rounded-xl border border-[var(--color-accent-soft)] text-sm bg-white text-[var(--color-ink)]" />
          <button onClick={uploadPcap} disabled={busy !== null} className={`${btn} self-start bg-[var(--color-accent)] text-white`}>
            <UploadIcon size={14} /> {busy === 'pcap' ? 'Analysing…' : 'Upload & analyse'}
          </button>
          {pcapResult && (
            <div className="text-xs rounded-xl p-3 bg-[var(--color-good)]/10 text-[var(--color-ink-dim)]">
              Monitored host <strong>{pcapResult.local_ip}</strong>: {pcapResult.stats.packets_used.toLocaleString()} TCP packets over {Math.round(pcapResult.stats.capture_seconds)} s →{' '}
              {pcapResult.stats.windows} windows from {pcapResult.stats.remote_hosts} remote host(s) that opened connections, {pcapResult.stats.alerts} tripwire alert(s).
              {pcapResult.hosts.length === 0 && ' No remote host initiated connections to the monitored host, so there is nothing to score.'}
            </div>
          )}
        </div>

        <div className="flex flex-col gap-2">
          <div className="text-sm font-semibold text-[var(--color-ink)]">Upload telemetry CSV</div>
          <p className="text-xs text-[var(--color-ink-dim)]">
            Pre-computed window features (one row per host per window, all feature columns). "Validate" checks it without storing anything.
          </p>
          <input ref={csvRef} type="file" accept=".csv" className="text-sm text-[var(--color-ink-dim)]" />
          <div className="flex gap-2">
            <button onClick={() => csv('test')} disabled={busy !== null} className={`${btn} bg-[var(--color-accent-soft)] text-[var(--color-accent)]`}>Validate</button>
            <button onClick={() => csv('ingest')} disabled={busy !== null} className={`${btn} bg-[var(--color-accent)] text-white`}>
              <UploadIcon size={14} /> {busy === 'csv' ? 'Working…' : 'Ingest'}
            </button>
          </div>
          {csvResult && (
            <div className={`text-xs rounded-xl p-3 ${csvResult.outcome === 'success' ? 'bg-[var(--color-good)]/10 text-[#1f8a5c]' : 'bg-[var(--color-bad)]/10 text-[var(--color-bad)]'}`}>
              {csvResult.outcome}: {csvResult.rows} rows, {csvResult.hosts} host(s)
              {csvResult.errors.length > 0 && <ul className="list-disc pl-5 mt-1">{csvResult.errors.map((e, i) => <li key={i}>{e}</li>)}</ul>}
            </div>
          )}
        </div>

        <div className="flex flex-col gap-2">
          <div className="text-sm font-semibold text-[var(--color-ink)]">Generate sample traffic</div>
          <p className="text-xs text-[var(--color-ink-dim)]">
            Fresh synthetic traffic (2 attacking + 2 normal hosts) from the simulator the model was trained on — a new random seed every time, labelled, so the step tracker can score itself.
          </p>
          <button onClick={sample} disabled={busy !== null} className={`${btn} self-start bg-[var(--color-accent)] text-white`}>
            {busy === 'sample' ? 'Generating…' : 'Generate sample data'}
          </button>
          {sampleResult && (
            <div className="text-xs rounded-xl p-3 bg-[var(--color-good)]/10 text-[var(--color-ink-dim)]">
              Added {sampleResult.hosts.length} hosts (seed {sampleResult.seed}).
            </div>
          )}
        </div>
      </div>

      <div className="mt-6 pt-5 border-t border-[var(--color-accent-soft)]">
        <div className="flex items-center justify-between mb-2 flex-wrap gap-2">
          <div className="text-sm font-semibold text-[var(--color-ink)]">Hosts in your workspace ({hosts.length})</div>
          <button onClick={reset} disabled={busy !== null || hosts.length === 0} className="text-xs text-[var(--color-bad)] disabled:opacity-40">Reset workspace</button>
        </div>
        {hosts.length === 0 ? (
          <div className="text-xs text-[var(--color-ink-faint)]">empty — start a sensor, upload a capture, or generate sample data</div>
        ) : (
          <div className="overflow-x-auto max-h-72 overflow-y-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-[var(--color-ink-faint)] text-xs uppercase tracking-wide">
                  <th className="py-2 pr-4 font-medium">Host</th>
                  <th className="py-2 pr-4 font-medium">Source</th>
                  <th className="py-2 pr-4 font-medium">Windows</th>
                  <th className="py-2 pr-4 font-medium">Last window</th>
                  <th className="py-2 pr-4 font-medium" />
                </tr>
              </thead>
              <tbody>
                {hosts.map((h) => (
                  <tr key={h.host_id} className="border-t border-[var(--color-accent-soft)]">
                    <td className="py-2 pr-4">
                      <button onClick={() => onSelectHost(h.host_id)} className="text-[var(--color-accent)] hover:underline font-medium">{h.host_id}</button>
                    </td>
                    <td className="py-2 pr-4 text-xs text-[var(--color-ink-dim)]">{SOURCE_LABELS[h.source] ?? h.source}</td>
                    <td className="py-2 pr-4 text-xs text-[var(--color-ink-dim)]">{h.n_windows}</td>
                    <td className="py-2 pr-4 text-xs text-[var(--color-ink-dim)]">{h.last_seen ? new Date(h.last_seen).toLocaleString() : '—'}</td>
                    <td className="py-2 pr-4 text-right"><button onClick={() => removeHost(h.host_id)} className="text-xs text-[var(--color-bad)]">Delete</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
