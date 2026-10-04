import { useEffect, useState } from 'react';
import { api, agentDownloadUrl, agentServerUrl } from '../api';
import type { CreatedSensor, Sensor } from '../types';
import CardHeader from './CardHeader';

function ago(iso: string | null): string {
  if (!iso) return 'never';
  const s = Math.max(0, Math.round((Date.now() - new Date(iso).getTime()) / 1000));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}

function CopyBlock({ label, text }: { label: string; text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div>
      <div className="text-[11px] text-[var(--color-ink-faint)] mb-1">{label}</div>
      <div className="flex items-stretch gap-2">
        <code className="flex-1 min-w-0 overflow-x-auto whitespace-nowrap text-xs bg-[#1f2233] text-[#e8e6ff] rounded-lg px-3 py-2">{text}</code>
        <button
          onClick={() => { navigator.clipboard?.writeText(text).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500); }); }}
          className="px-3 rounded-lg bg-[var(--color-accent-soft)] text-[var(--color-accent)] text-xs font-medium shrink-0"
        >
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
    </div>
  );
}

export default function SensorsPanel() {
  const [sensors, setSensors] = useState<Sensor[]>([]);
  const [name, setName] = useState('my-laptop');
  const [created, setCreated] = useState<CreatedSensor | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const load = () => api.sensors().then(setSensors).catch(() => {});
  useEffect(() => {
    load();
    const t = window.setInterval(load, 5000);
    return () => window.clearInterval(t);
  }, []);

  async function add() {
    setBusy(true);
    setError(null);
    try {
      const s = await api.createSensor(name.trim() || 'sensor');
      setCreated(s);
      load();
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? String(e));
    } finally {
      setBusy(false);
    }
  }

  async function revoke(s: Sensor) {
    if (!window.confirm(`Revoke sensor "${s.name}"? Its agent will stop being accepted. Data it already sent is kept.`)) return;
    await api.deleteSensor(s.id).catch(() => {});
    if (created?.id === s.id) setCreated(null);
    load();
  }

  const server = agentServerUrl();
  const winCmd = created ? `python nadf_agent.py --token ${created.token}` : '';
  const nixCmd = created ? `sudo python3 nadf_agent.py --token ${created.token}` : '';

  return (
    <div className="card p-6">
      <CardHeader
        title="Capture Sensors"
        subtitle="A browser cannot read network packets, so capture runs in a small agent on the machine you want to protect. It computes the model's 30-second window features locally and sends only those (plus packet headers for the drill-down) to your private workspace."
      />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="flex flex-col gap-3">
          <div className="text-sm font-semibold text-[var(--color-ink)]">1. Add a sensor</div>
          <div className="flex gap-2">
            <input value={name} onChange={(e) => setName(e.target.value)} maxLength={100} placeholder="sensor name"
              className="flex-1 px-3 py-2 rounded-xl border border-[var(--color-accent-soft)] text-sm bg-white text-[var(--color-ink)]" />
            <button onClick={add} disabled={busy}
              className="px-4 py-2 rounded-full bg-[var(--color-accent)] text-white text-sm font-medium disabled:opacity-60">
              Add sensor
            </button>
          </div>
          {error && <div className="text-sm text-[var(--color-bad)]">{error}</div>}

          <div className="text-sm font-semibold text-[var(--color-ink)] mt-2">2. Download and run the agent</div>
          <a href={agentDownloadUrl()}
            className="self-start px-4 py-2 rounded-full bg-[var(--color-accent-soft)] text-[var(--color-accent)] text-sm font-medium">
            Download agent (.zip)
          </a>
          <ol className="text-xs text-[var(--color-ink-dim)] list-decimal pl-5 flex flex-col gap-1">
            <li>Unzip, open a terminal in the folder, run <code>pip install -r requirements.txt</code> (Python 3.9+).</li>
            <li><strong>Windows:</strong> install <a className="text-[var(--color-accent)]" href="https://npcap.com" target="_blank" rel="noreferrer">Npcap</a> and open the terminal as Administrator. <strong>Linux/macOS:</strong> use <code>sudo</code>.</li>
            <li>Run the command shown after you add a sensor. It reports to <code>{server}</code>.</li>
          </ol>
        </div>

        <div>
          {created ? (
            <div className="rounded-xl border border-[var(--color-warn)] bg-[var(--color-warn)]/10 p-4 flex flex-col gap-3">
              <div className="text-sm font-semibold text-[var(--color-ink)]">Token for “{created.name}” — copy it now, it is shown only once</div>
              <CopyBlock label="Windows (Administrator terminal)" text={winCmd} />
              <CopyBlock label="Linux / macOS" text={nixCmd} />
              <p className="text-[11px] text-[var(--color-ink-dim)]">
                Lost it? Revoke the sensor and add a new one. Test it by port-scanning this machine from another device, e.g. <code>nmap -sS &lt;its IP&gt;</code>.
              </p>
            </div>
          ) : (
            <div className="h-full rounded-xl bg-[var(--color-accent-soft)]/40 p-4 text-xs text-[var(--color-ink-dim)] flex items-center">
              Add a sensor to get its private token and the exact command to start capturing.
            </div>
          )}
        </div>
      </div>

      <div className="mt-6">
        <div className="text-sm font-semibold text-[var(--color-ink)] mb-2">Your sensors</div>
        {sensors.length === 0 ? (
          <div className="text-xs text-[var(--color-ink-faint)]">none yet</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-[var(--color-ink-faint)] text-xs uppercase tracking-wide">
                  <th className="py-2 pr-4 font-medium">Sensor</th>
                  <th className="py-2 pr-4 font-medium">Status</th>
                  <th className="py-2 pr-4 font-medium">Machine</th>
                  <th className="py-2 pr-4 font-medium">Packets seen</th>
                  <th className="py-2 pr-4 font-medium">Last report</th>
                  <th className="py-2 pr-4 font-medium" />
                </tr>
              </thead>
              <tbody>
                {sensors.map((s) => (
                  <tr key={s.id} className="border-t border-[var(--color-accent-soft)]">
                    <td className="py-2.5 pr-4 font-medium text-[var(--color-ink)]">
                      {s.name} <span className="text-[10px] text-[var(--color-ink-faint)]">…{s.token_hint}</span>
                    </td>
                    <td className="py-2.5 pr-4">
                      {s.online ? (
                        <span className="flex items-center gap-1.5 text-xs text-[#1f8a5c] font-medium">
                          <span className="w-2 h-2 rounded-full bg-[var(--color-good)] animate-pulse" /> online
                        </span>
                      ) : (
                        <span className="text-xs text-[var(--color-ink-faint)]">{s.last_seen_at ? 'offline' : 'waiting for agent'}</span>
                      )}
                      {s.error && <div className="text-[11px] text-[var(--color-bad)] max-w-xs">{s.error}</div>}
                    </td>
                    <td className="py-2.5 pr-4 text-xs text-[var(--color-ink-dim)]">
                      {s.hostname ? `${s.hostname} · ${s.local_ip ?? '?'}${s.iface ? ` · ${s.iface}` : ''}` : '—'}
                    </td>
                    <td className="py-2.5 pr-4 text-xs text-[var(--color-ink-dim)]">{s.packets_seen.toLocaleString()}</td>
                    <td className="py-2.5 pr-4 text-xs text-[var(--color-ink-dim)]">{ago(s.last_seen_at)}</td>
                    <td className="py-2.5 pr-4 text-right">
                      <button onClick={() => revoke(s)} className="text-xs text-[var(--color-bad)]">Revoke</button>
                    </td>
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
