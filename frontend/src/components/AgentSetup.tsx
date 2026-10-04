import { useEffect, useState } from 'react';
import { api, agentDownloadUrl } from '../api';
import type { CreatedSensor, Sensor } from '../types';

function ago(iso: string | null): string {
  if (!iso) return 'never';
  const s = Math.max(0, Math.round((Date.now() - new Date(iso).getTime()) / 1000));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  return `${Math.round(s / 3600)}h ago`;
}

/**
 * Shown in the Live Capture panel when no capture agent is running. A
 * browser cannot read packets and the hosted backend only sees its own, so
 * live capture runs through a small agent on the user's computer: add a
 * sensor (token shown once), download the agent, run the command. Once it is
 * running, the panel's interface list and Start/Stop button control it.
 */
export default function AgentSetup() {
  const [sensors, setSensors] = useState<Sensor[]>([]);
  const [name, setName] = useState('my-laptop');
  const [created, setCreated] = useState<CreatedSensor | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const load = () => api.sensors().then(setSensors).catch(() => {});
  useEffect(() => {
    load();
    const t = window.setInterval(load, 4000);
    return () => window.clearInterval(t);
  }, []);

  async function add() {
    setError(null);
    try {
      setCreated(await api.createSensor(name.trim() || 'my-laptop'));
      load();
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? String(e));
    }
  }

  async function remove(s: Sensor) {
    if (!window.confirm(`Remove sensor "${s.name}"? Its agent will stop being accepted.`)) return;
    await api.deleteSensor(s.id).catch(() => {});
    if (created?.id === s.id) setCreated(null);
    load();
  }

  const command = created ? `python nadf_agent.py --token ${created.token}` : '';

  return (
    <div className="mt-2 mb-4 rounded-xl border border-[var(--color-accent-soft)] bg-[var(--color-accent-soft)]/30 p-4 text-sm">
      <div className="font-semibold text-[var(--color-ink)]">Connect your computer to capture its real traffic</div>
      <p className="text-xs text-[var(--color-ink-dim)] mt-1">
        A website can't read your network packets, so capture runs in a small agent on the computer you want to monitor.
        It sends only the 30-second window features and packet headers to your account.
      </p>
      <ol className="text-xs text-[var(--color-ink-dim)] list-decimal pl-5 mt-3 flex flex-col gap-2">
        <li>
          <div className="flex flex-wrap items-center gap-2">
            <span>Add a sensor:</span>
            <input value={name} onChange={(e) => setName(e.target.value)} maxLength={100}
              className="px-2 py-1 rounded-lg border border-[var(--color-accent-soft)] text-xs bg-white text-[var(--color-ink)]" />
            <button onClick={add} className="px-3 py-1 rounded-full bg-[var(--color-accent)] text-white text-xs font-medium">Add sensor</button>
          </div>
          {error && <div className="text-[var(--color-bad)] mt-1">{error}</div>}
        </li>
        <li>
          <a href={agentDownloadUrl()} className="text-[var(--color-accent)] font-medium">Download the agent (.zip)</a>, unzip it and run{' '}
          <code>pip install -r requirements.txt</code> in that folder (Python 3.9+).
          Windows: install <a className="text-[var(--color-accent)]" href="https://npcap.com" target="_blank" rel="noreferrer">Npcap</a> and use an Administrator terminal; Linux/macOS: use <code>sudo</code>.
        </li>
        <li>
          Run the agent with your sensor's token{created ? ' (copy it now -- it is shown only once):' : ' (shown after you add a sensor).'}
          {created && (
            <div className="flex items-stretch gap-2 mt-1">
              <code className="flex-1 min-w-0 overflow-x-auto whitespace-nowrap bg-[#1f2233] text-[#e8e6ff] rounded-lg px-3 py-1.5">{command}</code>
              <button onClick={() => navigator.clipboard?.writeText(command).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500); })}
                className="px-3 rounded-lg bg-white text-[var(--color-accent)] text-xs font-medium">{copied ? 'Copied' : 'Copy'}</button>
            </div>
          )}
        </li>
        <li>Keep it running. This panel then lists that computer's interfaces and the Start button captures on it.</li>
      </ol>

      {sensors.length > 0 && (
        <div className="mt-3 pt-3 border-t border-[var(--color-accent-soft)]">
          <div className="text-xs font-semibold text-[var(--color-ink)] mb-1">Your sensors</div>
          {sensors.map((s) => (
            <div key={s.id} className="flex items-center gap-3 text-xs py-1">
              <span className={`w-2 h-2 rounded-full ${s.online ? 'bg-[var(--color-good)] animate-pulse' : 'bg-[var(--color-ink-faint)]'}`} />
              <span className="font-medium text-[var(--color-ink)]">{s.name}</span>
              <span className="text-[var(--color-ink-dim)]">
                {s.online ? `online${s.hostname ? ` on ${s.hostname}` : ''}` : s.last_seen_at ? `offline (last seen ${ago(s.last_seen_at)})` : 'waiting for the agent to start'}
              </span>
              {s.error && <span className="text-[var(--color-bad)]">{s.error}</span>}
              <button onClick={() => remove(s)} className="ml-auto text-[var(--color-bad)]">Remove</button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
