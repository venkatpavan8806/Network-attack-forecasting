import { useState } from 'react';
import { api } from '../api';

/** Shown instead of a page's content while the user's workspace has no hosts. */
export default function EmptyWorkspace({ onGoToCapture, onSampleLoaded }: {
  onGoToCapture: () => void;
  onSampleLoaded?: (firstHost: string) => void;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function sample() {
    setBusy(true);
    setError(null);
    try {
      const res = await api.generateSample(2, 2);
      const first = res.hosts.find((h) => h.host_id.includes('attack')) ?? res.hosts[0];
      if (first) onSampleLoaded?.(first.host_id);
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card p-8 text-center">
      <h3 className="text-lg font-semibold text-[var(--color-ink)]">Your workspace is empty</h3>
      <p className="text-sm text-[var(--color-ink-dim)] mt-1 max-w-2xl mx-auto">
        Nothing here is pre-loaded: every forecast on this site comes from traffic you provide. Connect a capture sensor
        to watch a machine live, upload a Wireshark capture, or generate fresh sample traffic to explore the dashboard.
      </p>
      <div className="flex flex-wrap justify-center gap-3 mt-5">
        <button onClick={onGoToCapture} className="px-5 py-2.5 rounded-full bg-[var(--color-accent)] text-white text-sm font-medium">
          Connect a sensor / upload a capture
        </button>
        <button onClick={sample} disabled={busy} className="px-5 py-2.5 rounded-full bg-[var(--color-accent-soft)] text-[var(--color-accent)] text-sm font-medium disabled:opacity-60">
          {busy ? 'Generating…' : 'Generate sample traffic'}
        </button>
      </div>
      {error && <div className="text-sm text-[var(--color-bad)] mt-3">{error}</div>}
    </div>
  );
}
