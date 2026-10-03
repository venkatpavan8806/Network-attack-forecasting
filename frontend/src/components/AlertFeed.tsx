import { useEffect, useRef, useState } from 'react';
import { api } from '../api';
import type { TripwireAlert } from '../types';
import { AlertIcon } from '../icons';

function msAgo(iso: string): string {
  const ms = Date.now() - new Date(iso).getTime();
  if (ms < 1000) return `${Math.max(0, Math.round(ms))}ms ago`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s ago`;
  return `${Math.round(ms / 60000)}m ago`;
}

export default function AlertFeed({ running }: { running: boolean }) {
  const [alerts, setAlerts] = useState<TripwireAlert[]>([]);
  const pollRef = useRef<number | null>(null);

  useEffect(() => {
    if (!running) {
      setAlerts([]);
      return;
    }
    const poll = () => api.liveAlerts(30).then(setAlerts).catch(() => {});
    poll();
    pollRef.current = window.setInterval(poll, 1000);
    return () => { if (pollRef.current) window.clearInterval(pollRef.current); };
  }, [running]);

  if (!running) return null;

  return (
    <div className="mt-4">
      <div className="flex items-center gap-2 mb-2">
        <AlertIcon size={14} className="text-[var(--color-bad)]" />
        <span className="text-xs font-semibold text-[var(--color-ink)] uppercase tracking-wide">Instant Alerts</span>
        <span className="text-[10px] text-[var(--color-ink-faint)]">— fast rule-based tripwire, millisecond latency, independent of the LSTM forecast below</span>
      </div>
      {alerts.length === 0 ? (
        <div className="text-xs text-[var(--color-ink-faint)] bg-[var(--color-accent-soft)]/40 rounded-xl px-4 py-3">
          no tripwire alerts yet — fires instantly on contact with a watched port (22/445/3389/443) or rapid multi-port probing
        </div>
      ) : (
        <div className="flex flex-col gap-1.5 max-h-56 overflow-y-auto pr-1">
          {alerts.map((a) => (
            <div
              key={a.id}
              className={`flex items-center justify-between gap-3 text-xs rounded-xl px-3 py-2 ${
                a.severity === 'critical' ? 'bg-[var(--color-bad)]/10 text-[var(--color-bad)]' : 'bg-[var(--color-warn)]/15 text-[#8a5a00]'
              }`}
            >
              <span className="flex-1 min-w-0 truncate font-medium">{a.message}</span>
              <span className="shrink-0 opacity-70 tabular-nums">{msAgo(a.timestamp)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
