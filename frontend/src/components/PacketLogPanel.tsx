import { useEffect, useRef, useState } from 'react';
import { api } from '../api';
import type { LivePacket } from '../types';
import { ArrowDownIcon, ArrowUpIcon } from '../icons';

function flagBadgeColor(flags: string): string {
  if (flags.includes('R')) return 'var(--color-bad)';
  if (flags.includes('S') && !flags.includes('A')) return 'var(--color-accent)';
  if (flags.includes('S') && flags.includes('A')) return 'var(--color-good)';
  if (flags.includes('F')) return 'var(--color-ink-faint)';
  return 'var(--color-warn)';
}

export default function PacketLogPanel({ remoteIp }: { remoteIp: string }) {
  const [packets, setPackets] = useState<LivePacket[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);

  useEffect(() => {
    const poll = () => api.livePackets(remoteIp, 100).then(setPackets).catch((e) => setError(e?.response?.data?.detail ?? String(e)));
    poll();
    pollRef.current = window.setInterval(poll, 2000);
    return () => { if (pollRef.current) window.clearInterval(pollRef.current); };
  }, [remoteIp]);

  return (
    <div className="bg-[var(--color-accent-soft)]/40 rounded-xl p-4 mt-1">
      <div className="text-xs font-semibold text-[var(--color-ink)] mb-1">
        Packets to/from {remoteIp} <span className="font-normal text-[var(--color-ink-faint)]">(most recent first, live)</span>
      </div>
      {error && <div className="text-xs text-[var(--color-bad)]">{error}</div>}
      {!packets ? (
        <div className="text-xs text-[var(--color-ink-faint)] py-3">loading…</div>
      ) : packets.length === 0 ? (
        <div className="text-xs text-[var(--color-ink-faint)] py-3">no packets captured yet for this host</div>
      ) : (
        <div className="max-h-64 overflow-y-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-left text-[var(--color-ink-faint)] uppercase tracking-wide sticky top-0 bg-[var(--color-page)]">
                <th className="py-1.5 pr-3 font-medium">Dir</th>
                <th className="py-1.5 pr-3 font-medium">Local port</th>
                <th className="py-1.5 pr-3 font-medium">Remote port</th>
                <th className="py-1.5 pr-3 font-medium">Flags</th>
                <th className="py-1.5 pr-3 font-medium">TTL</th>
                <th className="py-1.5 pr-3 font-medium">Size</th>
                <th className="py-1.5 pr-3 font-medium">What this packet is</th>
              </tr>
            </thead>
            <tbody>
              {packets.map((p, i) => (
                <tr key={i} className="border-t border-white/60">
                  <td className="py-1.5 pr-3">
                    {p.direction === 'in' ? (
                      <ArrowDownIcon size={12} className="text-[var(--color-bad)]" />
                    ) : (
                      <ArrowUpIcon size={12} className="text-[var(--color-good)]" />
                    )}
                  </td>
                  <td className="py-1.5 pr-3 text-[var(--color-ink)]">{p.local_port}</td>
                  <td className="py-1.5 pr-3 text-[var(--color-ink-dim)]">{p.remote_port}</td>
                  <td className="py-1.5 pr-3">
                    <span className="px-1.5 py-0.5 rounded-full text-[10px] font-semibold text-white" style={{ background: flagBadgeColor(p.flags) }}>
                      {p.flags || '-'}
                    </span>
                  </td>
                  <td className="py-1.5 pr-3 text-[var(--color-ink-dim)]">{p.ttl}</td>
                  <td className="py-1.5 pr-3 text-[var(--color-ink-dim)]">{p.pkt_len}B</td>
                  <td className="py-1.5 pr-3 text-[var(--color-ink-dim)]">{p.description}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
