import { Fragment, useEffect, useMemo, useState } from 'react';
import { api, STAGE_COLORS, STAGE_LABELS } from '../api';
import type { LiveStatus, LiveWindowEntry } from '../types';
import CardHeader from '../components/CardHeader';
import AlertFeed from './AlertFeed';
import LiveTrajectoryChart, { type LiveTrajectoryPoint } from './LiveTrajectoryChart';
import PacketLogPanel from './PacketLogPanel';

const SEQ_LEN = 8; // must match backend app.config.SEQ_LEN

function formatTimestamp(iso: string): string {
  const d = new Date(iso);
  const clock = d.toLocaleTimeString(undefined, { hour12: false });
  const secondsAgo = Math.max(0, Math.round((Date.now() - d.getTime()) / 1000));
  const ago = secondsAgo < 60 ? `${secondsAgo}s ago` : secondsAgo < 3600 ? `${Math.round(secondsAgo / 60)}m ago` : `${Math.round(secondsAgo / 3600)}h ago`;
  return `${clock} (${ago})`;
}

/**
 * Live view of everything the user's capture agents have sent: one row per
 * remote host that opened connections to a monitored machine, with the
 * world model's prediction after every 30-second window (from the first
 * window -- warm-up windows are padded and flagged).
 */
export default function LiveCapturePanel({ onSelectHost }: { onSelectHost?: (h: string) => void }) {
  const [status, setStatus] = useState<LiveStatus | null>(null);
  const [recent, setRecent] = useState<LiveWindowEntry[]>([]);
  const [graphHost, setGraphHost] = useState<string | null>(null);
  const [showAll, setShowAll] = useState(false);
  const [expandedHost, setExpandedHost] = useState<string | null>(null);

  useEffect(() => {
    const poll = () => {
      api.liveStatus().then(setStatus).catch(() => {});
      api.liveRecent(200).then(setRecent).catch(() => {});
    };
    poll();
    const timer = window.setInterval(poll, 4000);
    return () => window.clearInterval(timer);
  }, []);

  // one row per host: its latest window's state, plus every port it has
  // EVER touched across the windows loaded (a scan often spreads ports
  // across windows)
  const hostRows = (() => {
    const byHost = new Map<string, { latest: LiveWindowEntry; windowCount: number; allPorts: Set<number> }>();
    for (const r of recent) {
      const existing = byHost.get(r.host_id);
      const ports = [22, 445, 3389, 443].filter((p) => (r.raw_features?.[`dst_port_is_${p}`] ?? 0) >= 0.5);
      if (!existing) {
        byHost.set(r.host_id, { latest: r, windowCount: 1, allPorts: new Set(ports) });
      } else {
        existing.windowCount += 1;
        ports.forEach((p) => existing.allPorts.add(p));
        if (r.window_idx > existing.latest.window_idx) existing.latest = r;
      }
    }
    const rows = Array.from(byHost.entries()).map(([hostId, v]) => ({ hostId, ...v }));
    rows.sort((a, b) => (b.latest.infiltration_probability_world_model ?? 0) - (a.latest.infiltration_probability_world_model ?? 0));
    return rows;
  })();

  const isFlagged = (row: (typeof hostRows)[number]) =>
    row.latest.predicted_stage != null && row.latest.predicted_stage !== 'benign';
  const visibleRows = showAll ? hostRows : hostRows.filter(isFlagged);
  const hiddenCount = hostRows.length - visibleRows.length;

  const hostsWithPredictions = useMemo(
    () => Array.from(new Set(recent.filter((r) => r.predicted_stage != null).map((r) => r.host_id))),
    [recent],
  );

  useEffect(() => {
    if (hostsWithPredictions.length === 0) setGraphHost(null);
    else if (!graphHost || !hostsWithPredictions.includes(graphHost)) setGraphHost(hostsWithPredictions[0]);
  }, [hostsWithPredictions]);

  const graphSeries: LiveTrajectoryPoint[] = useMemo(() => {
    if (!graphHost) return [];
    return recent
      .filter((r) => r.host_id === graphHost && r.infiltration_probability_world_model != null)
      .sort((a, b) => a.window_idx - b.window_idx)
      .map((r) => ({
        window_idx: r.window_idx,
        probability: r.infiltration_probability_world_model as number,
        action: r.predicted_stage,
        timestamp: r.timestamp,
      }));
  }, [recent, graphHost]);

  const running = !!status?.running;
  const hasData = recent.length > 0;

  return (
    <div className="card p-6">
      <CardHeader
        title="Live Capture"
        subtitle="Real packets captured by your agents, real feature extraction, real inference through the trained world model -- a prediction after every 30-second window, starting from the first one."
      />

      <div className="flex flex-wrap items-center gap-3 mb-2">
        {running ? (
          <span className="flex items-center gap-1.5 text-xs text-[#1f8a5c] font-medium">
            <span className="w-2 h-2 rounded-full bg-[var(--color-good)] animate-pulse" />
            {status!.sensors_online} sensor{status!.sensors_online === 1 ? '' : 's'} online
            {status!.local_ip && <> — watching <strong className="ml-1">{status!.local_ip}</strong></>}
          </span>
        ) : (
          <span className="text-xs text-[var(--color-ink-faint)]">
            {status && status.sensors_total > 0 ? 'no sensor online right now — start the agent on your machine' : 'no sensors yet — add one above to start capturing'}
          </span>
        )}
      </div>
      {status?.error && <div className="text-sm text-[var(--color-bad)] mb-2">{status.error}</div>}

      <AlertFeed running={running || hasData} />

      {(running || hasData) && (
        <div className="grid grid-cols-3 gap-4 mt-4 mb-2">
          <div className="bg-[var(--color-accent-soft)] rounded-xl p-3">
            <div className="text-xs text-[var(--color-ink-dim)]">Packets seen (all sensors)</div>
            <div className="text-lg font-bold text-[var(--color-ink)]">{(status?.packets_seen ?? 0).toLocaleString()}</div>
          </div>
          <div className="bg-[var(--color-accent-soft)] rounded-xl p-3">
            <div className="text-xs text-[var(--color-ink-dim)]">Remote hosts seen</div>
            <div className="text-lg font-bold text-[var(--color-ink)]">{status?.hosts_seen ?? 0}</div>
          </div>
          <div className="bg-[var(--color-accent-soft)] rounded-xl p-3">
            <div className="text-xs text-[var(--color-ink-dim)]">First prediction after</div>
            <div className="text-lg font-bold text-[var(--color-ink)]">1 window (30 s)</div>
          </div>
        </div>
      )}

      {hostRows.length > 0 && (
        <div className="mt-4">
          <div className="flex items-center justify-between flex-wrap gap-2 mb-2">
            <h4 className="text-sm font-semibold text-[var(--color-ink)]">{showAll ? 'All Remote Hosts' : 'Flagged Hosts'}</h4>
            <label className="flex items-center gap-2 text-xs text-[var(--color-ink-dim)] cursor-pointer select-none">
              <input type="checkbox" checked={showAll} onChange={(e) => setShowAll(e.target.checked)} className="accent-[var(--color-accent)]" />
              Show all traffic (including benign)
            </label>
          </div>
          {!showAll && hiddenCount > 0 && (
            <p className="text-xs text-[var(--color-ink-faint)] mb-2">
              {hiddenCount} benign host{hiddenCount === 1 ? '' : 's'} hidden — check "show all" to see them
            </p>
          )}
          {visibleRows.length === 0 ? (
            <div className="text-xs text-[var(--color-ink-faint)] bg-[var(--color-accent-soft)]/40 rounded-xl px-4 py-4 text-center">
              nothing flagged — {hostRows.length} host{hostRows.length === 1 ? '' : 's'} watched, all predicted benign
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-[var(--color-ink-faint)] text-xs uppercase tracking-wide">
                    <th className="py-2 pr-4 font-medium">Remote host</th>
                    <th className="py-2 pr-4 font-medium">History</th>
                    <th className="py-2 pr-4 font-medium">Predicted next action</th>
                    <th className="py-2 pr-4 font-medium">Infiltration prob.</th>
                    <th className="py-2 pr-4 font-medium">Ports touched</th>
                    <th className="py-2 pr-4 font-medium">Last updated</th>
                  </tr>
                </thead>
                <tbody>
                  {visibleRows.map((row) => {
                    const r = row.latest;
                    const ip = row.hostId.replace('live:', '');
                    const isExpanded = expandedHost === row.hostId;
                    const used = r.history_windows_used ?? Math.min(row.windowCount, SEQ_LEN);
                    return (
                      <Fragment key={row.hostId}>
                        <tr className="border-t border-[var(--color-accent-soft)]">
                          <td className="py-2.5 pr-4 font-medium">
                            <button
                              onClick={() => setExpandedHost(isExpanded ? null : row.hostId)}
                              className="text-[var(--color-accent)] hover:underline flex items-center gap-1"
                              title="Click to see individual packets from this host"
                            >
                              {ip}
                              <span className="text-[10px] text-[var(--color-ink-faint)]">{isExpanded ? '▲' : '▼'}</span>
                            </button>
                          </td>
                          <td className="py-2.5 pr-4 text-[var(--color-ink-dim)]">
                            window {r.window_idx}
                            {r.warmup && <span className="ml-1 text-[10px] text-[#b07400]">(warm-up {used}/{SEQ_LEN})</span>}
                          </td>
                          <td className="py-2.5 pr-4">
                            {r.predicted_stage ? (
                              <span className="flex items-center gap-1.5">
                                <span className="w-1.5 h-1.5 rounded-full" style={{ background: STAGE_COLORS[r.predicted_stage] ?? '#999' }} />
                                {STAGE_LABELS[r.predicted_stage] ?? r.predicted_stage}
                              </span>
                            ) : '—'}
                          </td>
                          <td className="py-2.5 pr-4 font-semibold text-[var(--color-ink)]">
                            {r.infiltration_probability_world_model != null ? `${(r.infiltration_probability_world_model * 100).toFixed(1)}%` : '—'}
                          </td>
                          <td className="py-2.5 pr-4 text-xs text-[var(--color-ink-dim)]">
                            {row.allPorts.size > 0 ? Array.from(row.allPorts).sort((a, b) => a - b).join(', ') : '—'}
                          </td>
                          <td className="py-2.5 pr-4 text-xs text-[var(--color-ink-dim)] whitespace-nowrap">
                            {formatTimestamp(r.timestamp)}
                            {onSelectHost && (
                              <button onClick={() => onSelectHost(row.hostId)} className="ml-2 text-[var(--color-accent)]">open →</button>
                            )}
                          </td>
                        </tr>
                        {isExpanded && (
                          <tr>
                            <td colSpan={6} className="pb-3">
                              <PacketLogPanel remoteIp={ip} />
                            </td>
                          </tr>
                        )}
                      </Fragment>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {hostsWithPredictions.length > 0 && (
        <div className="mt-6 pt-5 border-t border-[var(--color-accent-soft)]">
          <div className="flex items-center justify-between flex-wrap gap-3 mb-2">
            <div>
              <h4 className="text-sm font-semibold text-[var(--color-ink)]">Infiltration Probability Over Time (this host)</h4>
              <p className="text-xs text-[var(--color-ink-dim)] mt-0.5">Each real prediction as it happened -- dot color is the predicted action at that window</p>
            </div>
            <select
              value={graphHost ?? ''}
              onChange={(e) => setGraphHost(e.target.value)}
              className="px-3 py-1.5 rounded-xl border border-[var(--color-accent-soft)] text-xs bg-white text-[var(--color-ink)]"
            >
              {hostsWithPredictions.map((h) => <option key={h} value={h}>{h.replace('live:', '')}</option>)}
            </select>
          </div>
          <LiveTrajectoryChart data={graphSeries} />
        </div>
      )}
    </div>
  );
}
