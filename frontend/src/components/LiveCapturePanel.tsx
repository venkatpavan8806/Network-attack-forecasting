import { Fragment, useEffect, useMemo, useRef, useState } from 'react';
import { api, STAGE_COLORS, STAGE_LABELS } from '../api';
import type { LiveInterface, LiveStatus, LiveWindowEntry } from '../types';
import CardHeader from '../components/CardHeader';
import AlertFeed from './AlertFeed';
import LiveTrajectoryChart, { type LiveTrajectoryPoint } from './LiveTrajectoryChart';
import PacketLogPanel from './PacketLogPanel';
import AgentSetup from './AgentSetup';

const SEQ_LEN = 8; // must match backend app.config.SEQ_LEN

function formatTimestamp(iso: string): string {
  const d = new Date(iso);
  const clock = d.toLocaleTimeString(undefined, { hour12: false });
  const secondsAgo = Math.max(0, Math.round((Date.now() - d.getTime()) / 1000));
  const ago = secondsAgo < 60 ? `${secondsAgo}s ago` : `${Math.round(secondsAgo / 60)}m ago`;
  return `${clock} (${ago})`;
}

export default function LiveCapturePanel() {
  const [status, setStatus] = useState<LiveStatus | null>(null);
  const [recent, setRecent] = useState<LiveWindowEntry[]>([]);
  const [interfaces, setInterfaces] = useState<LiveInterface[]>([]);
  const [selected, setSelected] = useState<LiveInterface | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [graphHost, setGraphHost] = useState<string | null>(null);
  const [showAll, setShowAll] = useState(false);
  const [expandedHost, setExpandedHost] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);
  const autoStartAttempted = useRef(false);

  // Interfaces come from the user's capture agent (or, for a local demo, the
  // backend's own machine). While not capturing, keep checking: the agent may
  // be started -- or stopped -- at any time.
  useEffect(() => {
    if (status?.running) return;
    const refresh = () => {
      api.liveStatus().then(setStatus).catch(() => {});
      api.liveInterfaces().then((list) => {
        setInterfaces(list);
        setSelected((cur) => (cur && list.some((i) => i.name === cur.name) ? cur
          : list.find((i) => i.name.toLowerCase() === 'wi-fi') ?? list[0] ?? null));
      }).catch(() => setInterfaces([]));
    };
    refresh();
    const t = window.setInterval(refresh, 4000);
    return () => window.clearInterval(t);
  }, [status?.running]);

  // opening this panel should start watching automatically -- no click required
  useEffect(() => {
    if (autoStartAttempted.current) return;
    if (status && !status.running && selected) {
      autoStartAttempted.current = true;
      start();
    }
  }, [status, selected]);

  useEffect(() => {
    if (status?.running) {
      const poll = () => {
        api.liveStatus().then(setStatus).catch(() => {});
        // pull the full recent log (not just the last few rows) so per-host
        // dedup below sees every window a host has accumulated, not a
        // truncated slice that might miss its most recent one
        api.liveRecent(200).then(setRecent).catch(() => {});
      };
      poll();
      pollRef.current = window.setInterval(poll, 4000);
      return () => { if (pollRef.current) window.clearInterval(pollRef.current); };
    }
  }, [status?.running]);

  // one row per host: its latest window's state, plus every port it has
  // EVER touched across all windows seen so far (a scan often spreads
  // different ports across different windows)
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
    rows.sort((a, b) => {
      const aHas = a.latest.infiltration_probability_world_model != null;
      const bHas = b.latest.infiltration_probability_world_model != null;
      if (aHas && bHas) return (b.latest.infiltration_probability_world_model ?? 0) - (a.latest.infiltration_probability_world_model ?? 0);
      if (aHas !== bHas) return aHas ? -1 : 1;
      return b.latest.window_idx - a.latest.window_idx; // closest to first prediction first
    });
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
    if (hostsWithPredictions.length === 0) {
      setGraphHost(null);
    } else if (!graphHost || !hostsWithPredictions.includes(graphHost)) {
      setGraphHost(hostsWithPredictions[0]);
    }
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

  async function start() {
    if (!selected) return;
    setError(null);
    try {
      const s = await api.liveStart(selected.name, selected.ip);
      setStatus(s);
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? String(e));
    }
  }

  async function stop() {
    const s = await api.liveStop();
    setStatus(s);
  }

  return (
    <div className="card p-6">
      <CardHeader
        title="Live Capture"
        subtitle="Real packets off this machine's own NIC, real feature extraction, real inference through the trained world model. Not synthetic. No promiscuous mode -- only traffic to/from this host is seen."
      />

      {!status?.running && interfaces.length === 0 && <AgentSetup />}
      {status?.agent_online && (
        <p className="text-xs text-[var(--color-ink-dim)] mb-2">
          Capture agent <strong>{status.agent_name}</strong> is online{status.agent_hostname ? <> on <strong>{status.agent_hostname}</strong></> : null} — the interfaces below are that computer's.
        </p>
      )}

      <div className="flex flex-wrap items-end gap-3 mb-2">
        <div>
          <label className="text-xs text-[var(--color-ink-faint)] mb-1.5 block">Interface</label>
          <select
            value={selected?.name ?? ''}
            onChange={(e) => setSelected(interfaces.find((i) => i.name === e.target.value) ?? null)}
            disabled={!!status?.running}
            className="px-3 py-2 rounded-xl border border-[var(--color-accent-soft)] text-sm bg-white text-[var(--color-ink)] min-w-[220px] disabled:opacity-50"
          >
            {interfaces.map((i) => (
              <option key={i.name} value={i.name}>{i.name} ({i.ip})</option>
            ))}
          </select>
        </div>
        {status?.running ? (
          <button onClick={stop} className="px-4 py-2 rounded-full bg-[var(--color-bad)] text-white text-sm font-medium">
            Stop capture
          </button>
        ) : (
          <button onClick={start} disabled={!selected} className="px-4 py-2 rounded-full bg-[var(--color-accent)] text-white text-sm font-medium disabled:opacity-50">
            Start live capture
          </button>
        )}
        {status?.running && (
          <span className="flex items-center gap-1.5 text-xs text-[var(--color-good)] font-medium">
            <span className="w-2 h-2 rounded-full bg-[var(--color-good)] animate-pulse" /> capturing
          </span>
        )}
      </div>

      <p className="text-xs text-[var(--color-ink-faint)] mb-2">
        {status?.running
          ? <>Watching traffic to/from <strong>{status.local_ip}</strong>.</>
          : selected
            ? <>Starts automatically on this interface — no click needed. Point the other laptop's attack tools at <strong>{selected.ip}</strong>.</>
            : <>No capture interface available yet — set up the capture agent above.</>}
      </p>

      {error && <div className="text-sm text-[var(--color-bad)] mb-2">{error}</div>}
      {status?.error && <div className="text-sm text-[var(--color-bad)] mb-2">{status.error}</div>}

      <AlertFeed running={!!status?.running} />

      {status?.running && (
        <div className="grid grid-cols-3 gap-4 mt-4 mb-2">
          <div className="bg-[var(--color-accent-soft)] rounded-xl p-3">
            <div className="text-xs text-[var(--color-ink-dim)]">Packets seen</div>
            <div className="text-lg font-bold text-[var(--color-ink)]">{status.packets_seen}</div>
          </div>
          <div className="bg-[var(--color-accent-soft)] rounded-xl p-3">
            <div className="text-xs text-[var(--color-ink-dim)]">Remote hosts seen</div>
            <div className="text-lg font-bold text-[var(--color-ink)]">{status.hosts_seen}</div>
          </div>
          <div className="bg-[var(--color-accent-soft)] rounded-xl p-3">
            <div className="text-xs text-[var(--color-ink-dim)]">First forecast after</div>
            <div className="text-lg font-bold text-[var(--color-ink)]">1 window (30 s) per host</div>
          </div>
        </div>
      )}

      {status?.running && hostRows.length > 0 && (
        <div className="mt-4">
          <div className="flex items-center justify-between flex-wrap gap-2 mb-2">
            <h4 className="text-sm font-semibold text-[var(--color-ink)]">
              {showAll ? 'All Remote Hosts' : 'Flagged Hosts'}
            </h4>
            <label className="flex items-center gap-2 text-xs text-[var(--color-ink-dim)] cursor-pointer select-none">
              <input type="checkbox" checked={showAll} onChange={(e) => setShowAll(e.target.checked)} className="accent-[var(--color-accent)]" />
              Show all traffic (including benign / still building history)
            </label>
          </div>
          {!showAll && hiddenCount > 0 && (
            <p className="text-xs text-[var(--color-ink-faint)] mb-2">
              {hiddenCount} host{hiddenCount === 1 ? '' : 's'} hidden (benign or no prediction yet) — check "show all" to see them
            </p>
          )}
          {visibleRows.length === 0 ? (
            <div className="text-xs text-[var(--color-ink-faint)] bg-[var(--color-accent-soft)]/40 rounded-xl px-4 py-4 text-center">
              nothing flagged yet — {hostRows.length} host{hostRows.length === 1 ? '' : 's'} being watched, all benign or still building history
            </div>
          ) : (
          <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-[var(--color-ink-faint)] text-xs uppercase tracking-wide">
                <th className="py-2 pr-4 font-medium">Remote host</th>
                <th className="py-2 pr-4 font-medium">History</th>
                <th className="py-2 pr-4 font-medium">Predicted action</th>
                <th className="py-2 pr-4 font-medium">Infiltration prob.</th>
                <th className="py-2 pr-4 font-medium">Ports touched (all windows)</th>
                <th className="py-2 pr-4 font-medium">Last updated</th>
              </tr>
            </thead>
            <tbody>
              {visibleRows.map((row) => {
                const r = row.latest;
                const hasPrediction = r.predicted_stage != null;
                const ip = row.hostId.replace('live:', '');
                const isExpanded = expandedHost === row.hostId;
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
                        {hasPrediction ? `window ${r.window_idx}` : `${Math.min(row.windowCount, SEQ_LEN)}/${SEQ_LEN} windows`}
                      </td>
                      <td className="py-2.5 pr-4">
                        {hasPrediction ? (
                          <span className="flex items-center gap-1.5">
                            <span className="w-1.5 h-1.5 rounded-full" style={{ background: STAGE_COLORS[r.predicted_stage!] ?? '#999' }} />
                            {STAGE_LABELS[r.predicted_stage!] ?? r.predicted_stage}
                          </span>
                        ) : (
                          <span className="text-[var(--color-ink-faint)]">building history…</span>
                        )}
                      </td>
                      <td className="py-2.5 pr-4 font-semibold text-[var(--color-ink)]">
                        {r.infiltration_probability_world_model != null ? `${(r.infiltration_probability_world_model * 100).toFixed(1)}%` : '—'}
                      </td>
                      <td className="py-2.5 pr-4 text-xs text-[var(--color-ink-dim)]">
                        {row.allPorts.size > 0 ? Array.from(row.allPorts).sort((a, b) => a - b).join(', ') : '—'}
                      </td>
                      <td className="py-2.5 pr-4 text-xs text-[var(--color-ink-dim)] whitespace-nowrap">
                        {formatTimestamp(r.timestamp)}
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

      {status?.running && hostsWithPredictions.length > 0 && (
        <div className="mt-6 pt-5 border-t border-[var(--color-accent-soft)]">
          <div className="flex items-center justify-between flex-wrap gap-3 mb-2">
            <div>
              <h4 className="text-sm font-semibold text-[var(--color-ink)]">Infiltration Probability Over Time (this host)</h4>
              <p className="text-xs text-[var(--color-ink-dim)] mt-0.5">
                Each real prediction as it happened -- dot color is the predicted action at that window
              </p>
            </div>
            <select
              value={graphHost ?? ''}
              onChange={(e) => setGraphHost(e.target.value)}
              className="px-3 py-1.5 rounded-xl border border-[var(--color-accent-soft)] text-xs bg-white text-[var(--color-ink)]"
            >
              {hostsWithPredictions.map((h) => (
                <option key={h} value={h}>{h.replace('live:', '')}</option>
              ))}
            </select>
          </div>
          <LiveTrajectoryChart data={graphSeries} />
        </div>
      )}
    </div>
  );
}
