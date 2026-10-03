import type { HighestRiskHost } from '../types';
import { STAGE_COLORS, STAGE_LABELS } from '../api';
import { AlertIcon, ClockIcon } from '../icons';

export default function HighestRiskCard({ host, onView }: { host: HighestRiskHost | null; onView: (hostId: string) => void }) {
  if (!host) {
    return (
      <div className="card p-6 flex items-center justify-center text-sm text-[var(--color-ink-faint)] h-full min-h-[140px]">
        not yet available — no inference has run yet
      </div>
    );
  }
  const stage = host.predicted_stage ?? 'benign';
  const timeAgo = Math.max(0, Math.round((Date.now() - new Date(host.created_at).getTime()) / 1000));

  return (
    <div className="card p-6 flex flex-col gap-4">
      <div className="flex items-center gap-3">
        <div
          className="w-11 h-11 rounded-full flex items-center justify-center shrink-0"
          style={{ background: `${STAGE_COLORS[stage] ?? '#999'}22`, color: STAGE_COLORS[stage] ?? '#999' }}
        >
          <AlertIcon size={19} />
        </div>
        <div className="min-w-0">
          <div className="font-semibold text-[var(--color-ink)] truncate">{host.host_id}</div>
          <div className="text-xs text-[var(--color-ink-dim)]">window #{host.window_idx}</div>
        </div>
      </div>

      <div>
        <div className="text-xs text-[var(--color-ink-faint)] mb-1">Predicted stage</div>
        <div className="flex items-center gap-2">
          <span className="px-2.5 py-1 rounded-full text-xs font-semibold" style={{ background: `${STAGE_COLORS[stage] ?? '#999'}22`, color: STAGE_COLORS[stage] ?? '#999' }}>
            {STAGE_LABELS[stage] ?? stage}
          </span>
          <span className="text-sm font-bold text-[var(--color-ink)]">
            {(host.infiltration_probability * 100).toFixed(1)}% infiltration
          </span>
        </div>
      </div>

      {host.attack_mapping?.technique_id && (
        <div className="text-xs text-[var(--color-ink-dim)]">
          MITRE ATT&amp;CK: <span className="font-medium text-[var(--color-ink)]">{host.attack_mapping.technique_id}</span> — {host.attack_mapping.technique_name}
        </div>
      )}
      {host.attack_mapping?.likely_tools && (
        <div className="text-xs text-[var(--color-ink-faint)]">
          Likely tools: {host.attack_mapping.likely_tools}
        </div>
      )}

      <div className="flex items-center gap-1.5 text-xs text-[var(--color-ink-faint)]">
        <ClockIcon size={13} /> logged {timeAgo}s ago
      </div>

      <button
        onClick={() => onView(host.host_id)}
        className="mt-1 self-start px-4 py-2 rounded-full bg-[var(--color-accent)] text-white text-sm font-medium hover:opacity-90 transition"
      >
        View Trajectory
      </button>
    </div>
  );
}
