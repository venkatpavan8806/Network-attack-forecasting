import type { ForecastLogRow } from '../types';
import { STAGE_COLORS, STAGE_LABELS } from '../api';

export default function ExplainabilityDigest({
  rows, onSelect,
}: {
  rows: ForecastLogRow[];
  onSelect: (hostId: string) => void;
}) {
  const worldModelRows = rows.filter((r) => r.model === 'world_model_lstm' && r.infiltration_probability >= 0.2).slice(0, 5);
  if (!worldModelRows.length) {
    return <div className="text-sm text-[var(--color-ink-faint)] py-6 text-center">not yet available</div>;
  }
  return (
    <div className="flex flex-col divide-y divide-[var(--color-accent-soft)]">
      {worldModelRows.map((r, i) => (
        <div key={i} className="flex items-center gap-3 py-3">
          <span
            className="w-9 h-9 rounded-full flex items-center justify-center shrink-0 text-xs font-bold"
            style={{ background: `${STAGE_COLORS[r.predicted_stage ?? 'benign']}22`, color: STAGE_COLORS[r.predicted_stage ?? 'benign'] }}
          >
            {Math.round(r.infiltration_probability * 100)}%
          </span>
          <div className="min-w-0 flex-1">
            <div className="text-sm font-medium text-[var(--color-ink)] truncate">
              {r.host_id} &middot; {STAGE_LABELS[r.predicted_stage ?? 'benign']}
            </div>
            <div className="text-xs text-[var(--color-ink-dim)] truncate">
              window #{r.window_idx} — attention + input-gradient saliency on past {8} windows
            </div>
          </div>
          <button
            onClick={() => onSelect(r.host_id)}
            className="text-xs font-medium text-[var(--color-accent)] shrink-0 hover:underline"
          >
            View details
          </button>
        </div>
      ))}
    </div>
  );
}
