import type { ForecastLogRow } from '../types';
import { STAGE_COLORS, STAGE_LABELS } from '../api';

export default function ForecastLogTable({ rows }: { rows: ForecastLogRow[] }) {
  if (!rows.length) {
    return <div className="text-sm text-[var(--color-ink-faint)] py-6 text-center">not yet available</div>;
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-[var(--color-ink-faint)] text-xs uppercase tracking-wide">
            <th className="py-2 pr-4 font-medium">Host</th>
            <th className="py-2 pr-4 font-medium">Window</th>
            <th className="py-2 pr-4 font-medium">Model</th>
            <th className="py-2 pr-4 font-medium">Predicted Stage</th>
            <th className="py-2 pr-4 font-medium">Infiltration Prob.</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-t border-[var(--color-accent-soft)]">
              <td className="py-2.5 pr-4 font-medium text-[var(--color-ink)]">{r.host_id}</td>
              <td className="py-2.5 pr-4 text-[var(--color-ink-dim)]">{r.window_idx}</td>
              <td className="py-2.5 pr-4">
                <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${r.model === 'world_model_lstm' ? 'bg-[var(--color-accent-soft)] text-[var(--color-accent)]' : 'bg-[var(--color-bad)]/10 text-[var(--color-bad)]'}`}>
                  {r.model === 'world_model_lstm' ? 'World Model' : 'Baseline'}
                </span>
              </td>
              <td className="py-2.5 pr-4">
                {r.predicted_stage ? (
                  <span className="flex items-center gap-1.5">
                    <span className="w-1.5 h-1.5 rounded-full" style={{ background: STAGE_COLORS[r.predicted_stage] ?? '#999' }} />
                    {STAGE_LABELS[r.predicted_stage] ?? r.predicted_stage}
                  </span>
                ) : (
                  <span className="text-[var(--color-ink-faint)]">n/a (binary only)</span>
                )}
              </td>
              <td className="py-2.5 pr-4 font-semibold text-[var(--color-ink)]">
                {(r.infiltration_probability * 100).toFixed(1)}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
