import type { StageBreakdown } from '../types';
import { STAGE_COLORS, STAGE_LABELS } from '../api';

export default function AttackStageBreakdownPanel({ data }: { data: StageBreakdown | null }) {
  if (!data || data.total === 0) {
    return <div className="text-sm text-[var(--color-ink-faint)] py-6 text-center">not yet available — run inference to populate</div>;
  }
  const entries = Object.entries(data.counts).sort((a, b) => b[1] - a[1]);
  return (
    <div className="flex flex-col gap-4">
      {entries.map(([stage, count]) => {
        const pct = (count / data.total) * 100;
        return (
          <div key={stage} className="flex items-center gap-3">
            <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: STAGE_COLORS[stage] ?? '#999' }} />
            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between text-xs mb-1">
                <span className="font-medium text-[var(--color-ink)] truncate">{STAGE_LABELS[stage] ?? stage}</span>
                <span className="text-[var(--color-ink-dim)]">{pct.toFixed(1)}%</span>
              </div>
              <div className="h-1.5 rounded-full bg-[var(--color-accent-soft)] overflow-hidden">
                <div
                  className="h-full rounded-full"
                  style={{ width: `${pct}%`, background: STAGE_COLORS[stage] ?? '#999' }}
                />
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
