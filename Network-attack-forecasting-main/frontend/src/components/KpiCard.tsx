import type { ReactNode } from 'react';
import { ArrowUpIcon, ArrowDownIcon } from '../icons';

export default function KpiCard({
  icon, label, value, delta, deltaGood, iconColor,
}: {
  icon: ReactNode;
  label: string;
  value: string;
  delta?: string | null;
  deltaGood?: boolean;
  iconColor?: string;
}) {
  return (
    <div className="card p-6 flex flex-col gap-4 min-w-0">
      <div className="flex items-center gap-3">
        <div
          className="w-10 h-10 rounded-full flex items-center justify-center shrink-0"
          style={{ background: iconColor ? `${iconColor}22` : 'var(--color-accent-soft)', color: iconColor ?? 'var(--color-accent)' }}
        >
          {icon}
        </div>
        <span className="text-sm text-[var(--color-ink-dim)] font-medium truncate">{label}</span>
      </div>
      <div className="flex items-end justify-between gap-2">
        <span className="text-3xl font-bold text-[var(--color-ink)] truncate">{value}</span>
        {delta && (
          <span
            className={`flex items-center gap-0.5 text-xs font-semibold px-2 py-1 rounded-full shrink-0
              ${deltaGood ? 'bg-[var(--color-good)]/15 text-[var(--color-good)]' : 'bg-[var(--color-bad)]/15 text-[var(--color-bad)]'}`}
          >
            {deltaGood ? <ArrowUpIcon size={12} /> : <ArrowDownIcon size={12} />}
            {delta}
          </span>
        )}
      </div>
    </div>
  );
}
