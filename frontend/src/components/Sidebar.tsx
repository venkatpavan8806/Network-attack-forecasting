import type { FC } from 'react';
import { HomeIcon, RadarIcon, EyeIcon, BarsIcon, NetworkIcon } from '../icons';
import type { TabKey } from '../App';

const items: { key: TabKey; icon: FC<{ size?: number; className?: string }>; label: string; disabled?: boolean }[] = [
  { key: 'overview', icon: HomeIcon, label: 'Overview' },
  { key: 'forecasts', icon: RadarIcon, label: 'Forecasts' },
  { key: 'explainability', icon: EyeIcon, label: 'Explainability' },
  { key: 'benchmarks', icon: BarsIcon, label: 'Benchmarks' },
  { key: 'digital_twin', icon: NetworkIcon, label: 'Digital Twin' },
];

export default function Sidebar({ active, onChange }: { active: TabKey; onChange: (t: TabKey) => void }) {
  return (
    <aside className="hidden md:flex w-20 shrink-0 flex-col items-center py-6 gap-6">
      <div className="w-10 h-10 rounded-2xl bg-[var(--color-accent)] flex items-center justify-center text-white font-bold text-lg mb-4">
        C
      </div>
      <nav className="flex flex-col gap-3">
        {items.map(({ key, icon: Icon, label, disabled }) => {
          const isActive = active === key;
          return (
            <button
              key={key}
              title={disabled ? `${label} (deferred — Priority 3, not built in this pass)` : label}
              disabled={disabled}
              onClick={() => !disabled && onChange(key)}
              className={`group relative w-11 h-11 rounded-full flex items-center justify-center transition
                ${isActive ? 'bg-[var(--color-accent)] text-white shadow-lg shadow-[var(--color-accent)]/30' : 'text-[var(--color-ink-dim)] hover:bg-[var(--color-accent-soft)]'}
                ${disabled ? 'opacity-35 cursor-not-allowed hover:bg-transparent' : ''}`}
            >
              <Icon size={19} />
            </button>
          );
        })}
      </nav>
    </aside>
  );
}
