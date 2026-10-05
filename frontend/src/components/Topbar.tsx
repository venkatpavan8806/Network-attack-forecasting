import { BellIcon, LogIcon } from '../icons';
import type { TabKey } from '../App';

const tabs: { key: TabKey; label: string; disabled?: boolean }[] = [
  { key: 'overview', label: 'Overview' },
  { key: 'forecasts', label: 'Forecasts' },
  { key: 'explainability', label: 'Explainability' },
  { key: 'benchmarks', label: 'Benchmarks' },
  { key: 'digital_twin', label: 'Digital Twin' },
];

function greeting() {
  const h = new Date().getHours();
  if (h < 12) return 'Good morning';
  if (h < 18) return 'Good afternoon';
  return 'Good evening';
}

export default function Topbar({ active, onChange }: { active: TabKey; onChange: (t: TabKey) => void }) {
  return (
    <div className="flex flex-col gap-5">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-[var(--color-ink)]">{greeting()}, Team CARIBBEAN</h1>
          <p className="text-sm text-[var(--color-ink-dim)] mt-1">
            here's what your network looked like overnight
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button className="w-10 h-10 rounded-full bg-white card flex items-center justify-center text-[var(--color-ink-dim)]">
            <BellIcon size={17} />
          </button>
          <button className="w-10 h-10 rounded-full bg-white card flex items-center justify-center text-[var(--color-ink-dim)]">
            <LogIcon size={17} />
          </button>
          <div className="w-10 h-10 rounded-full bg-[var(--color-accent)] flex items-center justify-center text-white font-semibold text-sm">
            SIH
          </div>
        </div>
      </div>

      <nav className="flex items-center gap-2 flex-wrap">
        {tabs.map((t) => {
          const isActive = active === t.key;
          return (
            <button
              key={t.key}
              disabled={t.disabled}
              onClick={() => !t.disabled && onChange(t.key)}
              title={t.disabled ? 'Deferred — Priority 3, not built in this pass' : undefined}
              className={`px-4 py-2 rounded-full text-sm font-medium transition
                ${isActive ? 'bg-[var(--color-accent)] text-white shadow-md shadow-[var(--color-accent)]/25' : 'text-[var(--color-ink-dim)] hover:bg-white/70'}
                ${t.disabled ? 'opacity-40 cursor-not-allowed' : ''}`}
            >
              {t.label}
            </button>
          );
        })}
      </nav>
    </div>
  );
}
