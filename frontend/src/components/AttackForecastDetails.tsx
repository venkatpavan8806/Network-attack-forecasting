import type { AttackMapping } from '../types';
import { STAGE_COLORS, STAGE_LABELS } from '../api';
import { AlertIcon } from '../icons';

export default function AttackForecastDetails({ action, mapping }: { action: string; mapping: AttackMapping }) {
  if (action === 'benign') {
    return (
      <div className="rounded-xl bg-[var(--color-good)]/10 p-4 text-sm text-[var(--color-good)] font-medium">
        No attack predicted at this window — traffic looks normal.
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-[var(--color-accent-soft)] overflow-hidden">
      <div className="px-4 py-3 flex items-center gap-2" style={{ background: `${STAGE_COLORS[action] ?? '#999'}18` }}>
        <span style={{ color: STAGE_COLORS[action] ?? '#999' }}>
          <AlertIcon size={16} />
        </span>
        <div>
          <div className="text-sm font-semibold text-[var(--color-ink)]">
            Predicted attack: {STAGE_LABELS[action] ?? action}
          </div>
          {mapping.technique_id && (
            <div className="text-xs text-[var(--color-ink-dim)]">
              {mapping.technique_id} — {mapping.technique_name} &middot; {mapping.tactic}
            </div>
          )}
        </div>
      </div>
      <div className="divide-y divide-[var(--color-accent-soft)]">
        <div className="px-4 py-3">
          <div className="text-[10px] uppercase tracking-wide text-[var(--color-ink-faint)] font-semibold mb-1">Likely tools</div>
          <p className="text-sm text-[var(--color-ink)]">{mapping.likely_tools ?? 'not yet available'}</p>
        </div>
        <div className="px-4 py-3">
          <div className="text-[10px] uppercase tracking-wide text-[var(--color-ink-faint)] font-semibold mb-1">System state if this succeeds</div>
          <p className="text-sm text-[var(--color-ink)]">{mapping.likely_system_state ?? 'not yet available'}</p>
        </div>
      </div>
      <div className="px-4 py-2.5 bg-[var(--color-accent-soft)]/40 text-[10px] text-[var(--color-ink-faint)]">
        The model predicts the action from traffic alone; tools and system-state are curated reference context attached to that prediction, not something the model observes directly.
      </div>
    </div>
  );
}
