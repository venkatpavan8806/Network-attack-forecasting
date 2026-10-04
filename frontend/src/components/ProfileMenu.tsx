import { useEffect, useRef, useState } from 'react';
import { supabase } from '../supabase';

/** Round profile button (signed-in user's initial) with a small menu: email + sign out. */
export default function ProfileMenu() {
  const [email, setEmail] = useState<string | null>(null);
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    supabase?.auth.getUser().then(({ data }) => setEmail(data.user?.email ?? null));
  }, []);

  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', close);
    return () => document.removeEventListener('mousedown', close);
  }, [open]);

  const initial = (email?.[0] ?? '?').toUpperCase();

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen((o) => !o)}
        title={email ?? 'Profile'}
        className="w-10 h-10 rounded-full bg-[var(--color-accent)] flex items-center justify-center text-white font-semibold text-sm"
      >
        {initial}
      </button>
      {open && (
        <div className="absolute right-0 mt-2 w-64 card p-4 z-50 shadow-lg">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 shrink-0 rounded-full bg-[var(--color-accent)] flex items-center justify-center text-white font-semibold text-sm">
              {initial}
            </div>
            <div className="min-w-0">
              <div className="text-xs text-[var(--color-ink-faint)]">Signed in as</div>
              <div className="text-sm font-medium text-[var(--color-ink)] truncate">{email ?? '—'}</div>
            </div>
          </div>
          <button
            onClick={() => supabase?.auth.signOut()}
            className="mt-4 w-full px-4 py-2 rounded-full bg-[var(--color-accent-soft)] text-[var(--color-accent)] text-sm font-medium"
          >
            Sign out
          </button>
        </div>
      )}
    </div>
  );
}
