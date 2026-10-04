import { useState, type FormEvent } from 'react';
import { supabase, supabaseConfigured } from '../supabase';

export default function Login() {
  const [mode, setMode] = useState<'signin' | 'signup'>('signin');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  async function submit(e: FormEvent) {
    e.preventDefault();
    if (!supabase) return;
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      if (mode === 'signin') {
        const { error } = await supabase.auth.signInWithPassword({ email, password });
        if (error) throw error;
      } else {
        const { data, error } = await supabase.auth.signUp({
          email, password, options: { emailRedirectTo: window.location.origin },
        });
        if (error) throw error;
        if (!data.session) setNotice('Account created. Check your email to confirm it, then sign in.');
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-[var(--color-page)] px-4">
      <div className="card p-8 w-full max-w-md">
        <h1 className="text-2xl font-bold text-[var(--color-ink)]">{mode === 'signin' ? 'Sign in' : 'Create account'}</h1>
        <p className="text-sm text-[var(--color-ink-dim)] mt-1 mb-6">NCIIPC Network Attack Forecast Console</p>

        {!supabaseConfigured && (
          <div className="mb-4 text-sm rounded-xl p-3 bg-[var(--color-bad)]/10 text-[var(--color-bad)]">
            Login is not configured: set VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY.
          </div>
        )}

        <form onSubmit={submit} className="flex flex-col gap-3">
          <label className="text-xs text-[var(--color-ink-dim)]">
            Email
            <input type="email" required autoComplete="email" value={email} onChange={(e) => setEmail(e.target.value)}
              className="mt-1 w-full px-3 py-2 rounded-xl border border-[var(--color-accent-soft)] text-sm bg-white text-[var(--color-ink)]" />
          </label>
          <label className="text-xs text-[var(--color-ink-dim)]">
            Password
            <input type="password" required minLength={6} value={password} onChange={(e) => setPassword(e.target.value)}
              autoComplete={mode === 'signin' ? 'current-password' : 'new-password'}
              className="mt-1 w-full px-3 py-2 rounded-xl border border-[var(--color-accent-soft)] text-sm bg-white text-[var(--color-ink)]" />
          </label>
          {error && <div className="text-sm rounded-xl p-3 bg-[var(--color-bad)]/10 text-[var(--color-bad)]">{error}</div>}
          {notice && <div className="text-sm rounded-xl p-3 bg-[var(--color-good)]/10 text-[#1f8a5c]">{notice}</div>}
          <button type="submit" disabled={busy || !supabaseConfigured}
            className="mt-1 px-4 py-2.5 rounded-full bg-[var(--color-accent)] text-white text-sm font-medium disabled:opacity-60">
            {busy ? 'Please wait…' : mode === 'signin' ? 'Sign in' : 'Create account'}
          </button>
        </form>

        <button onClick={() => { setMode(mode === 'signin' ? 'signup' : 'signin'); setError(null); setNotice(null); }}
          className="mt-4 text-xs text-[var(--color-accent)]">
          {mode === 'signin' ? 'New here? Create an account' : 'Already have an account? Sign in'}
        </button>
      </div>
    </div>
  );
}
