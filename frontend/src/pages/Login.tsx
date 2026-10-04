import { useState, type FormEvent } from 'react';
import { supabase } from '../auth';

type Mode = 'signin' | 'signup' | 'reset';

export default function Login() {
  const [mode, setMode] = useState<Mode>('signin');
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
      } else if (mode === 'signup') {
        const { data, error } = await supabase.auth.signUp({
          email, password, options: { emailRedirectTo: window.location.origin },
        });
        if (error) throw error;
        if (!data.session) setNotice('Account created. Check your inbox to confirm your email, then sign in.');
      } else {
        const { error } = await supabase.auth.resetPasswordForEmail(email, { redirectTo: window.location.origin });
        if (error) throw error;
        setNotice('If that email has an account, a password-reset link is on its way.');
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function guest() {
    if (!supabase) return;
    setBusy(true);
    setError(null);
    const { error } = await supabase.auth.signInAnonymously();
    if (error) setError(`${error.message} (guest access must be enabled in Supabase: Authentication -> Sign In / Providers -> Anonymous)`);
    setBusy(false);
  }

  const title = mode === 'signin' ? 'Sign in' : mode === 'signup' ? 'Create your account' : 'Reset password';

  return (
    <div className="min-h-screen flex items-center justify-center bg-[var(--color-page)] px-4">
      <div className="card p-8 w-full max-w-md">
        <div className="mb-6">
          <div className="text-xs font-semibold uppercase tracking-wide text-[var(--color-accent)]">Network Attack Forecasting</div>
          <h1 className="text-2xl font-bold text-[var(--color-ink)] mt-1">{title}</h1>
          <p className="text-sm text-[var(--color-ink-dim)] mt-1">
            Each account gets its own private workspace: your sensors, captures and forecasts are visible only to you.
          </p>
        </div>

        <form onSubmit={submit} className="flex flex-col gap-3">
          <label className="text-xs text-[var(--color-ink-dim)]">
            Email
            <input type="email" required autoComplete="email" value={email} onChange={(e) => setEmail(e.target.value)}
              className="mt-1 w-full px-3 py-2 rounded-xl border border-[var(--color-accent-soft)] text-sm bg-white text-[var(--color-ink)]" />
          </label>
          {mode !== 'reset' && (
            <label className="text-xs text-[var(--color-ink-dim)]">
              Password
              <input type="password" required minLength={6} value={password} onChange={(e) => setPassword(e.target.value)}
                autoComplete={mode === 'signup' ? 'new-password' : 'current-password'}
                className="mt-1 w-full px-3 py-2 rounded-xl border border-[var(--color-accent-soft)] text-sm bg-white text-[var(--color-ink)]" />
            </label>
          )}
          {error && <div className="text-sm rounded-xl p-3 bg-[var(--color-bad)]/10 text-[var(--color-bad)]">{error}</div>}
          {notice && <div className="text-sm rounded-xl p-3 bg-[var(--color-good)]/10 text-[#1f8a5c]">{notice}</div>}
          <button type="submit" disabled={busy}
            className="mt-1 px-4 py-2.5 rounded-full bg-[var(--color-accent)] text-white text-sm font-medium disabled:opacity-60">
            {busy ? 'Please wait…' : title}
          </button>
        </form>

        <div className="flex flex-wrap justify-between gap-2 mt-4 text-xs">
          {mode !== 'signin' && <button onClick={() => setMode('signin')} className="text-[var(--color-accent)]">Have an account? Sign in</button>}
          {mode !== 'signup' && <button onClick={() => setMode('signup')} className="text-[var(--color-accent)]">New here? Create an account</button>}
          {mode === 'signin' && <button onClick={() => setMode('reset')} className="text-[var(--color-ink-dim)]">Forgot password?</button>}
        </div>

        <div className="mt-6 pt-5 border-t border-[var(--color-accent-soft)]">
          <button onClick={guest} disabled={busy}
            className="w-full px-4 py-2.5 rounded-full bg-[var(--color-accent-soft)] text-[var(--color-accent)] text-sm font-medium disabled:opacity-60">
            Continue as guest
          </button>
          <p className="text-[11px] text-[var(--color-ink-faint)] mt-2 text-center">
            A guest gets a fresh, empty workspace that lasts while this browser keeps its session.
          </p>
        </div>
      </div>
    </div>
  );
}
