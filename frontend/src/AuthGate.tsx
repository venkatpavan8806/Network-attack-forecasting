import { useEffect, useState, type ReactNode } from 'react';
import type { Session } from '@supabase/supabase-js';
import { supabase, supabaseConfigured } from './supabase';
import Login from './pages/Login';

/** Shows the login page until the user is signed in, then the dashboard.
 *  Local development without Supabase settings: no login (the backend then
 *  treats every request as one local user). Production always has them. */
export default function AuthGate({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!supabase) {
      setLoading(false);
      return;
    }
    supabase.auth.getSession().then(({ data }) => {
      setSession(data.session);
      setLoading(false);
    });
    const { data } = supabase.auth.onAuthStateChange((_event, s) => setSession(s));
    return () => data.subscription.unsubscribe();
  }, []);

  if (loading) {
    return <div className="min-h-screen flex items-center justify-center text-sm text-[var(--color-ink-faint)]">loading…</div>;
  }
  if (!supabaseConfigured) return <>{children}</>;
  if (!session) return <Login />;
  return <>{children}</>;
}
