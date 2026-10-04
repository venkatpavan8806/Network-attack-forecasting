import { createContext, useContext, useEffect, useState, type ReactNode } from 'react';
import { createClient, type Session, type SupabaseClient } from '@supabase/supabase-js';

/**
 * Who is using the site.
 *
 * Production: Supabase Auth (email + password, optional guest / anonymous
 * sign-in). Every API call carries the Supabase access token; the backend
 * verifies it and scopes all data to that user, so every account starts
 * with an empty workspace.
 *
 * Local development without Supabase (VITE_SUPABASE_URL unset): each
 * browser gets a random id stored in localStorage and sent as X-Dev-User,
 * so even locally two browsers never share a workspace.
 */

const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL as string | undefined;
const SUPABASE_KEY = (import.meta.env.VITE_SUPABASE_ANON_KEY ?? import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY) as string | undefined;

export const supabase: SupabaseClient | null =
  SUPABASE_URL && SUPABASE_KEY ? createClient(SUPABASE_URL, SUPABASE_KEY) : null;

export const AUTH_MODE: 'supabase' | 'dev' = supabase ? 'supabase' : 'dev';

export interface AppUser {
  id: string;
  email: string | null;
  isAnonymous: boolean;
}

interface AuthState {
  user: AppUser | null;
  loading: boolean;
  signOut: () => Promise<void>;
}

const AuthContext = createContext<AuthState>({ user: null, loading: true, signOut: async () => {} });

const DEV_KEY = 'nadf-dev-user';

function devUserId(): string {
  try {
    let id = localStorage.getItem(DEV_KEY);
    if (!id) {
      id = `dev-${crypto.randomUUID()}`;
      localStorage.setItem(DEV_KEY, id);
    }
    return id;
  } catch {
    return 'dev-local';
  }
}

let currentSession: Session | null = null;

/** Headers that identify the caller to the backend (used by api.ts). */
export function authHeaders(): Record<string, string> {
  if (AUTH_MODE === 'dev') return { 'X-Dev-User': devUserId() };
  return currentSession ? { Authorization: `Bearer ${currentSession.access_token}` } : {};
}

function toUser(session: Session | null): AppUser | null {
  if (!session) return null;
  const u = session.user;
  return { id: u.id, email: u.email ?? null, isAnonymous: Boolean((u as { is_anonymous?: boolean }).is_anonymous) };
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AppUser | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!supabase) {
      setUser({ id: devUserId(), email: null, isAnonymous: true });
      setLoading(false);
      return;
    }
    supabase.auth.getSession().then(({ data }) => {
      currentSession = data.session;
      setUser(toUser(data.session));
      setLoading(false);
    });
    const { data: sub } = supabase.auth.onAuthStateChange((_event, session) => {
      currentSession = session;
      setUser(toUser(session));
    });
    return () => sub.subscription.unsubscribe();
  }, []);

  async function signOut() {
    if (supabase) {
      await supabase.auth.signOut();
    } else {
      try { localStorage.removeItem(DEV_KEY); } catch { /* ignore */ }
      setUser({ id: devUserId(), email: null, isAnonymous: true });
    }
  }

  return <AuthContext.Provider value={{ user, loading, signOut }}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  return useContext(AuthContext);
}
