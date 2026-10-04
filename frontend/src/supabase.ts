import { createClient } from '@supabase/supabase-js';

// Supabase project settings come from environment variables (Vercel ->
// Project -> Settings -> Environment Variables, or frontend/.env.local),
// never from the source code.
function projectOrigin(raw: string | undefined): string | undefined {
  if (!raw) return undefined;
  try {
    return new URL(raw.trim()).origin; // tolerate ".../rest/v1/" or a trailing slash
  } catch {
    return raw.trim();
  }
}

const url = projectOrigin(import.meta.env.VITE_SUPABASE_URL as string | undefined);
const key = import.meta.env.VITE_SUPABASE_ANON_KEY as string | undefined;

export const supabaseConfigured = Boolean(url && key);

export const supabase = supabaseConfigured ? createClient(url!, key!) : null;
