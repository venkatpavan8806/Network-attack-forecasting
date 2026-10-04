import { useCallback, useEffect, useState } from 'react';
import Topbar from './components/Topbar';
import Overview from './pages/Overview';
import Forecasts from './pages/Forecasts';
import Explainability from './pages/Explainability';
import Benchmarks from './pages/Benchmarks';
import DigitalTwin from './pages/DigitalTwin';
import Capture from './pages/Capture';
import Login from './pages/Login';
import EmptyWorkspace from './components/EmptyWorkspace';
import { AUTH_MODE, useAuth } from './auth';
import { api } from './api';

export type TabKey = 'overview' | 'capture' | 'forecasts' | 'explainability' | 'benchmarks' | 'digital_twin';

// pages that only make sense once the user's workspace has traffic in it
const NEEDS_DATA: TabKey[] = ['overview', 'forecasts', 'explainability', 'digital_twin'];

export default function App() {
  const { user, loading } = useAuth();

  if (loading) {
    return <div className="min-h-screen flex items-center justify-center text-sm text-[var(--color-ink-faint)]">loading…</div>;
  }
  if (AUTH_MODE === 'supabase' && !user) return <Login />;
  // key on the user id: signing in as someone else remounts everything, so no
  // state from the previous account can leak into the new session
  return <Dashboard key={user?.id ?? 'anon'} />;
}

function Dashboard() {
  const [tab, setTab] = useState<TabKey>('overview');
  const [selectedHost, setSelectedHost] = useState<string | null>(null);
  const [hostCount, setHostCount] = useState<number | null>(null);
  const [backendError, setBackendError] = useState<string | null>(null);

  const refreshHosts = useCallback(() => {
    api.hosts()
      .then((h) => {
        setHostCount(h.length);
        setBackendError(null);
        setSelectedHost((cur) => (cur && h.includes(cur) ? cur : (h.find((x) => x.includes("attack")) ?? h[0] ?? null)));
      })
      .catch((e) => setBackendError(e?.response?.data?.detail ?? e?.message ?? 'backend unreachable'));
  }, []);

  useEffect(() => {
    refreshHosts();
    const t = window.setInterval(refreshHosts, 15000);
    return () => window.clearInterval(t);
  }, [refreshHosts, tab]);

  function goToHost(hostId: string) {
    setSelectedHost(hostId);
    setHostCount((c) => (c ? c : 1));
    setTab('forecasts');
  }

  const empty = hostCount === 0 && NEEDS_DATA.includes(tab);

  return (
    <div className="min-h-screen flex bg-[var(--color-page)]">
      <main className="flex-1 px-4 sm:px-8 py-6 max-w-[1400px] mx-auto w-full">
        <Topbar active={tab} onChange={setTab} />
        {backendError && (
          <div className="mt-4 text-sm rounded-xl p-3 bg-[var(--color-bad)]/10 text-[var(--color-bad)]">
            Cannot reach the analysis server: {backendError}. If it was idle it may take up to a minute to wake up — this page retries automatically.
          </div>
        )}
        <div className="mt-6">
          {empty ? (
            <EmptyWorkspace onGoToCapture={() => setTab('capture')} onSampleLoaded={(h) => { refreshHosts(); goToHost(h); }} />
          ) : (
            <>
              {tab === 'overview' && <Overview onNavigateToHost={goToHost} />}
              {tab === 'capture' && <Capture onOpenHost={goToHost} />}
              {tab === 'forecasts' && <Forecasts selectedHost={selectedHost} onSelectHost={setSelectedHost} />}
              {tab === 'explainability' && <Explainability selectedHost={selectedHost} />}
              {tab === 'benchmarks' && <Benchmarks />}
              {tab === 'digital_twin' && <DigitalTwin selectedHost={selectedHost} onSelectHost={setSelectedHost} />}
            </>
          )}
        </div>
      </main>
    </div>
  );
}
