import { useState } from 'react';
import Sidebar from './components/Sidebar';
import Topbar from './components/Topbar';
import Overview from './pages/Overview';
import Forecasts from './pages/Forecasts';
import Explainability from './pages/Explainability';
import Benchmarks from './pages/Benchmarks';
import DigitalTwin from './pages/DigitalTwin';

export type TabKey = 'overview' | 'forecasts' | 'explainability' | 'benchmarks' | 'digital_twin';

export default function App() {
  const [tab, setTab] = useState<TabKey>('overview');
  const [selectedHost, setSelectedHost] = useState<string | null>(null);

  function goToHost(hostId: string) {
    setSelectedHost(hostId);
    setTab('forecasts');
  }

  return (
    <div className="min-h-screen flex bg-[var(--color-page)]">
      <Sidebar active={tab} onChange={setTab} />
      <main className="flex-1 px-4 sm:px-8 py-6 max-w-[1400px] mx-auto w-full">
        <Topbar active={tab} onChange={setTab} />
        <div className="mt-6">
          {tab === 'overview' && <Overview onNavigateToHost={goToHost} />}
          {tab === 'forecasts' && <Forecasts selectedHost={selectedHost} onSelectHost={setSelectedHost} />}
          {tab === 'explainability' && <Explainability selectedHost={selectedHost} />}
          {tab === 'benchmarks' && <Benchmarks />}
          {tab === 'digital_twin' && <DigitalTwin selectedHost={selectedHost} onSelectHost={setSelectedHost} />}
        </div>
      </main>
    </div>
  );
}
