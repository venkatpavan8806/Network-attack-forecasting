import SensorsPanel from '../components/SensorsPanel';
import LiveCapturePanel from '../components/LiveCapturePanel';
import DataSourcesPanel from '../components/DataSourcesPanel';

export default function Capture({ onOpenHost }: { onOpenHost: (hostId: string) => void }) {
  return (
    <div className="flex flex-col gap-6">
      <SensorsPanel />
      <LiveCapturePanel onSelectHost={onOpenHost} />
      <DataSourcesPanel onSelectHost={onOpenHost} />
    </div>
  );
}
