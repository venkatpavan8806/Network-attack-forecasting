import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

export interface TrajectoryPoint {
  step: string;
  worldModel: number | null;
  baseline: number | null;
}

function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload || !payload.length) return null;
  return (
    <div className="relative bg-[#211f36] text-white text-xs rounded-xl px-3 py-2.5 shadow-xl min-w-[150px]">
      <div className="font-semibold mb-1.5 text-white/90">{label}</div>
      {payload.map((p: any) => (
        <div key={p.dataKey} className="flex items-center justify-between gap-4 py-0.5">
          <span className="flex items-center gap-1.5 text-white/70">
            <span className="w-2 h-2 rounded-full" style={{ background: p.color }} />
            {p.dataKey === 'worldModel' ? 'World Model' : 'Baseline (LR)'}
          </span>
          <span className="font-semibold">{p.value != null ? `${(p.value * 100).toFixed(1)}%` : '—'}</span>
        </div>
      ))}
      <div className="absolute left-1/2 -bottom-1.5 -translate-x-1/2 w-3 h-3 bg-[#211f36] rotate-45" />
    </div>
  );
}

export default function TrajectoryChart({ data }: { data: TrajectoryPoint[] }) {
  return (
    <div className="w-full h-72">
      <div className="flex items-center justify-end gap-4 mb-1 text-xs text-[var(--color-ink-dim)]">
        <span className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-[var(--color-accent)]" /> World Model
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-[var(--color-bad)]" /> Baseline
        </span>
      </div>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
          <CartesianGrid stroke="#eceafa" vertical={false} />
          <XAxis dataKey="step" tick={{ fontSize: 11, fill: '#8b8a9e' }} axisLine={false} tickLine={false} />
          <YAxis
            tickFormatter={(v) => `${Math.round(v * 100)}%`}
            tick={{ fontSize: 11, fill: '#8b8a9e' }}
            axisLine={false}
            tickLine={false}
            domain={[0, 1]}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ stroke: '#d8d4f0', strokeWidth: 1 }} />
          <Line type="monotone" dataKey="worldModel" stroke="#6c5dd3" strokeWidth={2.5} dot={{ r: 3 }} activeDot={{ r: 5 }} connectNulls />
          <Line type="monotone" dataKey="baseline" stroke="#ff6b81" strokeWidth={2.5} dot={{ r: 3 }} activeDot={{ r: 5 }} connectNulls />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
