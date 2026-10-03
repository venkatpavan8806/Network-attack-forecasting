import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { STAGE_COLORS, STAGE_LABELS } from '../api';

export interface LiveTrajectoryPoint {
  window_idx: number;
  probability: number;
  action: string | null;
  timestamp: string;
}

function CustomTooltip({ active, payload }: any) {
  if (!active || !payload || !payload.length) return null;
  const p = payload[0].payload as LiveTrajectoryPoint;
  return (
    <div className="relative bg-[#211f36] text-white text-xs rounded-xl px-3 py-2.5 shadow-xl min-w-[160px]">
      <div className="font-semibold mb-1 text-white/90">window {p.window_idx}</div>
      <div className="flex items-center justify-between gap-4 py-0.5">
        <span className="text-white/70">Infiltration prob.</span>
        <span className="font-semibold">{(p.probability * 100).toFixed(1)}%</span>
      </div>
      {p.action && (
        <div className="flex items-center justify-between gap-4 py-0.5">
          <span className="text-white/70">Action</span>
          <span className="font-semibold">{STAGE_LABELS[p.action] ?? p.action}</span>
        </div>
      )}
      <div className="text-white/50 mt-1">{new Date(p.timestamp).toLocaleTimeString(undefined, { hour12: false })}</div>
      <div className="absolute left-1/2 -bottom-1.5 -translate-x-1/2 w-3 h-3 bg-[#211f36] rotate-45" />
    </div>
  );
}

export default function LiveTrajectoryChart({ data }: { data: LiveTrajectoryPoint[] }) {
  if (data.length < 2) {
    return (
      <div className="h-56 flex items-center justify-center text-sm text-[var(--color-ink-faint)]">
        not enough predicted windows yet to plot a trend
      </div>
    );
  }
  return (
    <div className="w-full h-56">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
          <CartesianGrid stroke="#eceafa" vertical={false} />
          <XAxis dataKey="window_idx" tick={{ fontSize: 11, fill: '#8b8a9e' }} axisLine={false} tickLine={false} tickFormatter={(v) => `w${v}`} />
          <YAxis
            tickFormatter={(v) => `${Math.round(v * 100)}%`}
            tick={{ fontSize: 11, fill: '#8b8a9e' }}
            axisLine={false}
            tickLine={false}
            domain={[0, 1]}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ stroke: '#d8d4f0', strokeWidth: 1 }} />
          <Line
            type="monotone"
            dataKey="probability"
            stroke="#6c5dd3"
            strokeWidth={2.5}
            dot={(props: any) => {
              const point = props.payload as LiveTrajectoryPoint;
              const color = point.action ? STAGE_COLORS[point.action] ?? '#6c5dd3' : '#6c5dd3';
              return <circle key={props.key} cx={props.cx} cy={props.cy} r={4} fill={color} stroke="white" strokeWidth={1.5} />;
            }}
            activeDot={{ r: 6 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
