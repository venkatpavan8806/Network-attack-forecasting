import { useEffect, useState } from 'react';
import { api } from '../api';
import type { Kpis, HighestRiskHost, ForecastLogRow, StageBreakdown } from '../types';
import KpiCard from '../components/KpiCard';
import HighestRiskCard from '../components/HighestRiskCard';
import CardHeader from '../components/CardHeader';
import AttackStageBreakdownPanel from '../components/AttackStageBreakdown';
import ForecastLogTable from '../components/ForecastLogTable';
import ExplainabilityDigest from '../components/ExplainabilityDigest';
import TrajectoryChart, { type TrajectoryPoint } from '../components/TrajectoryChart';
import { NetworkIcon, RadarIcon, AlertIcon, ClockIcon } from '../icons';

export default function Overview({ onNavigateToHost }: { onNavigateToHost: (hostId: string) => void }) {
  const [kpis, setKpis] = useState<Kpis | null>(null);
  const [highRisk, setHighRisk] = useState<HighestRiskHost | null>(null);
  const [log, setLog] = useState<ForecastLogRow[]>([]);
  const [breakdown, setBreakdown] = useState<StageBreakdown | null>(null);
  const [trajectory, setTrajectory] = useState<TrajectoryPoint[]>([]);

  useEffect(() => {
    api.kpis().then(setKpis).catch(() => {});
    api.highestRiskHost().then(setHighRisk).catch(() => {});
    api.forecastLog(30).then(setLog).catch(() => {});
    api.attackStageBreakdown().then(setBreakdown).catch(() => {});
  }, []);

  useEffect(() => {
    if (!highRisk) return;
    api.forecast(highRisk.host_id).then((f) => {
      const points: TrajectoryPoint[] = [
        { step: 'now', worldModel: f.infiltration_probability_world_model, baseline: f.infiltration_probability_baseline },
        ...f.rollout.infiltration_probs_world_model.map((p, i) => ({
          step: `t+${i + 1}`,
          worldModel: p,
          baseline: null,
        })),
      ];
      setTrajectory(points);
    }).catch(() => {});
  }, [highRisk]);

  return (
    <div className="flex flex-col gap-6">
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-5">
        <KpiCard
          icon={<NetworkIcon size={18} />}
          label="Hosts Monitored"
          value={kpis ? String(kpis.hosts_monitored) : '—'}
        />
        <KpiCard
          icon={<RadarIcon size={18} />}
          label="Forecasts Generated (today)"
          value={kpis ? String(kpis.forecasts_generated_today) : '—'}
        />
        <KpiCard
          icon={<AlertIcon size={18} />}
          label="High-Risk Trajectories"
          value={kpis ? String(kpis.high_risk_trajectories) : '—'}
          iconColor="var(--color-bad)"
        />
        <KpiCard
          icon={<ClockIcon size={18} />}
          label="Median Lead-Time (minutes)"
          value={kpis && kpis.median_lead_time_minutes != null ? kpis.median_lead_time_minutes.toFixed(2) : 'n/a'}
        />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <div className="xl:col-span-2 card p-6">
          <CardHeader title="Infiltration Probability Trajectory" subtitle="World model K-step rollout vs. static baseline, for the current highest-risk host" />
          {trajectory.length ? <TrajectoryChart data={trajectory} /> : (
            <div className="h-72 flex items-center justify-center text-sm text-[var(--color-ink-faint)]">not yet available</div>
          )}
        </div>
        <HighestRiskCard host={highRisk} onView={onNavigateToHost} />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <div className="card p-6">
          <CardHeader title="ATT&CK Stage Breakdown" subtitle="Share of recent windows per predicted stage" />
          <AttackStageBreakdownPanel data={breakdown} />
        </div>
        <div className="xl:col-span-2 card p-6">
          <CardHeader title="Recent Forecast Log" subtitle="Real rows from real inference runs" />
          <ForecastLogTable rows={log.slice(0, 10)} />
        </div>
      </div>

      <div className="card p-6">
        <CardHeader title="Recent Explainability Digest" subtitle="Attention + input-gradient saliency from the trained world model" />
        <ExplainabilityDigest rows={log} onSelect={onNavigateToHost} />
      </div>
    </div>
  );
}
