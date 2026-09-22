import { useEffect, useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
  Scatter, Line, ComposedChart,
} from 'recharts';
import { api } from '../api';
import type { BenchmarkReport, CalibrationReport, LeadTimeReport } from '../types';
import CardHeader from '../components/CardHeader';

function metricRows(report: BenchmarkReport) {
  const keys: (keyof BenchmarkReport['world_model_lstm'])[] = ['precision', 'recall', 'f1', 'false_positive_rate'];
  const labels: Record<string, string> = { precision: 'Precision', recall: 'Recall', f1: 'F1', false_positive_rate: 'False Positive Rate' };
  return keys.map((k) => ({
    metric: labels[k],
    Baseline: report.baseline_logistic_regression[k] as number,
    'World Model': report.world_model_lstm[k] as number,
  }));
}

export default function Benchmarks() {
  const [benchmark, setBenchmark] = useState<BenchmarkReport | null>(null);
  const [calibration, setCalibration] = useState<CalibrationReport | null>(null);
  const [leadTime, setLeadTime] = useState<LeadTimeReport | null>(null);

  useEffect(() => {
    api.benchmark().then(setBenchmark).catch(() => {});
    api.calibration().then(setCalibration).catch(() => {});
    api.leadTime().then(setLeadTime).catch(() => {});
  }, []);

  const calibData = calibration?.bins.filter((b) => b.count > 0).map((b) => ({
    predicted: b.mean_predicted,
    observed: b.observed_frequency,
    perfect: b.mean_predicted,
    count: b.count,
  })) ?? [];

  return (
    <div className="flex flex-col gap-6">
      <div className="card p-6">
        <CardHeader title="Benchmark: World Model vs. Logistic Regression Baseline" subtitle={benchmark?.note ?? 'computed on a held-out, by-host test split'} />
        {benchmark ? (
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={metricRows(benchmark)} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                <CartesianGrid stroke="#eceafa" vertical={false} />
                <XAxis dataKey="metric" tick={{ fontSize: 12, fill: '#8b8a9e' }} axisLine={false} tickLine={false} />
                <YAxis domain={[0, 1]} tick={{ fontSize: 11, fill: '#8b8a9e' }} axisLine={false} tickLine={false} />
                <Tooltip formatter={(v) => Number(v).toFixed(4)} />
                <Legend />
                <Bar dataKey="Baseline" fill="#ff6b81" radius={[6, 6, 0, 0]} />
                <Bar dataKey="World Model" fill="#6c5dd3" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="h-80 flex items-center justify-center text-sm text-[var(--color-ink-faint)]">not yet available — run `python -m app.train`</div>
        )}
        {benchmark && (
          <div className="grid grid-cols-2 gap-4 mt-4 text-xs text-[var(--color-ink-dim)]">
            <div>Baseline: TP {benchmark.baseline_logistic_regression.tp} · FP {benchmark.baseline_logistic_regression.fp} · FN {benchmark.baseline_logistic_regression.fn} · TN {benchmark.baseline_logistic_regression.tn} (n={benchmark.baseline_logistic_regression.n_samples})</div>
            <div>World Model: TP {benchmark.world_model_lstm.tp} · FP {benchmark.world_model_lstm.fp} · FN {benchmark.world_model_lstm.fn} · TN {benchmark.world_model_lstm.tn} (n={benchmark.world_model_lstm.n_samples})</div>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <div className="card p-6">
          <CardHeader
            title={`Calibration @ horizon t+${calibration?.horizon_windows ?? '?'}`}
            subtitle={calibration ? `Brier score ${calibration.brier_score.toFixed(4)} · ${calibration.n_points} held-out rollout points` : undefined}
          />
          {calibData.length ? (
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={calibData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                  <CartesianGrid stroke="#eceafa" />
                  <XAxis dataKey="predicted" type="number" domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} tick={{ fontSize: 11, fill: '#8b8a9e' }} axisLine={false} tickLine={false} name="Predicted" />
                  <YAxis dataKey="observed" type="number" domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} tick={{ fontSize: 11, fill: '#8b8a9e' }} axisLine={false} tickLine={false} name="Observed" />
                  <Tooltip formatter={(v) => `${(Number(v) * 100).toFixed(1)}%`} />
                  <Line type="monotone" dataKey="perfect" stroke="#d8d4f0" strokeDasharray="4 4" dot={false} name="Perfect calibration" />
                  <Scatter dataKey="observed" fill="#6c5dd3" name="Observed frequency" />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="h-72 flex items-center justify-center text-sm text-[var(--color-ink-faint)]">not yet available</div>
          )}
          <p className="text-xs text-[var(--color-ink-faint)] mt-2">
            Points near the dashed diagonal mean predicted probability tracks observed attack frequency — evidence of real forecasting, not a relabeled classifier.
          </p>
        </div>

        <div className="card p-6">
          <CardHeader title="Lead-Time vs. Baseline" subtitle={leadTime ? leadTime.note : undefined} />
          {leadTime ? (
            <div className="flex flex-col gap-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-[var(--color-accent-soft)] rounded-xl p-4">
                  <div className="text-xs text-[var(--color-ink-dim)]">Median (all attack hosts)</div>
                  <div className="text-2xl font-bold text-[var(--color-ink)]">
                    {leadTime.median_lead_time_minutes_all_hosts != null ? `${leadTime.median_lead_time_minutes_all_hosts.toFixed(2)} min` : 'n/a'}
                  </div>
                </div>
                <div className="bg-[var(--color-accent-soft)] rounded-xl p-4">
                  <div className="text-xs text-[var(--color-ink-dim)]">Median (held-out only)</div>
                  <div className="text-2xl font-bold text-[var(--color-ink)]">
                    {leadTime.median_lead_time_minutes_heldout_only != null ? `${leadTime.median_lead_time_minutes_heldout_only.toFixed(2)} min` : 'n/a'}
                  </div>
                </div>
              </div>
              <table className="w-full text-sm mt-2">
                <thead>
                  <tr className="text-left text-[var(--color-ink-faint)] text-xs uppercase tracking-wide">
                    <th className="py-2 pr-3 font-medium">Host</th>
                    <th className="py-2 pr-3 font-medium">Split</th>
                    <th className="py-2 pr-3 font-medium">Baseline fires</th>
                    <th className="py-2 pr-3 font-medium">World model alerts</th>
                    <th className="py-2 pr-3 font-medium">Lead time</th>
                  </tr>
                </thead>
                <tbody>
                  {leadTime.per_host.map((h) => (
                    <tr key={h.host_id} className="border-t border-[var(--color-accent-soft)]">
                      <td className="py-2 pr-3 font-medium text-[var(--color-ink)]">{h.host_id}</td>
                      <td className="py-2 pr-3 text-[var(--color-ink-dim)] capitalize">{h.split}</td>
                      <td className="py-2 pr-3 text-[var(--color-ink-dim)]">{h.baseline_fire_window ?? '—'}</td>
                      <td className="py-2 pr-3 text-[var(--color-ink-dim)]">{h.worldmodel_alert_window ?? '—'}</td>
                      <td className="py-2 pr-3 font-semibold text-[var(--color-ink)]">
                        {h.lead_time_minutes != null ? `${h.lead_time_minutes.toFixed(2)} min` : 'not flagged by both'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="h-40 flex items-center justify-center text-sm text-[var(--color-ink-faint)]">not yet available</div>
          )}
        </div>
      </div>
    </div>
  );
}
