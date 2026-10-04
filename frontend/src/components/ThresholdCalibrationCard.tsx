import { useEffect, useState } from 'react';
import { api } from '../api';
import type { ThresholdCalibrationReport } from '../types';
import CardHeader from './CardHeader';

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

export default function ThresholdCalibrationCard() {
  const [report, setReport] = useState<ThresholdCalibrationReport | null>(null);
  const [notComputed, setNotComputed] = useState(false);

  useEffect(() => {
    api.thresholdCalibration().then(setReport).catch(() => setNotComputed(true));
  }, []);

  return (
    <div className="card p-6">
      <CardHeader
        title="Alert Threshold Calibration"
        subtitle="What threshold a real alert-per-day budget would actually require, computed from the model's real score distribution on your benign traffic -- not the fixed 0.5 used elsewhere."
      />
      {notComputed && (
        <div className="text-sm text-[var(--color-ink-faint)] py-8 text-center">
          No labelled traffic yet — upload a CSV containing fully benign hosts with a true_stage column (Forecasts → CSV Ingestion) to see this computed on your data.
        </div>
      )}
      {!notComputed && !report && (
        <div className="text-sm text-[var(--color-ink-faint)] py-8 text-center">loading…</div>
      )}
      {report && (
        <div className="flex flex-col gap-5">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="bg-[var(--color-accent-soft)] rounded-xl p-4">
              <div className="text-[11px] text-[var(--color-ink-dim)] uppercase tracking-wide mb-1">
                Current fixed threshold (0.5), on your benign traffic
              </div>
              <div className="text-lg font-bold text-[var(--color-ink)]">
                {pct(report.current_fixed_threshold.false_positive_rate_on_held_out_benign)} false-positive rate
              </div>
              <div className="text-xs text-[var(--color-ink-dim)] mt-1">
                &asymp; {report.current_fixed_threshold.implied_alerts_per_day.toFixed(1)} false alerts/day across{' '}
                {report.n_hosts_monitored_assumption} hosts
              </div>
            </div>
            <div className="bg-[var(--color-card)] border border-[var(--color-accent-soft)] rounded-xl p-4">
              <div className="text-[11px] text-[var(--color-ink-dim)] uppercase tracking-wide mb-1">Sample size</div>
              <div className="text-lg font-bold text-[var(--color-ink)]">{report.n_benign_windows.toLocaleString()}</div>
              <div className="text-xs text-[var(--color-ink-dim)] mt-1">benign windows from your uploads, {report.window_seconds}s each</div>
            </div>
          </div>

          <div>
            <div className="text-xs text-[var(--color-ink-faint)] mb-2">
              Threshold required for a given alerts/day budget, at {report.n_hosts_monitored_assumption} hosts monitored
            </div>
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-[var(--color-ink-faint)] text-xs uppercase tracking-wide">
                  <th className="py-1.5 pr-4 font-medium">Target alerts/day</th>
                  <th className="py-1.5 pr-4 font-medium">Required threshold</th>
                </tr>
              </thead>
              <tbody>
                {report.budgets.map((b) => (
                  <tr key={b.target_alerts_per_day} className="border-t border-[var(--color-accent-soft)]">
                    <td className="py-2 pr-4 text-[var(--color-ink)]">{b.target_alerts_per_day}</td>
                    <td className="py-2 pr-4 font-semibold text-[var(--color-ink)]">{b.required_threshold.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <p className="text-xs text-[var(--color-ink-faint)] leading-relaxed">{report.note}</p>
        </div>
      )}
    </div>
  );
}
