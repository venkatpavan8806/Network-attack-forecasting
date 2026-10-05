import { useEffect, useState } from 'react';
import { api } from '../api';
import type { RobustnessReport, RobustnessMetricStat } from '../types';
import CardHeader from './CardHeader';

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

function StatRow({ label, stat }: { label: string; stat: RobustnessMetricStat }) {
  return (
    <tr className="border-t border-[var(--color-accent-soft)]">
      <td className="py-2 pr-4 font-medium text-[var(--color-ink)]">{label}</td>
      <td className="py-2 pr-4 text-[var(--color-ink)]">{pct(stat.mean)} &plusmn; {pct(stat.std)}</td>
      <td className="py-2 pr-4 text-[var(--color-ink-dim)]">
        [{pct(stat.ci95_low)}, {pct(stat.ci95_high)}]
      </td>
      <td className="py-2 pr-4 text-[var(--color-ink-faint)] text-xs">
        {stat.values_by_seed.map((v) => pct(v)).join(', ')}
      </td>
    </tr>
  );
}

export default function RobustnessReportCard() {
  const [report, setReport] = useState<RobustnessReport | null>(null);
  const [notComputed, setNotComputed] = useState(false);

  useEffect(() => {
    api.robustnessReport().then(setReport).catch(() => setNotComputed(true));
  }, []);

  return (
    <div className="card p-6">
      <CardHeader
        title="Robustness: Variance Across Seeds & Cross-Run Generalization"
        subtitle="Is the benchmark's headline F1 trustworthy, or does it swing between runs? And does the model generalize past this run's own synthetic quirks?"
      />
      {notComputed && (
        <div className="text-sm text-[var(--color-ink-faint)] py-8 text-center">
          not yet computed -- run <code className="font-mono">python -m app.evaluate_robustness</code> (takes a few minutes)
        </div>
      )}
      {!notComputed && !report && (
        <div className="text-sm text-[var(--color-ink-faint)] py-8 text-center">loading…</div>
      )}
      {report && (
        <div className="flex flex-col gap-6">
          <div>
            <div className="text-xs text-[var(--color-ink-faint)] mb-2">
              World model, across {report.variance_across_seeds.seeds.length} independent seeds ({report.variance_across_seeds.seeds.join(', ')})
              -- each its own synthetic world, host split, and trained model
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-[var(--color-ink-faint)] text-xs uppercase tracking-wide">
                    <th className="py-1.5 pr-4 font-medium">Metric</th>
                    <th className="py-1.5 pr-4 font-medium">Mean &plusmn; std</th>
                    <th className="py-1.5 pr-4 font-medium">95% CI</th>
                    <th className="py-1.5 pr-4 font-medium">Per-seed values</th>
                  </tr>
                </thead>
                <tbody>
                  {(['precision', 'recall', 'f1', 'false_positive_rate'] as const).map((k) => (
                    <StatRow key={k} label={k.replace(/_/g, ' ')} stat={report.variance_across_seeds.world_model_lstm[k]} />
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="rounded-xl bg-[var(--color-accent-soft)] p-4">
            <div className="text-xs font-semibold uppercase tracking-wide text-[var(--color-ink-dim)] mb-2">
              Cross-run generalization (stand-in for a time-based split)
            </div>
            <div className="flex flex-wrap gap-6 text-sm">
              <div>
                <div className="text-[var(--color-ink-faint)] text-xs">Own held-out test F1</div>
                <div className="font-bold text-[var(--color-ink)]">{pct(report.cross_run_generalization.own_held_out_test_f1)}</div>
              </div>
              <div>
                <div className="text-[var(--color-ink-faint)] text-xs">Mean F1 on fresh, never-seen worlds</div>
                <div className="font-bold text-[var(--color-ink)]">{pct(report.cross_run_generalization.mean_fresh_world_f1)}</div>
              </div>
              <div>
                <div className="text-[var(--color-ink-faint)] text-xs">Generalization gap</div>
                <div className={`font-bold ${report.cross_run_generalization.generalization_gap > 0.05 ? 'text-[var(--color-bad)]' : 'text-[var(--color-ink)]'}`}>
                  {pct(report.cross_run_generalization.generalization_gap)}
                </div>
              </div>
            </div>
            <p className="text-xs text-[var(--color-ink-dim)] mt-3 leading-relaxed">{report.cross_run_generalization.note}</p>
          </div>

          <p className="text-xs text-[var(--color-ink-faint)] leading-relaxed">{report.variance_across_seeds.note}</p>
        </div>
      )}
    </div>
  );
}
