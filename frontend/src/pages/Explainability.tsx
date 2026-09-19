import { useEffect, useState } from 'react';
import { api, STAGE_COLORS } from '../api';
import type { ForecastResponse, FalseAlarmExample } from '../types';
import CardHeader from '../components/CardHeader';

export default function Explainability({ selectedHost }: { selectedHost: string | null }) {
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [falseAlarms, setFalseAlarms] = useState<FalseAlarmExample[]>([]);

  useEffect(() => {
    if (!selectedHost) return;
    api.forecast(selectedHost).then(setForecast).catch(() => {});
  }, [selectedHost]);

  useEffect(() => {
    api.falseAlarms().then(setFalseAlarms).catch(() => {});
  }, []);

  const maxAttn = forecast ? Math.max(...forecast.explanation.attention_over_past_windows, 1e-6) : 1;

  return (
    <div className="flex flex-col gap-6">
      <div className="card p-6">
        <CardHeader
          title="Attention Over Past Windows"
          subtitle={forecast ? `${forecast.host_id} — window ${forecast.window_idx}: which of the last 8 real windows the model leaned on for this forecast` : 'select a host'}
        />
        {forecast ? (
          <div className="flex items-end gap-2 h-40">
            {forecast.explanation.attention_over_past_windows.map((w, i) => (
              <div key={i} className="flex-1 flex flex-col items-center gap-1.5">
                <div className="text-[10px] text-[var(--color-ink-faint)]">{(w * 100).toFixed(0)}%</div>
                <div
                  className="w-full rounded-t-lg bg-[var(--color-accent)]"
                  style={{ height: `${Math.max(4, (w / maxAttn) * 100)}%`, opacity: 0.4 + 0.6 * (w / maxAttn) }}
                />
                <div className="text-[10px] text-[var(--color-ink-faint)]">t-{7 - i}</div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-sm text-[var(--color-ink-faint)] py-8 text-center">not yet available — pick a host in Forecasts</div>
        )}
      </div>

      <div className="card p-6">
        <CardHeader title="Top Contributing Features" subtitle="Attention-weighted input-gradient saliency, computed on the actual trained model and the actual input" />
        {forecast && forecast.explanation.top_contributors.length ? (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-[var(--color-ink-faint)] text-xs uppercase tracking-wide">
                  <th className="py-2 pr-4 font-medium">Feature</th>
                  <th className="py-2 pr-4 font-medium">Window offset</th>
                  <th className="py-2 pr-4 font-medium">Attention weight</th>
                  <th className="py-2 pr-4 font-medium">Contribution</th>
                </tr>
              </thead>
              <tbody>
                {forecast.explanation.top_contributors.map((c, i) => (
                  <tr key={i} className="border-t border-[var(--color-accent-soft)]">
                    <td className="py-2.5 pr-4 font-medium text-[var(--color-ink)]">{c.feature}</td>
                    <td className="py-2.5 pr-4 text-[var(--color-ink-dim)]">{c.timestep_offset === 0 ? 'most recent' : `${c.timestep_offset}`}</td>
                    <td className="py-2.5 pr-4 text-[var(--color-ink-dim)]">{(c.attention_weight * 100).toFixed(1)}%</td>
                    <td className={`py-2.5 pr-4 font-semibold ${c.contribution_score >= 0 ? 'text-[var(--color-bad)]' : 'text-[var(--color-good)]'}`}>
                      {c.contribution_score >= 0 ? '+' : ''}{c.contribution_score.toFixed(5)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-sm text-[var(--color-ink-faint)] py-8 text-center">not yet available</div>
        )}
      </div>

      <div className="card p-6">
        <CardHeader
          title="False-Alarm Honesty"
          subtitle="Real examples from held-out benign hosts where the model raised rising concern that never became an attack"
        />
        {falseAlarms.length ? (
          <div className="flex flex-col divide-y divide-[var(--color-accent-soft)]">
            {falseAlarms.map((f, i) => (
              <div key={i} className="flex items-center justify-between py-3 text-sm">
                <div>
                  <div className="font-medium text-[var(--color-ink)]">{f.host_id} &middot; window #{f.window_idx}</div>
                  <div className="text-xs text-[var(--color-ink-dim)]">{f.outcome}</div>
                </div>
                <span className="px-2.5 py-1 rounded-full text-xs font-semibold" style={{ background: `${STAGE_COLORS.ambiguous_pre_attack}22`, color: STAGE_COLORS.ambiguous_pre_attack }}>
                  {(f.infiltration_probability * 100).toFixed(1)}% flagged
                </span>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-sm text-[var(--color-ink-faint)] py-8 text-center">no false alarms found on held-out benign hosts at the current watch threshold</div>
        )}
      </div>
    </div>
  );
}
