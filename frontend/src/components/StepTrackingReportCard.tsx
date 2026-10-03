import { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { api } from '../api';
import type { MethodScores, StepTrackingReport } from '../types';
import CardHeader from './CardHeader';

const pct = (v: number | null | undefined) => (v == null ? 'n/a' : `${(v * 100).toFixed(1)}%`);

const METHOD_LABELS: Record<string, string> = {
  unigram_most_frequent: 'Most-frequent move',
  markov_order1: 'Markov chain (order 1)',
  trigram_order2: 'N-gram (trigram)',
  lstm_only: 'LSTM only',
  hybrid_trigram_plus_lstm: 'Trigram + LSTM hybrid',
};
const METHOD_COLORS: Record<string, string> = {
  markov_order1: '#ffb84d',
  trigram_order2: '#3dd598',
  lstm_only: '#ff6b81',
  hybrid_trigram_plus_lstm: '#6c5dd3',
};

function ScoreTable({ rows }: { rows: [string, MethodScores][] }) {
  return (
    <table className="w-full text-xs">
      <thead>
        <tr className="text-left text-[var(--color-ink-faint)] uppercase tracking-wide">
          <th className="py-1.5 pr-2">Method</th>
          <th className="py-1.5 pr-2">Next 1 (top-3)</th>
          <th className="py-1.5 pr-2">Next 2</th>
          <th className="py-1.5 pr-2">Next 3</th>
          <th className="py-1.5 pr-2">Next 3, tactic level</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(([name, s]) => (
          <tr key={name} className="border-t border-[var(--color-accent-soft)]">
            <td className="py-1.5 pr-2 font-medium text-[var(--color-ink)]">{METHOD_LABELS[name] ?? name}</td>
            <td className="py-1.5 pr-2">{pct(s.next_1?.exact_match)} ({pct(s.next_1?.top3_accuracy)})</td>
            <td className="py-1.5 pr-2">{s.next_2 ? pct(s.next_2.exact_match) : 'n/a'}</td>
            <td className="py-1.5 pr-2">{s.next_3 ? pct(s.next_3.exact_match) : 'n/a'}</td>
            <td className="py-1.5 pr-2">{s.next_3 ? pct(s.next_3.tactic_level_match) : 'n/a'}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function StepTrackingReportCard() {
  const [r, setR] = useState<StepTrackingReport | null>(null);
  useEffect(() => { api.stepTrackingReport().then(setR).catch(() => {}); }, []);

  if (!r) {
    return (
      <div className="card p-6">
        <CardHeader title="Step-by-Step Tracking & Next 1/2/3-Move Prediction" />
        <div className="h-32 flex items-center justify-center text-sm text-[var(--color-ink-faint)]">
          not yet available — run `python -m app.evaluate_step_tracking`
        </div>
      </div>
    );
  }

  const w = r.window_level_tracking;
  const warmKey = Object.keys(w).find((k) => k.startsWith('accuracy_warmup_windows'));
  const h2h = r.multi_step_moves.heldout_hosts_vs_lstm;
  const h2hRows = (['markov_order1', 'trigram_order2', 'lstm_only', 'hybrid_trigram_plus_lstm'] as const)
    .filter((k) => h2h[k]).map((k) => [k, h2h[k] as MethodScores] as [string, MethodScores]);
  const lohoRows = Object.entries(r.multi_step_moves.leave_one_host_out_all_attack_hosts);
  const chart = (['next_1', 'next_2', 'next_3'] as const).map((k) => {
    const row: Record<string, any> = { horizon: k.replace('next_', 'Next ') + (k === 'next_1' ? ' move' : ' moves') };
    for (const [name, s] of h2hRows) if (s[k]) row[name] = s[k]!.exact_match;
    return row;
  });

  return (
    <div className="card p-6 flex flex-col gap-6">
      <CardHeader
        title="Step-by-Step Tracking & Next 1/2/3-Move Prediction"
        subtitle="How each step-by-step prediction is decided and measured, and how other algorithms compare with the LSTM for predicting 2 or 3 moves ahead."
      />

      <div>
        <h4 className="text-sm font-semibold text-[var(--color-ink)] mb-2">Which metric decides a prediction</h4>
        <ul className="text-xs text-[var(--color-ink-dim)] flex flex-col gap-1 list-disc pl-5">
          {Object.entries(r.decision_metric).map(([k, v]) => (
            <li key={k}><strong className="text-[var(--color-ink)]">{k.replace(/_/g, ' ')}:</strong> {v}</li>
          ))}
        </ul>
      </div>

      <div>
        <h4 className="text-sm font-semibold text-[var(--color-ink)] mb-2">Prediction after every window (held-out hosts, {w.n_predictions} predictions)</h4>
        <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-3 text-center">
          {[
            ['Top-1 accuracy', w.accuracy_top1],
            ['Top-3 accuracy', w.accuracy_top3],
            ['Macro-F1 (11 classes)', w.macro_f1],
            [`Warm-up windows 1–7`, warmKey ? w[warmKey] : null],
            ['Full-history windows', w.accuracy_full_history_windows],
            [`At phase changes (top-3)`, w.top3_on_transitions],
          ].map(([label, v]) => (
            <div key={label as string} className="bg-[var(--color-accent-soft)] rounded-xl p-3">
              <div className="text-[11px] text-[var(--color-ink-dim)]">{label}</div>
              <div className="text-lg font-bold text-[var(--color-ink)]">{pct(v as number)}</div>
            </div>
          ))}
        </div>
        <p className="text-[11px] text-[var(--color-ink-faint)] mt-2">
          Cold start (attacker first seen at window 1): warm-up accuracy {pct(r.cold_start.accuracy_warmup_overall)}; previously no prediction until window {r.cold_start.old_pipeline_first_prediction_after_window}.
          Attack-path tracking: kill-chain order correct on {pct(r.path_recognition.tactic_level_exact_match)} of {r.path_recognition.n_attack_hosts} attack hosts;
          exact service variant (SSH/RDP/SMB) on {pct(r.path_recognition.technique_level_exact_match)}.
        </p>
      </div>

      <div>
        <h4 className="text-sm font-semibold text-[var(--color-ink)] mb-1">Predicting the next 1, 2 and 3 moves: algorithms compared</h4>
        <p className="text-[11px] text-[var(--color-ink-faint)] mb-3">{r.multi_step_moves.unit}</p>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chart} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
              <CartesianGrid stroke="#eceafa" vertical={false} />
              <XAxis dataKey="horizon" tick={{ fontSize: 12, fill: '#8b8a9e' }} axisLine={false} tickLine={false} />
              <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} tick={{ fontSize: 11, fill: '#8b8a9e' }} axisLine={false} tickLine={false} />
              <Tooltip formatter={(v) => pct(Number(v))} />
              <Legend />
              {Object.keys(METHOD_COLORS).map((m) => (
                <Bar key={m} dataKey={m} name={METHOD_LABELS[m]} fill={METHOD_COLORS[m]} radius={[6, 6, 0, 0]} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 mt-4">
          <div>
            <div className="text-xs font-semibold text-[var(--color-ink)] mb-1">Held-out attack hosts ({h2h.hosts.join(', ')})</div>
            <ScoreTable rows={h2hRows} />
            <p className="text-[11px] text-[var(--color-ink-faint)] mt-1">{h2h.note}</p>
          </div>
          <div>
            <div className="text-xs font-semibold text-[var(--color-ink)] mb-1">Leave-one-attack-host-out (all attack hosts, sequence models only)</div>
            <ScoreTable rows={lohoRows} />
          </div>
        </div>
      </div>
    </div>
  );
}
