import { useEffect, useRef, useState } from 'react';
import { api, STAGE_COLORS, STAGE_LABELS, STAGE_SHORT_LABELS } from '../api';
import type { TrackResponse, TrackStep } from '../types';
import CardHeader from './CardHeader';

const END = '<END>';
const moveLabel = (m: string) => (m === END ? 'Attack ends' : STAGE_LABELS[m] ?? m);
const moveColor = (m: string) => (m === END ? '#b7b6c9' : STAGE_COLORS[m] ?? '#b7b6c9');
const pct = (v: number | null | undefined) => (v == null ? 'n/a' : `${(v * 100).toFixed(1)}%`);

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="bg-[var(--color-accent-soft)] rounded-xl p-3" title={hint}>
      <div className="text-[11px] text-[var(--color-ink-dim)]">{label}</div>
      <div className="text-lg font-bold text-[var(--color-ink)]">{value}</div>
    </div>
  );
}

function PathChips({ moves, empty }: { moves: string[]; empty: string }) {
  if (!moves.length) return <span className="text-xs text-[var(--color-ink-faint)]">{empty}</span>;
  return (
    <div className="flex flex-wrap items-center gap-1">
      {moves.map((m, i) => (
        <span key={i} className="flex items-center gap-1">
          {i > 0 && <span className="text-[var(--color-ink-faint)] text-xs">→</span>}
          <span className="px-2 py-0.5 rounded-full text-[11px] font-medium text-white" style={{ background: moveColor(m) }}>
            {moveLabel(m)}
          </span>
        </span>
      ))}
    </div>
  );
}

function StepDetail({ s }: { s: TrackStep }) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
      <div className="flex flex-col gap-3">
        <div className="text-xs text-[var(--color-ink-dim)]">
          After window <strong className="text-[var(--color-ink)]">{s.window_idx}</strong> (step {s.step}) · history used:{' '}
          <strong className="text-[var(--color-ink)]">{s.history_windows_used}</strong> window(s)
          {s.warmup && (
            <span className="ml-2 px-2 py-0.5 rounded-full bg-[var(--color-warn)]/20 text-[#b07400] text-[10px] font-semibold">WARM-UP (padded)</span>
          )}
        </div>
        <div>
          <div className="text-xs text-[var(--color-ink-faint)] mb-1">Predicted next step (window {s.window_idx + 1})</div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full" style={{ background: STAGE_COLORS[s.predicted_next_action] }} />
            <span className="font-semibold text-[var(--color-ink)]">{STAGE_LABELS[s.predicted_next_action]}</span>
            <span className="text-sm text-[var(--color-ink-dim)]">confidence {pct(s.confidence)}</span>
          </div>
          {s.actual_next_action !== undefined && s.actual_next_action !== null && (
            <div className="text-xs mt-1">
              actual:{' '}
              <span className="font-medium text-[var(--color-ink-dim)]">{STAGE_LABELS[s.actual_next_action]}</span>{' '}
              {s.correct ? (
                <span className="text-[var(--color-good)] font-semibold">✓ correct</span>
              ) : (
                <span className="text-[var(--color-bad)] font-semibold">✗ {s.correct_top3 ? 'in top-3' : 'missed'}</span>
              )}
            </div>
          )}
        </div>
        <div className="flex flex-col gap-1.5">
          {s.top_candidates.map((c) => (
            <div key={c.action} className="flex items-center gap-2 text-xs">
              <span className="w-36 truncate text-[var(--color-ink-dim)]">{STAGE_LABELS[c.action]}</span>
              <div className="flex-1 h-2 rounded-full bg-[var(--color-accent-soft)] overflow-hidden">
                <div className="h-full rounded-full" style={{ width: `${c.probability * 100}%`, background: STAGE_COLORS[c.action] }} />
              </div>
              <span className="w-12 text-right font-semibold text-[var(--color-ink)]">{pct(c.probability)}</span>
            </div>
          ))}
        </div>
        <div className="text-xs text-[var(--color-ink-dim)]">
          Infiltration probability (1 − P(benign)): <strong className="text-[var(--color-ink)]">{pct(s.infiltration_probability)}</strong>
        </div>
      </div>

      <div className="flex flex-col gap-3">
        <div>
          <div className="text-xs text-[var(--color-ink-faint)] mb-1">Attack path tracked so far</div>
          <PathChips moves={s.attack_path_so_far} empty="no attacker activity recognised yet" />
        </div>
        <div>
          <div className="text-xs text-[var(--color-ink-faint)] mb-1">
            Attacker's next 1 / 2 / 3 moves {s.next_moves && <span>({s.next_moves.method})</span>}
          </div>
          {s.next_moves ? (
            <div className="flex flex-col gap-2">
              <div className="grid grid-cols-3 gap-2">
                {s.next_moves.per_step.map((cands, k) => (
                  <div key={k} className="rounded-xl border border-[var(--color-accent-soft)] p-2">
                    <div className="text-[10px] uppercase tracking-wide text-[var(--color-ink-faint)] mb-1">Move +{k + 1}</div>
                    {cands.map((c) => (
                      <div key={c.move} className="flex items-center gap-1.5 text-[11px]">
                        <span className="w-2 h-2 rounded-full shrink-0" style={{ background: moveColor(c.move) }} />
                        <span className="flex-1 truncate text-[var(--color-ink-dim)]">{c.move === END ? 'Ends' : STAGE_SHORT_LABELS[c.move] ?? c.move}</span>
                        <span className="font-semibold text-[var(--color-ink)]">{Math.round(c.probability * 100)}%</span>
                      </div>
                    ))}
                  </div>
                ))}
              </div>
              <div className="text-xs text-[var(--color-ink-faint)]">Most likely 3-move continuation:</div>
              {s.next_moves.top_sequences.slice(0, 2).map((seq, i) => (
                <div key={i} className="flex items-center gap-2">
                  <PathChips moves={seq.moves} empty="" />
                  <span className="text-[11px] text-[var(--color-ink-dim)]">p={seq.probability.toFixed(3)}</span>
                </div>
              ))}
            </div>
          ) : (
            <span className="text-xs text-[var(--color-ink-faint)]">shown once an attacker move has been recognised</span>
          )}
        </div>
      </div>
    </div>
  );
}

export default function AttackerStepTracker({ hostId }: { hostId: string | null }) {
  const [data, setData] = useState<TrackResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [sel, setSel] = useState(0);
  const [playing, setPlaying] = useState(false);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    if (!hostId) return;
    const isLive = hostId.startsWith('live:');
    setLoading(true);
    setError(null);
    setPlaying(false);
    api.track(hostId)
      .then((d) => {
        setData(d);
        const firstAttack = d.summary.first_attack_alert_step;
        // live host: follow the newest window; recorded host: start at its first attack
        setSel(isLive ? Math.max(0, d.steps.length - 1) : firstAttack ? Math.max(0, firstAttack - 1) : 0);
      })
      .catch((e) => { setData(null); setError(e?.response?.data?.detail ?? String(e)); })
      .finally(() => setLoading(false));
    if (!isLive) return;
    // live capture adds a window every 30 s: keep the track current (no spinner)
    const t = window.setInterval(() => {
      api.track(hostId).then((d) => {
        setData(d);
        setSel(Math.max(0, d.steps.length - 1));
      }).catch(() => {});
    }, 10000);
    return () => window.clearInterval(t);
  }, [hostId]);

  useEffect(() => {
    if (!playing || !data) return;
    timer.current = window.setInterval(() => {
      setSel((i) => {
        if (i >= data.steps.length - 1) { setPlaying(false); return i; }
        return i + 1;
      });
    }, 350);
    return () => { if (timer.current) window.clearInterval(timer.current); };
  }, [playing, data]);

  const steps = data?.steps ?? [];
  const cur = steps[sel];
  const sm = data?.summary;
  const hasTruth = sm?.accuracy_top1 != null;

  return (
    <div className="card p-6">
      <CardHeader
        title="Step-by-Step Attacker Tracking"
        subtitle="A next-step prediction after EVERY window, starting from window 1 (no 8-window wait), with the attack path tracked from the start and the attacker's next 1 / 2 / 3 moves."
      />
      {loading && <div className="text-sm text-[var(--color-ink-faint)] py-8 text-center">tracking host from its first window…</div>}
      {error && !loading && <div className="text-sm text-[var(--color-bad)] py-4">{error}</div>}

      {!loading && data && sm && cur && (
        <div className="flex flex-col gap-5">
          <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-3">
            <Stat label="First prediction after" value={`window ${sm.first_prediction_after_window}`} hint={`previously window ${sm.old_first_prediction_after_window}`} />
            <Stat label="First attack alert" value={sm.first_attack_alert_step ? `step ${sm.first_attack_alert_step}` : 'none'} />
            {hasTruth && <Stat label="Top-1 accuracy" value={pct(sm.accuracy_top1)} hint="predicted next action == actual next action" />}
            {hasTruth && <Stat label="Top-3 accuracy" value={pct(sm.accuracy_top3)} />}
            {hasTruth && <Stat label={`Warm-up accuracy (steps 1–${sm.n_warmup_windows})`} value={pct(sm.accuracy_warmup_windows)} />}
            {hasTruth && <Stat label={`At phase changes (${sm.n_transitions})`} value={pct(sm.accuracy_on_transitions)} hint="windows where the next action differs from the current one" />}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <div className="text-xs text-[var(--color-ink-faint)] mb-1">Attack path recognised (whole timeline)</div>
              <PathChips moves={sm.attack_path_recognised} empty="no attack recognised" />
            </div>
            {sm.attack_path_actual && (
              <div>
                <div className="text-xs text-[var(--color-ink-faint)] mb-1">Actual attack path (ground truth)</div>
                <PathChips moves={sm.attack_path_actual} empty="benign host — no attack" />
              </div>
            )}
          </div>

          <div>
            <div className="flex items-center gap-3 mb-2">
              <button
                onClick={() => { if (sel >= steps.length - 1) setSel(0); setPlaying((p) => !p); }}
                className="px-3 py-1.5 rounded-full bg-[var(--color-accent)] text-white text-xs font-medium"
              >
                {playing ? 'Pause' : 'Replay from here'}
              </button>
              <button onClick={() => { setPlaying(false); setSel(0); }} className="px-3 py-1.5 rounded-full bg-[var(--color-accent-soft)] text-[var(--color-accent)] text-xs font-medium">
                Back to window 1
              </button>
              <input type="range" min={0} max={steps.length - 1} value={sel} onChange={(e) => { setPlaying(false); setSel(Number(e.target.value)); }} className="flex-1" />
              <span className="text-xs text-[var(--color-ink-dim)] w-24 text-right">step {cur.step} / {steps.length}</span>
            </div>
            <div className="overflow-x-auto">
              <div className="flex flex-col gap-0.5 min-w-max">
                {(['predicted', ...(hasTruth ? ['actual'] : [])] as const).map((row) => (
                  <div key={row} className="flex items-center gap-[2px]">
                    <span className="w-16 text-[10px] text-[var(--color-ink-faint)] shrink-0">{row === 'predicted' ? 'predicted' : 'actual'}</span>
                    {steps.map((s, i) => {
                      const a = row === 'predicted' ? s.predicted_next_action : s.actual_next_action;
                      return (
                        <button
                          key={i}
                          onClick={() => { setPlaying(false); setSel(i); }}
                          title={`step ${s.step}: ${a ? STAGE_LABELS[a] : '—'}`}
                          className={`w-2 h-5 rounded-[2px] ${i === sel ? 'ring-2 ring-[var(--color-ink)]' : ''} ${i > sel ? 'opacity-25' : ''}`}
                          style={{ background: a ? STAGE_COLORS[a] : '#eceafa', outline: s.warmup && row === 'predicted' ? '1px dashed #b07400' : undefined }}
                        />
                      );
                    })}
                  </div>
                ))}
              </div>
            </div>
            <div className="text-[10px] text-[var(--color-ink-faint)] mt-1">Each cell = the prediction made after that window for the next one. Dashed cells = warm-up (fewer than 8 real windows, padded).</div>
          </div>

          <div className="pt-4 border-t border-[var(--color-accent-soft)]">
            <StepDetail s={cur} />
          </div>

          <details className="pt-2">
            <summary className="text-xs text-[var(--color-accent)] cursor-pointer">Full step-by-step log ({steps.length} windows)</summary>
            <div className="max-h-80 overflow-y-auto mt-2">
              <table className="w-full text-xs">
                <thead className="sticky top-0 bg-white">
                  <tr className="text-left text-[var(--color-ink-faint)] uppercase tracking-wide">
                    <th className="py-1.5 pr-2">Step</th>
                    <th className="py-1.5 pr-2">Window</th>
                    <th className="py-1.5 pr-2">History</th>
                    {hasTruth && <th className="py-1.5 pr-2">Current (actual)</th>}
                    <th className="py-1.5 pr-2">Predicted next</th>
                    <th className="py-1.5 pr-2">Confidence</th>
                    {hasTruth && <th className="py-1.5 pr-2">Actual next</th>}
                    <th className="py-1.5 pr-2">Next 3 moves</th>
                  </tr>
                </thead>
                <tbody>
                  {steps.map((s, i) => (
                    <tr key={i} onClick={() => setSel(i)} className={`border-t border-[var(--color-accent-soft)] cursor-pointer ${i === sel ? 'bg-[var(--color-accent-soft)]' : ''}`}>
                      <td className="py-1 pr-2">{s.step}</td>
                      <td className="py-1 pr-2">{s.window_idx}</td>
                      <td className="py-1 pr-2">{s.history_windows_used}{s.warmup && <span className="text-[#b07400]"> (warm-up)</span>}</td>
                      {hasTruth && <td className="py-1 pr-2 text-[var(--color-ink-dim)]">{s.actual_current_action ? STAGE_SHORT_LABELS[s.actual_current_action] : ''}</td>}
                      <td className="py-1 pr-2 font-medium">{STAGE_SHORT_LABELS[s.predicted_next_action]}</td>
                      <td className="py-1 pr-2">{pct(s.confidence)}</td>
                      {hasTruth && (
                        <td className="py-1 pr-2">
                          {s.actual_next_action ? STAGE_SHORT_LABELS[s.actual_next_action] : '—'}{' '}
                          {s.correct === true && <span className="text-[var(--color-good)]">✓</span>}
                          {s.correct === false && <span className="text-[var(--color-bad)]">✗</span>}
                        </td>
                      )}
                      <td className="py-1 pr-2 text-[var(--color-ink-dim)]">
                        {s.next_moves?.top_sequences[0]?.moves.map((m) => (m === END ? 'End' : STAGE_SHORT_LABELS[m])).join(' → ') ?? '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>

          <p className="text-[11px] text-[var(--color-ink-faint)]">Decision rule: {sm.decision_rule}.</p>
        </div>
      )}
    </div>
  );
}
