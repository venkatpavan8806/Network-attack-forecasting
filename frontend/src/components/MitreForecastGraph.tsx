import { STAGE_COLORS, STAGE_SHORT_LABELS, STAGE_LABELS } from '../api';
import type { AttackMapping, BranchingHorizon } from '../types';

interface Props {
  stageProbabilitiesNow: Record<string, number>;
  branchingForecast: BranchingHorizon[];
  attackMapping: AttackMapping[];
  topK?: number;
}

// real MITRE tactics, in kill-chain order -- this is the vertical axis
const ROW_ORDER = [
  'N/A',
  'Reconnaissance (Pre-Confirmation)',
  'Reconnaissance',
  'Credential Access / Initial Access',
  'Lateral Movement',
  'Command and Control',
  'Exfiltration',
];
const ROW_LABELS: Record<string, string> = {
  'N/A': 'Benign',
  'Reconnaissance (Pre-Confirmation)': 'Pre-Attack',
  'Reconnaissance': 'Reconnaissance',
  'Credential Access / Initial Access': 'Credential Access',
  'Lateral Movement': 'Lateral Movement',
  'Command and Control': 'C2',
  'Exfiltration': 'Exfiltration',
};

const ROW_H = 42;
const COL_W = 108;
const COL_START = 190;
const TOP_PAD = 24;
const NODE_R_MAIN = 15;
const NODE_R_GHOST = 8;

interface Candidate { action: string; probability: number; tactic: string; technique_id: string | null; technique_name: string }

export default function MitreForecastGraph({ stageProbabilitiesNow, branchingForecast, attackMapping, topK = 3 }: Props) {
  const tacticOf: Record<string, string> = {};
  for (const m of attackMapping) tacticOf[m.state_label] = m.tactic;

  const nowCandidates: Candidate[] = Object.entries(stageProbabilitiesNow)
    .sort((a, b) => b[1] - a[1])
    .slice(0, topK)
    .map(([action, probability]) => {
      const m = attackMapping.find((x) => x.state_label === action);
      return { action, probability, tactic: tacticOf[action] ?? 'N/A', technique_id: m?.technique_id ?? null, technique_name: m?.technique_name ?? '' };
    });

  const columns: Candidate[][] = [nowCandidates, ...branchingForecast.map((h) => h.candidates.slice(0, topK))];
  const colLabels = ['now', ...branchingForecast.map((h) => `t+${h.horizon}`)];

  const rowIndex = (tactic: string) => {
    const idx = ROW_ORDER.indexOf(tactic);
    return idx === -1 ? 0 : idx;
  };
  const yFor = (tactic: string) => TOP_PAD + rowIndex(tactic) * ROW_H + ROW_H / 2;
  const xFor = (col: number) => COL_START + col * COL_W;

  const viewW = COL_START + (columns.length - 1) * COL_W + 90;
  const viewH = TOP_PAD + ROW_ORDER.length * ROW_H + 30;

  // the "real" rollout path: the top candidate (index 0) of each column, connected in sequence
  const mainPath = columns.map((c) => c[0]).filter(Boolean);

  return (
    <div className="w-full overflow-x-auto">
      <svg viewBox={`0 0 ${viewW} ${viewH}`} width="100%" style={{ minWidth: Math.min(viewW, 760) }}>
        {/* row bands + labels */}
        {ROW_ORDER.map((tactic, i) => (
          <g key={tactic}>
            <rect x={0} y={TOP_PAD + i * ROW_H} width={viewW} height={ROW_H} fill={i % 2 === 0 ? '#faf9ff' : 'transparent'} />
            <text x={8} y={TOP_PAD + i * ROW_H + ROW_H / 2 + 4} fontSize="10.5" fill="var(--color-ink-dim)" fontWeight={600}>
              {ROW_LABELS[tactic]}
            </text>
          </g>
        ))}

        {/* column labels */}
        {colLabels.map((label, i) => (
          <text key={label} x={xFor(i)} y={viewH - 8} fontSize="10" fill="var(--color-ink-faint)" textAnchor="middle">
            {label}
          </text>
        ))}

        {/* the real rollout path, connecting the top candidate at each step */}
        {mainPath.slice(1).map((c, i) => {
          const prev = mainPath[i];
          return (
            <line
              key={`path-${i}`}
              x1={xFor(i)} y1={yFor(prev.tactic)} x2={xFor(i + 1)} y2={yFor(c.tactic)}
              stroke="var(--color-accent)" strokeWidth={2.5} markerEnd="url(#mitre-arrow)"
            />
          );
        })}
        <defs>
          <marker id="mitre-arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill="var(--color-accent)" />
          </marker>
        </defs>

        {/* ghost candidates (2nd, 3rd most likely) -- NOT connected across columns,
            since the model only computes a real transition along the top path */}
        {columns.map((col, ci) =>
          col.slice(1).map((c, gi) => (
            <g key={`ghost-${ci}-${gi}`} opacity={0.35 + c.probability * 0.5}>
              <circle cx={xFor(ci)} cy={yFor(c.tactic)} r={NODE_R_GHOST} fill={STAGE_COLORS[c.action] ?? '#999'} stroke="white" strokeWidth={1}>
                <title>{STAGE_LABELS[c.action] ?? c.action} — {(c.probability * 100).toFixed(0)}%{c.technique_id ? ` (${c.technique_id})` : ''}</title>
              </circle>
            </g>
          )),
        )}

        {/* main path nodes (top candidate per column) */}
        {mainPath.map((c, i) => (
          <g key={`main-${i}`}>
            <circle cx={xFor(i)} cy={yFor(c.tactic)} r={NODE_R_MAIN} fill={STAGE_COLORS[c.action] ?? '#999'} stroke="white" strokeWidth={2.5}>
              <title>{STAGE_LABELS[c.action] ?? c.action} — {(c.probability * 100).toFixed(0)}%{c.technique_id ? ` (${c.technique_id})` : ''}</title>
            </circle>
            <text x={xFor(i)} y={yFor(c.tactic) - NODE_R_MAIN - 6} fontSize="9" fill="var(--color-ink)" fontWeight={600} textAnchor="middle">
              {STAGE_SHORT_LABELS[c.action] ?? c.action}
            </text>
            <text x={xFor(i)} y={yFor(c.tactic) + NODE_R_MAIN + 13} fontSize="8.5" fill="var(--color-ink-faint)" textAnchor="middle">
              {(c.probability * 100).toFixed(0)}%
            </text>
          </g>
        ))}
      </svg>
      <p className="text-[10px] text-[var(--color-ink-faint)] mt-2">
        Large connected nodes = the model's actual rollout path (its top prediction at each step). Small faded nodes = other candidate actions it also considered at that step, sized by probability — not connected across time, since the model only computes a real transition along the top path.
      </p>
    </div>
  );
}
