import { STAGE_COLORS, STAGE_SHORT_LABELS, STAGE_LABELS } from '../api';

interface Props {
  withoutActions: string[];
  withActions: string[];
}

const COL_WIDTH = 100;
const COL_START = 110;
const ROW_WITHOUT_Y = 55;
const ROW_WITH_Y = 155;
const NODE_R = 18;
const VIEW_H = 210;

export default function TrajectoryPathDiagram({ withoutActions, withActions }: Props) {
  const n = Math.max(withoutActions.length, withActions.length);
  const viewW = COL_START + (n - 1) * COL_WIDTH + 110;
  const xAt = (i: number) => COL_START + i * COL_WIDTH;

  return (
    <div className="w-full overflow-x-auto">
      <svg viewBox={`0 0 ${viewW} ${VIEW_H}`} width="100%" style={{ minWidth: Math.min(viewW, 640) }}>
        {/* row labels */}
        <text x={8} y={ROW_WITHOUT_Y + 4} fontSize="11" fill="var(--color-bad)" fontWeight={600}>Without</text>
        <text x={8} y={ROW_WITH_Y + 4} fontSize="11" fill="var(--color-good)" fontWeight={600}>With</text>

        {/* timestep labels */}
        {Array.from({ length: n }).map((_, i) => (
          <text key={`t${i}`} x={xAt(i)} y={VIEW_H - 8} fontSize="10" fill="var(--color-ink-faint)" textAnchor="middle">
            t+{i + 1}
          </text>
        ))}

        {/* within-lane arrows: without */}
        {withoutActions.slice(1).map((_, i) => (
          <line
            key={`wo-arrow-${i}`}
            x1={xAt(i) + NODE_R} y1={ROW_WITHOUT_Y} x2={xAt(i + 1) - NODE_R} y2={ROW_WITHOUT_Y}
            stroke="var(--color-bad)" strokeWidth={2} markerEnd="url(#arrow-without)"
          />
        ))}
        {/* within-lane arrows: with */}
        {withActions.slice(1).map((_, i) => (
          <line
            key={`w-arrow-${i}`}
            x1={xAt(i) + NODE_R} y1={ROW_WITH_Y} x2={xAt(i + 1) - NODE_R} y2={ROW_WITH_Y}
            stroke="var(--color-good)" strokeWidth={2} markerEnd="url(#arrow-with)"
          />
        ))}

        {/* vertical connectors between lanes -- highlighted where actions diverge */}
        {Array.from({ length: n }).map((_, i) => {
          const diverged = withoutActions[i] !== withActions[i];
          return (
            <line
              key={`conn-${i}`}
              x1={xAt(i)} y1={ROW_WITHOUT_Y + NODE_R} x2={xAt(i)} y2={ROW_WITH_Y - NODE_R}
              stroke={diverged ? 'var(--color-accent)' : '#e3e0f5'}
              strokeWidth={diverged ? 2.5 : 1.5}
              strokeDasharray={diverged ? '0' : '3 3'}
            />
          );
        })}

        <defs>
          <marker id="arrow-without" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill="var(--color-bad)" />
          </marker>
          <marker id="arrow-with" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill="var(--color-good)" />
          </marker>
        </defs>

        {/* nodes: without */}
        {withoutActions.map((a, i) => (
          <g key={`wo-node-${i}`}>
            <circle cx={xAt(i)} cy={ROW_WITHOUT_Y} r={NODE_R} fill={STAGE_COLORS[a] ?? '#999'} stroke="white" strokeWidth={2}>
              <title>{STAGE_LABELS[a] ?? a}</title>
            </circle>
            <text x={xAt(i)} y={ROW_WITHOUT_Y - NODE_R - 6} fontSize="9" fill="var(--color-ink-dim)" textAnchor="middle">
              {STAGE_SHORT_LABELS[a] ?? a}
            </text>
          </g>
        ))}

        {/* nodes: with */}
        {withActions.map((a, i) => {
          const diverged = withoutActions[i] !== a;
          return (
            <g key={`w-node-${i}`}>
              <circle cx={xAt(i)} cy={ROW_WITH_Y} r={NODE_R} fill={STAGE_COLORS[a] ?? '#999'} stroke={diverged ? 'var(--color-accent)' : 'white'} strokeWidth={diverged ? 3 : 2}>
                <title>{STAGE_LABELS[a] ?? a}</title>
              </circle>
              <text x={xAt(i)} y={ROW_WITH_Y + NODE_R + 14} fontSize="9" fill={diverged ? 'var(--color-accent)' : 'var(--color-ink-dim)'} fontWeight={diverged ? 700 : 400} textAnchor="middle">
                {STAGE_SHORT_LABELS[a] ?? a}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
