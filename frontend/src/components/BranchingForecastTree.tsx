import { useMemo, useState } from 'react';
import type { BranchNode, BranchPath } from '../types';
import { STAGE_COLORS, STAGE_LABELS } from '../api';

const COL_WIDTH = 176;
const NODE_W = 152;
const NODE_H = 44;
const ROW_GAP = 10;

interface LaidOutNode {
  node: BranchNode;
  x: number;
  y: number;
  children: LaidOutNode[];
}

function layout(root: BranchNode): { laid: LaidOutNode; leafCount: number } {
  let leafCounter = 0;

  function place(node: BranchNode, depth: number): LaidOutNode {
    const x = depth * COL_WIDTH;
    if (!node.children.length) {
      const y = leafCounter * (NODE_H + ROW_GAP) + NODE_H / 2;
      leafCounter += 1;
      return { node, x, y, children: [] };
    }
    const children = node.children.map((c) => place(c, depth + 1));
    const y = children.reduce((s, c) => s + c.y, 0) / children.length;
    return { node, x, y, children };
  }

  const laid = place(root, 0);
  return { laid, leafCount: Math.max(leafCounter, 1) };
}

function stageLabel(stage: string | null): string {
  if (stage == null) return 'Now';
  return STAGE_LABELS[stage] ?? stage;
}

function riskColor(p: number | null): string {
  if (p == null) return 'var(--color-ink-faint)';
  if (p > 0.66) return 'var(--color-bad)';
  if (p > 0.33) return 'var(--color-warn)';
  return 'var(--color-good)';
}

function flattenEdges(node: LaidOutNode, out: { from: LaidOutNode; to: LaidOutNode }[] = []) {
  for (const child of node.children) {
    out.push({ from: node, to: child });
    flattenEdges(child, out);
  }
  return out;
}

function flattenNodes(node: LaidOutNode, out: LaidOutNode[] = []) {
  out.push(node);
  for (const child of node.children) flattenNodes(child, out);
  return out;
}

export default function BranchingForecastTree({ tree }: { tree: BranchNode }) {
  const [selected, setSelected] = useState<LaidOutNode | null>(null);

  const { laid, leafCount } = useMemo(() => layout(tree), [tree]);
  const edges = useMemo(() => flattenEdges(laid), [laid]);
  const nodes = useMemo(() => flattenNodes(laid), [laid]);

  const maxDepth = Math.max(...nodes.map((n) => n.node.depth));
  const width = (maxDepth + 1) * COL_WIDTH + NODE_W;
  const height = leafCount * (NODE_H + ROW_GAP);

  const active = selected ?? laid;

  return (
    <div className="flex flex-col lg:flex-row gap-4">
      <div className="flex-1 overflow-x-auto">
        <svg width={width} height={Math.max(height, 60)} className="min-w-full">
          {edges.map(({ from, to }, i) => {
            const x1 = from.x + NODE_W;
            const y1 = from.y;
            const x2 = to.x;
            const y2 = to.y;
            const midX = (x1 + x2) / 2;
            const strokeWidth = 1 + to.node.path_probability * 5;
            const stageColor = to.node.stage ? STAGE_COLORS[to.node.stage] ?? '#8b8a9e' : '#8b8a9e';
            return (
              <path
                key={i}
                d={`M ${x1} ${y1} C ${midX} ${y1}, ${midX} ${y2}, ${x2} ${y2}`}
                fill="none"
                stroke={stageColor}
                strokeOpacity={0.35 + to.node.path_probability * 0.5}
                strokeWidth={strokeWidth}
              />
            );
          })}
          {nodes.map((n, i) => {
            const isRoot = n.node.stage == null;
            const color = isRoot ? '#6c5dd3' : STAGE_COLORS[n.node.stage as string] ?? '#8b8a9e';
            const isSelected = active === n;
            return (
              <g
                key={i}
                transform={`translate(${n.x}, ${n.y - NODE_H / 2})`}
                className="cursor-pointer"
                onClick={() => setSelected(n)}
              >
                <rect
                  width={NODE_W}
                  height={NODE_H}
                  rx={12}
                  fill={isSelected ? color : 'white'}
                  fillOpacity={isSelected ? 0.14 : 1}
                  stroke={color}
                  strokeWidth={isSelected ? 2 : 1.2}
                />
                <circle cx={14} cy={NODE_H / 2} r={4} fill={color} />
                <text x={26} y={NODE_H / 2 - 3} fontSize={10.5} fontWeight={600} fill="var(--color-ink)">
                  {stageLabel(n.node.stage).length > 17 ? stageLabel(n.node.stage).slice(0, 16) + '…' : stageLabel(n.node.stage)}
                </text>
                <text x={26} y={NODE_H / 2 + 11} fontSize={9.5} fill="var(--color-ink-dim)">
                  {n.node.step_probability != null ? `${(n.node.step_probability * 100).toFixed(0)}% step` : 'observed'}
                  {n.node.path_probability < 1 ? ` · ${(n.node.path_probability * 100).toFixed(0)}% path` : ''}
                </text>
                {n.node.infiltration_probability != null && (
                  <circle cx={NODE_W - 10} cy={10} r={4} fill={riskColor(n.node.infiltration_probability)} />
                )}
              </g>
            );
          })}
        </svg>
      </div>

      <div className="lg:w-64 shrink-0 flex flex-col gap-3">
        <div className="text-xs text-[var(--color-ink-faint)]">
          Click a node to inspect its branch. Edge thickness/opacity = cumulative path probability.
        </div>
        <div className="bg-[var(--color-accent-soft)] rounded-xl p-3">
          <div className="text-xs font-semibold text-[var(--color-ink)] mb-1">
            {selected ? stageLabel(selected.node.stage) : 'Now'}
          </div>
          {selected?.node.attack_mapping?.technique_id && (
            <div className="text-[11px] text-[var(--color-ink-dim)] mb-1">
              {selected.node.attack_mapping.technique_id} — {selected.node.attack_mapping.technique_name}
              <div className="mt-0.5">{selected.node.attack_mapping.tactic}</div>
            </div>
          )}
          <div className="text-[11px] text-[var(--color-ink-dim)] flex flex-col gap-0.5 mt-1.5">
            {selected?.node.step_probability != null && (
              <span>Step probability: <strong className="text-[var(--color-ink)]">{(selected.node.step_probability * 100).toFixed(1)}%</strong></span>
            )}
            <span>Path probability: <strong className="text-[var(--color-ink)]">{((selected?.node.path_probability ?? 1) * 100).toFixed(1)}%</strong></span>
            {selected?.node.infiltration_probability != null && (
              <span>Infiltration risk at this node: <strong className="text-[var(--color-ink)]">{(selected.node.infiltration_probability * 100).toFixed(1)}%</strong></span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export function PathSummaryList({ paths, title }: { paths: BranchPath[]; title: string }) {
  return (
    <div>
      <div className="text-xs font-semibold text-[var(--color-ink)] mb-2">{title}</div>
      <div className="flex flex-col gap-2">
        {paths.map((p, i) => (
          <div key={i} className="text-xs bg-[var(--color-accent-soft)] rounded-xl p-2.5">
            <div className="flex items-center justify-between mb-1">
              <span className="font-semibold text-[var(--color-ink)]">
                {(p.path_probability * 100).toFixed(1)}% path
              </span>
              <span style={{ color: riskColor(p.final_infiltration_probability) }} className="font-semibold">
                {(p.final_infiltration_probability * 100).toFixed(0)}% risk @ horizon
              </span>
            </div>
            <div className="flex flex-wrap items-center gap-1 text-[var(--color-ink-dim)]">
              {p.mitre_kill_chain.map((m, j) => (
                <span key={j} className="flex items-center gap-1">
                  <span className="w-1.5 h-1.5 rounded-full" style={{ background: STAGE_COLORS[m.stage] ?? '#8b8a9e' }} />
                  {STAGE_LABELS[m.stage] ?? m.stage}
                  {m.technique_id ? ` (${m.technique_id})` : ''}
                  {j < p.mitre_kill_chain.length - 1 && <span className="mx-0.5">→</span>}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
