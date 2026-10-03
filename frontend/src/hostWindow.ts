import type { HostTimelineEntry } from './types';

/**
 * Picks a window worth showing by default for a demo host: the end of its
 * first non-benign action segment, so pages that default to "the current
 * window" land somewhere an attack is actually in progress instead of the
 * host's final window -- which, by the generator's own design, is always a
 * benign tail (see app/data_gen/generator.py). Returns undefined for a
 * fully-benign host or an empty timeline, so callers naturally fall back to
 * "the last real window" for those.
 */
export function pickInterestingWindowIdx(timeline: HostTimelineEntry[]): number | undefined {
  let currentAction: string | null = null;
  let currentEnd: number | null = null;
  for (const row of timeline) {
    if (row.true_stage !== currentAction) {
      if (currentAction && currentAction !== 'benign') return currentEnd ?? undefined;
      currentAction = row.true_stage;
    }
    currentEnd = row.window_idx;
  }
  if (currentAction && currentAction !== 'benign') return currentEnd ?? undefined;
  return undefined;
}
