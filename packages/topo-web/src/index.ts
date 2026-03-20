/**
 * @topo/core — Structural analysis for codebases, powered by WASM.
 *
 * Usage:
 *   import { init, analyze, summarize } from "@topo/core";
 *   await init();
 *   const result = analyze({ nodes, edges });
 *   const summary = summarize(result, edges.length);
 */

import type {
  AnalyzerInput,
  AnalyzerOutput,
  AnalysisSummary,
  Module,
  RoleAssignment,
  StructuralRole,
} from "./types.js";

export type {
  AnalyzerInput,
  AnalyzerOutput,
  AnalysisSummary,
  Module,
  RoleAssignment,
  StructuralRole,
  NodeEntry,
  EdgeEntry,
} from "./types.js";

// Re-export the raw WASM init for advanced usage.
import initWasm, {
  analyze as wasmAnalyze,
  type InitInput,
} from "../wasm/topo_analyzer.js";

let initialized = false;

/**
 * Initialize the WASM module. Must be called once before `analyze()`.
 * Accepts an optional URL or buffer for the .wasm file.
 */
export async function init(input?: InitInput): Promise<void> {
  if (initialized) return;
  await initWasm(input);
  initialized = true;
}

/**
 * Run structural analysis on a code graph.
 * Requires `init()` to have been called first.
 */
export function analyze(input: AnalyzerInput): AnalyzerOutput {
  if (!initialized) {
    throw new Error("Call init() before analyze()");
  }
  const resultJson = wasmAnalyze(JSON.stringify(input));
  return JSON.parse(resultJson) as AnalyzerOutput;
}

// ── Role classification (mirrors Python roles.py) ──────────────────

const PCT_THRESHOLD = 0.9;
const MIN_DIRECTIONAL_DEGREE = 3;
const DIRECTION_THRESHOLD = 0.6;
const MIN_HUB_GAP = 2;

function percentileRanks(values: number[]): number[] {
  const n = values.length;
  if (n <= 1) return new Array(n).fill(0);
  const sorted = [...values].sort((a, b) => a - b);
  return values.map((v) => {
    let lo = 0,
      hi = n;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (sorted[mid] < v) lo = mid + 1;
      else hi = mid;
    }
    return lo / (n - 1);
  });
}

function classifyNode(
  degree: number,
  inDegree: number,
  outDegree: number,
  degreePct: number,
  betweennessPct: number,
  medianDegree: number,
): StructuralRole {
  if (degree === 0) return "orphan";

  const direction = (outDegree - inDegree) / degree;

  if (betweennessPct >= PCT_THRESHOLD && degreePct < PCT_THRESHOLD)
    return "bridge";

  if (degree >= MIN_DIRECTIONAL_DEGREE && direction <= -DIRECTION_THRESHOLD)
    return "utility";

  if (degree >= MIN_DIRECTIONAL_DEGREE && direction >= DIRECTION_THRESHOLD)
    return "entry_point";

  if (
    degreePct >= PCT_THRESHOLD &&
    degree >= medianDegree + MIN_HUB_GAP &&
    Math.abs(direction) < DIRECTION_THRESHOLD
  )
    return "hub";

  return "regular";
}

/**
 * Classify structural roles from analyzer output.
 * This mirrors the Python `roles.py` logic, running entirely in JS.
 */
export function classifyRoles(
  output: AnalyzerOutput,
  edges: { source: string; target: string }[],
): RoleAssignment[] {
  const nodeIds = Object.keys(output.clusters);

  // Build adjacency.
  const inDeg: Record<string, number> = {};
  const outDeg: Record<string, number> = {};
  for (const nid of nodeIds) {
    inDeg[nid] = 0;
    outDeg[nid] = 0;
  }
  for (const e of edges) {
    if (e.source in outDeg) outDeg[e.source]++;
    if (e.target in inDeg) inDeg[e.target]++;
  }

  const degrees = nodeIds.map((n) => (inDeg[n] || 0) + (outDeg[n] || 0));
  const btwValues = nodeIds.map((n) => output.betweenness[n] || 0);

  const degreePcts = percentileRanks(degrees);
  const btwPcts = percentileRanks(btwValues);

  const sortedDegrees = [...degrees].sort((a, b) => a - b);
  const mid = Math.floor(sortedDegrees.length / 2);
  const medianDegree =
    sortedDegrees.length % 2 === 0
      ? (sortedDegrees[mid - 1] + sortedDegrees[mid]) / 2
      : sortedDegrees[mid];

  return nodeIds.map((nid, i) => ({
    nodeId: nid,
    role: classifyNode(
      degrees[i],
      inDeg[nid] || 0,
      outDeg[nid] || 0,
      degreePcts[i],
      btwPcts[i],
      medianDegree,
    ),
    degree: degrees[i],
    betweenness: btwValues[i],
    inDegree: inDeg[nid] || 0,
    outDegree: outDeg[nid] || 0,
  }));
}

/**
 * Produce a high-level summary from raw analyzer output.
 */
export function summarize(
  output: AnalyzerOutput,
  edges: { source: string; target: string }[],
): AnalysisSummary {
  // Build modules from clusters.
  const clusterMembers: Record<number, string[]> = {};
  for (const [nid, cid] of Object.entries(output.clusters)) {
    (clusterMembers[cid] ??= []).push(nid);
  }
  const modules: Module[] = Object.entries(clusterMembers)
    .map(([id, nodeIds]) => ({
      id: Number(id),
      nodeIds: nodeIds.sort(),
      size: nodeIds.length,
    }))
    .sort((a, b) => a.id - b.id);

  const roles = classifyRoles(output, edges);

  return {
    nodeCount: Object.keys(output.clusters).length,
    edgeCount: edges.length,
    moduleCount: modules.length,
    modules,
    silhouette: output.silhouette,
    fiedlerValue: output.fiedler_value,
    componentCount: output.component_sizes.length,
    sccCount: output.sccs.length,
    degenerate: output.degenerate,
    roles,
    topBridges: roles.filter((r) => r.role === "bridge").sort((a, b) => b.betweenness - a.betweenness),
    topHubs: roles.filter((r) => r.role === "hub").sort((a, b) => b.degree - a.degree),
    orphans: roles.filter((r) => r.role === "orphan").map((r) => r.nodeId),
  };
}
