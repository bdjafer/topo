#!/usr/bin/env node
/**
 * topo-wasm — Run structural analysis via WASM from Node.js.
 *
 * Usage:
 *   node topo-wasm.mjs <graph.json> [--format json|summary]
 *   uv run topo parse . -o graph.json && node topo-wasm.mjs graph.json
 */

import { readFileSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const wasmPath = join(__dirname, "../../topo-analyzer/pkg/topo_analyzer_bg.wasm");
const wasmJsPath = join(__dirname, "../../topo-analyzer/pkg/topo_analyzer.js");

// ── Parse args ────────────────────────────────────────────────────

const args = process.argv.slice(2);
let graphPath = null;
let format = "summary";

for (let i = 0; i < args.length; i++) {
  if (args[i] === "--format" && args[i + 1]) {
    format = args[i + 1];
    i++;
  } else if (args[i] === "--json") {
    format = "json";
  } else if (!args[i].startsWith("-")) {
    graphPath = args[i];
  }
}

if (!graphPath) {
  console.error("Usage: topo-wasm <graph.json> [--format json|summary] [--json]");
  console.error("");
  console.error("Generate graph.json with: uv run topo parse <path> -o graph.json");
  process.exit(1);
}

// ── Load WASM ─────────────────────────────────────────────────────

const { initSync, analyze } = await import(wasmJsPath);

const wasmBytes = readFileSync(wasmPath);
initSync({ module: wasmBytes });

// ── Load & analyze ────────────────────────────────────────────────

const graphJson = readFileSync(graphPath, "utf-8");
const graph = JSON.parse(graphJson);

if (!graph.nodes || !graph.edges) {
  console.error("Error: JSON must have 'nodes' and 'edges' arrays");
  process.exit(1);
}

const t0 = performance.now();
const resultJson = analyze(JSON.stringify(graph));
const t1 = performance.now();
const result = JSON.parse(resultJson);

// ── Role classification (mirrors Python roles.py) ─────────────────

function percentileRanks(values) {
  const n = values.length;
  if (n <= 1) return new Array(n).fill(0);
  const sorted = [...values].sort((a, b) => a - b);
  return values.map((v) => {
    let lo = 0, hi = n;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (sorted[mid] < v) lo = mid + 1;
      else hi = mid;
    }
    return lo / (n - 1);
  });
}

function classifyRoles(output, edges) {
  const PCT = 0.9, MIN_DIR_DEG = 3, DIR_THRESH = 0.6, HUB_GAP = 2;
  const nodeIds = Object.keys(output.clusters);
  const inDeg = {}, outDeg = {};
  for (const nid of nodeIds) { inDeg[nid] = 0; outDeg[nid] = 0; }
  for (const e of edges) {
    if (e.source in outDeg) outDeg[e.source]++;
    if (e.target in inDeg) inDeg[e.target]++;
  }

  const degrees = nodeIds.map((n) => (inDeg[n] || 0) + (outDeg[n] || 0));
  const btwValues = nodeIds.map((n) => output.betweenness[n] || 0);
  const degreePcts = percentileRanks(degrees);
  const btwPcts = percentileRanks(btwValues);
  const sortedDeg = [...degrees].sort((a, b) => a - b);
  const mid = Math.floor(sortedDeg.length / 2);
  const medianDeg = sortedDeg.length % 2 === 0
    ? (sortedDeg[mid - 1] + sortedDeg[mid]) / 2
    : sortedDeg[mid];

  return nodeIds.map((nid, i) => {
    const deg = degrees[i], ind = inDeg[nid] || 0, outd = outDeg[nid] || 0;
    let role = "regular";
    if (deg === 0) role = "orphan";
    else {
      const dir = (outd - ind) / deg;
      if (btwPcts[i] >= PCT && degreePcts[i] < PCT) role = "bridge";
      else if (deg >= MIN_DIR_DEG && dir <= -DIR_THRESH) role = "utility";
      else if (deg >= MIN_DIR_DEG && dir >= DIR_THRESH) role = "entry_point";
      else if (degreePcts[i] >= PCT && deg >= medianDeg + HUB_GAP && Math.abs(dir) < DIR_THRESH) role = "hub";
    }
    return { nodeId: nid, role, degree: deg, inDegree: ind, outDegree: outd, betweenness: btwValues[i] };
  });
}

// ── Output ────────────────────────────────────────────────────────

if (format === "json") {
  console.log(JSON.stringify(result, null, 2));
  process.exit(0);
}

// Summary format.
const roles = classifyRoles(result, graph.edges);
const clusterMembers = {};
for (const [nid, cid] of Object.entries(result.clusters)) {
  (clusterMembers[cid] ??= []).push(nid);
}
const modules = Object.entries(clusterMembers)
  .map(([id, nodes]) => ({ id: Number(id), nodes: nodes.sort(), size: nodes.length }))
  .sort((a, b) => a.id - b.id);

const hubs = roles.filter((r) => r.role === "hub").sort((a, b) => b.degree - a.degree);
const bridges = roles.filter((r) => r.role === "bridge").sort((a, b) => b.betweenness - a.betweenness);
const utilities = roles.filter((r) => r.role === "utility");
const entryPoints = roles.filter((r) => r.role === "entry_point");
const orphans = roles.filter((r) => r.role === "orphan");

console.log("═══ topo structural analysis (WASM) ═══");
console.log("");
console.log(`  Nodes:        ${graph.nodes.length}`);
console.log(`  Edges:        ${graph.edges.length}`);
console.log(`  Modules:      ${modules.length}${result.degenerate ? " (degenerate — package fallback)" : ""}`);
console.log(`  Silhouette:   ${result.silhouette.toFixed(4)}`);
console.log(`  Fiedler:      ${result.fiedler_value.toFixed(4)}`);
console.log(`  Components:   ${result.component_sizes.length}`);
console.log(`  Cycles (SCC): ${result.sccs.length}`);
console.log(`  Time:         ${(t1 - t0).toFixed(1)}ms`);

console.log("");
console.log("── Modules ──");
for (const m of modules) {
  console.log(`  [${m.id}] ${m.size} nodes: ${m.nodes.slice(0, 8).join(", ")}${m.size > 8 ? ` (+${m.size - 8} more)` : ""}`);
}

if (hubs.length) {
  console.log("");
  console.log("── Hubs ──");
  for (const h of hubs.slice(0, 5)) {
    console.log(`  ${h.nodeId}  (degree=${h.degree}, btw=${h.betweenness.toFixed(4)})`);
  }
}

if (bridges.length) {
  console.log("");
  console.log("── Bridges ──");
  for (const b of bridges.slice(0, 5)) {
    console.log(`  ${b.nodeId}  (degree=${b.degree}, btw=${b.betweenness.toFixed(4)})`);
  }
}

if (orphans.length) {
  console.log("");
  console.log("── Orphans ──");
  for (const o of orphans) {
    console.log(`  ${o.nodeId}`);
  }
}

if (result.sccs.length) {
  console.log("");
  console.log("── Dependency Cycles ──");
  for (const scc of result.sccs) {
    console.log(`  ${scc.join(" → ")} → ${scc[0]}`);
  }
}

console.log("");
console.log(`Role summary: ${hubs.length} hub, ${bridges.length} bridge, ${utilities.length} utility, ${entryPoints.length} entry_point, ${orphans.length} orphan, ${roles.filter((r) => r.role === "regular").length} regular`);
