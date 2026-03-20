#!/usr/bin/env node
/**
 * End-to-end WASM test — verifies the full pipeline produces valid results.
 *
 * Tests:
 *   1. Benchmark fixture (layered_app) — known small graph
 *   2. Topo self-analysis — real codebase (requires uv run topo parse)
 *
 * Usage:
 *   node test-e2e.mjs                    # run all tests
 *   node test-e2e.mjs --skip-parse       # skip Python parse step (uses cached /tmp/topo-graph.json)
 */

import { readFileSync, existsSync } from "fs";
import { execSync } from "child_process";
import { dirname, join } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = join(__dirname, "../../..");
const wasmPath = join(__dirname, "../../topo-analyzer/pkg/topo_analyzer_bg.wasm");
const wasmJsPath = join(__dirname, "../../topo-analyzer/pkg/topo_analyzer.js");

const skipParse = process.argv.includes("--skip-parse");

let passed = 0;
let failed = 0;

function assert(condition, message) {
  if (condition) {
    passed++;
  } else {
    failed++;
    console.error(`  FAIL: ${message}`);
  }
}

// ── Load WASM ─────────────────────────────────────────────────────

console.log("Loading WASM...");
const { initSync, analyze } = await import(wasmJsPath);
const wasmBytes = readFileSync(wasmPath);
initSync({ module: wasmBytes });
console.log(`  WASM loaded (${(wasmBytes.length / 1024).toFixed(0)} KB)`);

// ── Test 1: Benchmark fixture ─────────────────────────────────────

console.log("\n═══ Test 1: Benchmark fixture (layered_app) ═══");

const fixturePath = join(repoRoot, "benchmark/datasets/architecture/layered_app/graph.json");
if (existsSync(fixturePath)) {
  const graph = JSON.parse(readFileSync(fixturePath, "utf-8"));

  const t0 = performance.now();
  const result = JSON.parse(analyze(JSON.stringify(graph)));
  const elapsed = performance.now() - t0;

  console.log(`  Analyzed ${graph.nodes.length} nodes, ${graph.edges.length} edges in ${elapsed.toFixed(1)}ms`);

  // Structural assertions.
  assert(Object.keys(result.clusters).length === graph.nodes.length,
    `cluster count (${Object.keys(result.clusters).length}) should match node count (${graph.nodes.length})`);
  assert(Object.keys(result.betweenness).length === graph.nodes.length,
    `betweenness count should match node count`);
  assert(Array.isArray(result.eigenvalues) && result.eigenvalues.length > 0,
    `eigenvalues should be non-empty`);
  assert(typeof result.fiedler_value === "number" && !isNaN(result.fiedler_value),
    `fiedler_value should be a number`);
  assert(typeof result.silhouette === "number" && !isNaN(result.silhouette),
    `silhouette should be a number`);
  assert(Array.isArray(result.component_sizes),
    `component_sizes should be an array`);
  assert(Array.isArray(result.sccs),
    `sccs should be an array`);
  assert(Array.isArray(result.connected_components) && result.connected_components.length > 0,
    `connected_components should be non-empty`);
  assert(typeof result.degenerate === "boolean",
    `degenerate should be a boolean`);

  // Every node should have a cluster.
  for (const node of graph.nodes) {
    assert(node.id in result.clusters,
      `node ${node.id} should have a cluster assignment`);
  }

  // Fingerprints dimension should be consistent.
  const fpDims = new Set(Object.values(result.fingerprints).map((v) => v.length));
  assert(fpDims.size <= 2, `fingerprint dimensions should be consistent (got ${[...fpDims]})`);

  console.log(`  Modules: ${new Set(Object.values(result.clusters)).size}, Silhouette: ${result.silhouette.toFixed(4)}, Fiedler: ${result.fiedler_value.toFixed(4)}`);
} else {
  console.log("  SKIP: fixture not found");
}

// ── Test 2: Topo self-analysis ────────────────────────────────────

console.log("\n═══ Test 2: Topo self-analysis ═══");

const topoGraphPath = "/tmp/topo-graph.json";

if (!skipParse || !existsSync(topoGraphPath)) {
  console.log("  Parsing topo codebase...");
  try {
    execSync(`uv run topo parse packages/ -o ${topoGraphPath}`, {
      cwd: repoRoot,
      stdio: ["pipe", "pipe", "pipe"],
      timeout: 30000,
    });
  } catch (e) {
    console.error(`  FAIL: parse failed: ${e.stderr?.toString() || e.message}`);
    failed++;
  }
}

if (existsSync(topoGraphPath)) {
  const graph = JSON.parse(readFileSync(topoGraphPath, "utf-8"));
  console.log(`  Graph: ${graph.nodes.length} nodes, ${graph.edges.length} edges`);

  const t0 = performance.now();
  const result = JSON.parse(analyze(JSON.stringify(graph)));
  const elapsed = performance.now() - t0;

  console.log(`  Analysis: ${elapsed.toFixed(1)}ms`);

  // Structural assertions.
  assert(Object.keys(result.clusters).length === graph.nodes.length,
    `cluster count should match node count`);
  assert(Object.keys(result.betweenness).length === graph.nodes.length,
    `betweenness count should match node count`);
  assert(result.eigenvalues.length > 0,
    `eigenvalues should be non-empty`);
  assert(!isNaN(result.silhouette),
    `silhouette should not be NaN`);
  assert(!isNaN(result.fiedler_value),
    `fiedler_value should not be NaN`);
  assert(graph.nodes.length > 100,
    `topo should have >100 nodes (got ${graph.nodes.length})`);
  assert(graph.edges.length > 200,
    `topo should have >200 edges (got ${graph.edges.length})`);

  // Verify non-trivial results.
  const moduleCount = new Set(Object.values(result.clusters)).size;
  assert(moduleCount >= 2,
    `should find >=2 modules (got ${moduleCount})`);
  assert(result.fiedler_value > 0,
    `fiedler_value should be positive (got ${result.fiedler_value})`);

  // Check key topo nodes are present.
  const expectedNodes = [
    "topo_parser.graph.CodeGraph",
    "topo_analyzer.analysis.analyze",
    "topo_cli.main",
  ];
  for (const nid of expectedNodes) {
    assert(nid in result.clusters, `expected node "${nid}" to be in clusters`);
  }

  console.log(`  Modules: ${moduleCount}, Silhouette: ${result.silhouette.toFixed(4)}, Fiedler: ${result.fiedler_value.toFixed(4)}`);
  console.log(`  SCCs: ${result.sccs.length}, Components: ${result.component_sizes.length}`);
} else {
  console.log("  SKIP: topo-graph.json not found (run without --skip-parse)");
}

// ── Test 3: Edge cases ────────────────────────────────────────────

console.log("\n═══ Test 3: Edge cases ═══");

// Empty graph.
const emptyResult = JSON.parse(analyze(JSON.stringify({ nodes: [], edges: [] })));
assert(emptyResult.degenerate === true, "empty graph should be degenerate");
assert(Object.keys(emptyResult.clusters).length === 0, "empty graph should have 0 clusters");

// Single node.
const singleResult = JSON.parse(analyze(JSON.stringify({
  nodes: [{ id: "a", kind: "function" }],
  edges: [],
})));
assert(Object.keys(singleResult.clusters).length === 1, "single node should have 1 cluster");

// Disconnected nodes.
const disconnResult = JSON.parse(analyze(JSON.stringify({
  nodes: [{ id: "a", kind: "function" }, { id: "b", kind: "function" }, { id: "c", kind: "function" }],
  edges: [],
})));
assert(disconnResult.connected_components.length === 3, "3 disconnected nodes should have 3 components");

console.log("  Edge cases passed");

// ── Summary ───────────────────────────────────────────────────────

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
