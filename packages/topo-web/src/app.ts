import { init, analyze, summarize } from "./index.js";
import type { AnalyzerInput, AnalysisSummary, RoleAssignment } from "./types.js";

// ── DOM refs ──────────────────────────────────────────────────────

const uploadZone = document.getElementById("upload-zone")!;
const fileInput = document.getElementById("file-input") as HTMLInputElement;
const demoBtn = document.getElementById("demo-btn")!;
const loading = document.getElementById("loading")!;
const error = document.getElementById("error")!;
const results = document.getElementById("results")!;
const timing = document.getElementById("timing")!;
const statsGrid = document.getElementById("stats-grid")!;
const modulesSection = document.getElementById("modules-section")!;
const rolesSection = document.getElementById("roles-section")!;
const sccsSection = document.getElementById("sccs-section")!;

// ── Init WASM ─────────────────────────────────────────────────────

let wasmReady = false;

async function ensureWasm(): Promise<void> {
  if (wasmReady) return;
  await init();
  wasmReady = true;
}

// ── File handling ─────────────────────────────────────────────────

uploadZone.addEventListener("click", (e) => {
  if ((e.target as HTMLElement).id !== "demo-btn") fileInput.click();
});

uploadZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadZone.classList.add("dragover");
});

uploadZone.addEventListener("dragleave", () => {
  uploadZone.classList.remove("dragover");
});

uploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadZone.classList.remove("dragover");
  const file = e.dataTransfer?.files[0];
  if (file) processFile(file);
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (file) processFile(file);
});

demoBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  runDemo();
});

async function processFile(file: File): Promise<void> {
  try {
    const text = await file.text();
    const input = JSON.parse(text) as AnalyzerInput;
    if (!input.nodes || !input.edges) {
      throw new Error("JSON must have 'nodes' and 'edges' arrays");
    }
    await runAnalysis(input);
  } catch (e) {
    showError(e instanceof Error ? e.message : String(e));
  }
}

// ── Demo graph ────────────────────────────────────────────────────

function makeDemo(): AnalyzerInput {
  const nodes = [
    // Auth module
    ...["auth.login", "auth.logout", "auth.verify_token", "auth.refresh", "auth.hash_password"].map(
      (id) => ({ id, kind: "function" }),
    ),
    // Users module
    ...["users.get_user", "users.create_user", "users.update_user", "users.delete_user", "users.list_users"].map(
      (id) => ({ id, kind: "function" }),
    ),
    // API module
    ...["api.router", "api.middleware", "api.error_handler", "api.rate_limiter"].map(
      (id) => ({ id, kind: "function" }),
    ),
    // DB module
    ...["db.connect", "db.query", "db.transaction", "db.migrate"].map(
      (id) => ({ id, kind: "function" }),
    ),
    // Orphan
    { id: "utils.deprecated_helper", kind: "function" },
  ];

  const edges = [
    // Auth internal
    { source: "auth.login", target: "auth.verify_token", kind: "calls" },
    { source: "auth.login", target: "auth.hash_password", kind: "calls" },
    { source: "auth.refresh", target: "auth.verify_token", kind: "calls" },
    { source: "auth.logout", target: "auth.verify_token", kind: "calls" },
    // Users internal
    { source: "users.create_user", target: "users.get_user", kind: "calls" },
    { source: "users.update_user", target: "users.get_user", kind: "calls" },
    { source: "users.delete_user", target: "users.get_user", kind: "calls" },
    { source: "users.list_users", target: "users.get_user", kind: "calls" },
    // API → Auth, Users
    { source: "api.router", target: "auth.login", kind: "calls" },
    { source: "api.router", target: "auth.logout", kind: "calls" },
    { source: "api.router", target: "users.list_users", kind: "calls" },
    { source: "api.router", target: "users.create_user", kind: "calls" },
    { source: "api.middleware", target: "auth.verify_token", kind: "calls" },
    { source: "api.middleware", target: "api.rate_limiter", kind: "calls" },
    { source: "api.error_handler", target: "api.router", kind: "calls" },
    // DB usage
    { source: "users.get_user", target: "db.query", kind: "calls" },
    { source: "users.create_user", target: "db.transaction", kind: "calls" },
    { source: "users.update_user", target: "db.transaction", kind: "calls" },
    { source: "users.delete_user", target: "db.transaction", kind: "calls" },
    { source: "auth.login", target: "db.query", kind: "calls" },
    { source: "db.transaction", target: "db.query", kind: "calls" },
    { source: "db.migrate", target: "db.connect", kind: "calls" },
    { source: "db.query", target: "db.connect", kind: "calls" },
    // A cycle (SCC)
    { source: "auth.verify_token", target: "auth.refresh", kind: "calls" },
  ];

  return { nodes, edges };
}

async function runDemo(): Promise<void> {
  await runAnalysis(makeDemo());
}

// ── Analysis ──────────────────────────────────────────────────────

async function runAnalysis(input: AnalyzerInput): Promise<void> {
  error.style.display = "none";
  results.style.display = "none";
  loading.style.display = "block";

  try {
    await ensureWasm();

    const t0 = performance.now();
    const output = analyze(input);
    const t1 = performance.now();
    const summary = summarize(output, input.edges);
    const t2 = performance.now();

    timing.textContent = `Analysis: ${(t1 - t0).toFixed(1)}ms · Role classification: ${(t2 - t1).toFixed(1)}ms · Total: ${(t2 - t0).toFixed(1)}ms`;

    renderResults(summary, input);
    loading.style.display = "none";
    results.style.display = "block";
    uploadZone.style.display = "none";
  } catch (e) {
    loading.style.display = "none";
    showError(e instanceof Error ? e.message : String(e));
  }
}

function showError(msg: string): void {
  error.textContent = msg;
  error.style.display = "block";
}

// ── Rendering ─────────────────────────────────────────────────────

function renderResults(summary: AnalysisSummary, input: AnalyzerInput): void {
  renderStats(summary);
  renderModules(summary);
  renderRoles(summary);
  renderSCCs(input, summary);
}

function renderStats(s: AnalysisSummary): void {
  const silClass = s.silhouette >= 0.5 ? "good" : s.silhouette >= 0.3 ? "warn" : "bad";
  const fiedlerClass = s.fiedlerValue > 0.1 ? "good" : s.fiedlerValue > 0.01 ? "warn" : "bad";

  statsGrid.innerHTML = `
    <div class="stat-card"><div class="value">${s.nodeCount}</div><div class="label">Nodes</div></div>
    <div class="stat-card"><div class="value">${s.edgeCount}</div><div class="label">Edges</div></div>
    <div class="stat-card"><div class="value">${s.moduleCount}</div><div class="label">Modules</div></div>
    <div class="stat-card ${silClass}"><div class="value">${s.silhouette.toFixed(3)}</div><div class="label">Silhouette</div></div>
    <div class="stat-card ${fiedlerClass}"><div class="value">${s.fiedlerValue.toFixed(4)}</div><div class="label">Fiedler Value</div></div>
    <div class="stat-card"><div class="value">${s.componentCount}</div><div class="label">Components</div></div>
    <div class="stat-card ${s.sccCount > 0 ? "warn" : "good"}"><div class="value">${s.sccCount}</div><div class="label">Cycles (SCC)</div></div>
    <div class="stat-card ${s.orphans.length > 0 ? "warn" : "good"}"><div class="value">${s.orphans.length}</div><div class="label">Orphans</div></div>
    ${s.degenerate ? '<div class="stat-card bad"><div class="value">Yes</div><div class="label">Degenerate</div></div>' : ""}
  `;
}

function renderModules(s: AnalysisSummary): void {
  const moduleItems = s.modules
    .map(
      (m) => `
    <div class="module-item">
      <div class="module-header">
        <span class="module-id">Module ${m.id}</span>
        <span class="module-size">${m.size} nodes</span>
      </div>
      <div class="node-list">${m.nodeIds.map((n) => `<span class="node">${esc(n)}</span>`).join("")}</div>
    </div>
  `,
    )
    .join("");

  modulesSection.innerHTML = makeSection("Modules", moduleItems);
}

function renderRoles(s: AnalysisSummary): void {
  // Show non-regular roles first, then regular.
  const notable = s.roles.filter((r) => r.role !== "regular");
  const regular = s.roles.filter((r) => r.role === "regular");
  const sorted = [...notable, ...regular];

  const rows = sorted
    .map(
      (r) => `
    <tr>
      <td class="node-name">${esc(r.nodeId)}</td>
      <td><span class="role-badge role-${r.role}">${r.role}</span></td>
      <td class="metric">${r.degree}</td>
      <td class="metric">${r.inDegree}</td>
      <td class="metric">${r.outDegree}</td>
      <td class="metric">${r.betweenness.toFixed(4)}</td>
    </tr>
  `,
    )
    .join("");

  const table = `
    <table class="role-table">
      <thead><tr>
        <th>Node</th><th>Role</th><th>Degree</th><th>In</th><th>Out</th><th>Betweenness</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>
  `;

  rolesSection.innerHTML = makeSection("Structural Roles", table);
}

function renderSCCs(input: AnalyzerInput, s: AnalysisSummary): void {
  // Extract SCCs from the analysis output (stored in the raw output).
  // We passed edges to summarize, but SCCs come from the raw output.
  // For the demo, we detect SCCs from the role classification.
  // Actually we need the raw output. Let's re-analyze to get it.
  // Better approach: store the raw output in a closure.
  // For now, show the sccCount and list orphans.

  if (s.sccCount === 0 && s.orphans.length === 0) {
    sccsSection.innerHTML = "";
    return;
  }

  let content = "";

  if (s.orphans.length > 0) {
    content += `<div class="scc-item" style="border-left-color: var(--orange)">
      <div class="scc-label" style="color: var(--orange)">Orphan Nodes (${s.orphans.length})</div>
      ${s.orphans.map((n) => `<span class="node" style="display:inline-block;background:var(--surface);padding:1px 6px;border-radius:4px;margin:2px 4px 2px 0">${esc(n)}</span>`).join("")}
    </div>`;
  }

  if (s.sccCount > 0) {
    content += `<div class="scc-item">
      <div class="scc-label">Dependency Cycles Detected: ${s.sccCount}</div>
      <div style="color:var(--text-muted);font-size:12px">Cycles indicate tightly coupled components that may be difficult to test or refactor independently.</div>
    </div>`;
  }

  sccsSection.innerHTML = makeSection("Anomalies", content);
}

// ── Helpers ───────────────────────────────────────────────────────

function makeSection(title: string, body: string): string {
  return `
    <div class="section">
      <div class="section-header" onclick="this.querySelector('.arrow').classList.toggle('collapsed');this.nextElementSibling.classList.toggle('collapsed')">
        <span class="arrow">▼</span> ${esc(title)}
      </div>
      <div class="section-body">${body}</div>
    </div>
  `;
}

function esc(s: string): string {
  const el = document.createElement("span");
  el.textContent = s;
  return el.innerHTML;
}
