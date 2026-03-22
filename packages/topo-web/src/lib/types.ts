// ── Web App Contracts ─────────────────────────────────────────────
// These types define the shape consumed by the frontend.
// They map to the CLI's AnalysisOutput + DOMAINS.md domain tree spec.

/** Root analysis result — single payload from POST /api/analyze. */
export interface TopoResult {
  meta: ResultMeta;
  graph: GraphData;
  domain: DomainNode;
  health: HealthScore;
  issues: Issue[];
}

export interface ResultMeta {
  repo: string;
  analyzed_at: string;
  node_count: number;
  edge_count: number;
}

// ── Graph ─────────────────────────────────────────────────────────

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface GraphNode {
  id: string;
  kind: NodeKind;
  file: string;
  line: number;
  role: StructuralRole;
  /** Domain path in the tree, e.g. "auth/token" */
  domain_path: string;
  /** Flat cluster assignment from spectral analysis */
  cluster_id: number;
}

export type NodeKind = "module" | "class" | "function" | "interface";

export type StructuralRole =
  | "hub"
  | "bridge"
  | "utility"
  | "entry_point"
  | "orphan"
  | "regular";

export type EdgeKind = "calls" | "imports" | "inherits" | "defines";

export interface GraphEdge {
  source: string;
  target: string;
  kind: EdgeKind;
}

// ── Domain Tree ───────────────────────────────────────────────────

export interface DomainNode {
  label: string;
  /** Unique path, e.g. "auth/token" */
  path: string;
  depth: number;
  size: number;
  /** Node IDs — populated at leaves only */
  members: string[];
  top_terms: string[];
  coherence: number | null;
  archetype: { label: string; confidence: number } | null;
  health: HealthScore | null;
  children: DomainNode[];
  /** Cross-cutting nodes — only at root */
  cross_cutting: CrossCuttingNode[];
  /** Nodes near domain boundaries */
  boundary_nodes: string[];
  /** Inter-domain dependencies */
  dependencies: DomainDependency[];
}

export interface HealthScore {
  topo_health_score: number; // [0, 1]
  coherence: number; // [0, 1]
  flow: number; // [0, 1]
}

// ── Issues ────────────────────────────────────────────────────────

export interface Issue {
  id: string;
  kind: string;
  title: string;
  description: string;
  severity: number; // [0, 1]
  severity_label: "high" | "medium" | "low";
  confidence: number;
  anchors: IssueAnchor[];
}

export interface IssueAnchor {
  node_id: string;
  file: string;
  line: number;
  kind: string;
}

// ── Supporting Types ──────────────────────────────────────────────

export interface CrossCuttingNode {
  node_id: string;
  silhouette: number;
  caller_diversity: number;
  dominant_edges: Record<string, number>;
}

export interface DomainDependency {
  source_path: string;
  target_path: string;
  weight: number;
  edge_kinds: Record<string, number>;
}
