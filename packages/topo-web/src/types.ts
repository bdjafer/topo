/** TypeScript types matching the Rust topo-core API. */

export interface NodeEntry {
  id: string;
  kind: string;
}

export interface EdgeEntry {
  source: string;
  target: string;
  kind: string;
}

export interface AnalyzerInput {
  nodes: NodeEntry[];
  edges: EdgeEntry[];
  /** Number of clusters (auto-detect if omitted). */
  k?: number;
  /** Which edge kinds to include (e.g. ["calls", "imports"]). */
  edge_kinds?: string[];
  /** Per-layer weights for multilayer analysis. */
  layer_weights?: Record<string, number>;
}

export interface AnalyzerOutput {
  /** Spectral fingerprints: node_id → eigenvector coordinates. */
  fingerprints: Record<string, number[]>;
  /** Cluster assignments: node_id → cluster_id. */
  clusters: Record<string, number>;
  /** Eigenvalues from the primary component. */
  eigenvalues: number[];
  /** Fiedler value (second-smallest Laplacian eigenvalue). */
  fiedler_value: number;
  /** Silhouette score of the clustering. */
  silhouette: number;
  /** Sizes of connected components (descending). */
  component_sizes: number[];
  /** Betweenness centrality: node_id → score. */
  betweenness: Record<string, number>;
  /** Strongly connected components with >1 node. */
  sccs: string[][];
  /** Connected components. */
  connected_components: string[][];
  /** Whether clustering was degenerate (fell back to package grouping). */
  degenerate: boolean;
}

/** A module (cluster) derived from spectral analysis. */
export interface Module {
  id: number;
  nodeIds: string[];
  size: number;
}

/** Structural role of a node. */
export type StructuralRole =
  | "hub"
  | "bridge"
  | "utility"
  | "entry_point"
  | "orphan"
  | "regular";

/** Role assignment with supporting metrics. */
export interface RoleAssignment {
  nodeId: string;
  role: StructuralRole;
  degree: number;
  betweenness: number;
  inDegree: number;
  outDegree: number;
}

/** High-level analysis summary derived from AnalyzerOutput. */
export interface AnalysisSummary {
  nodeCount: number;
  edgeCount: number;
  moduleCount: number;
  modules: Module[];
  silhouette: number;
  fiedlerValue: number;
  componentCount: number;
  sccCount: number;
  degenerate: boolean;
  roles: RoleAssignment[];
  topBridges: RoleAssignment[];
  topHubs: RoleAssignment[];
  orphans: string[];
}
