"""Topo Health Score (THS) computation from R-GIN model outputs.

Implements the HEALTH.md specification:
  THS = coherence^α × flow^(1-α)

Where:
  coherence = clamp(1 - median(reconstruction_error), 0, 1)
  flow = cycle_freedom × layer_conformance
  α = 0.7 (initial hypothesis, subject to calibration)

Known simplifications vs HEALTH.md spec:
  - layer_conformance uses direction_surprise alone, without the semantic depth
    probe for layer assignment. The full spec requires per-module semantic
    centroids + a learned depth probe (w^T · centroid + b) to identify layer
    ordering, and only then uses direction_surprise to weight violations.
    Here, we treat all edges with positive direction_surprise as violations.
    This is valid when R has strong asymmetry (which our model has: R_asymmetry=1.24).
  - SCC false positive suppression (trait cycles, test-only cycles) is not yet
    implemented. The Rust analyzer handles this at a higher level.
"""

import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from topo_model.config import RGINConfig
from topo_model.rgin import RGIN


# R asymmetry threshold below which direction_surprise is unreliable
# (spec: HEALTH.md line 238)
R_ASYMMETRY_THRESHOLD = 0.1


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class HealthScore:
    """Topo Health Score output, matching HEALTH.md spec."""
    topo_health_score: float  # THS ∈ [0, 1]
    coherence: float          # ∈ [0, 1]
    flow: float               # ∈ [0, 1]
    # Sub-components for diagnostics
    cycle_freedom: float      # ∈ [0, 1]
    layer_conformance: float  # ∈ [0, 1]
    # Metadata
    n_nodes: int
    n_edges: int
    median_reconstruction_error: float
    r_asymmetry: float
    alpha: float = 0.7

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: Path, device: torch.device = torch.device("cpu")) -> RGIN:
    """Load a trained R-GIN model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config_dict = ckpt.get("config", {})
    if isinstance(config_dict, dict) and config_dict:
        config = RGINConfig(**{
            k: v for k, v in config_dict.items()
            if k in RGINConfig.__dataclass_fields__
        })
    else:
        config = RGINConfig()

    model = RGIN(config)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_batched(data) -> "HeteroData":
    """Ensure a single HeteroData has batch attributes for model forward pass.

    PyG's DataLoader creates .batch (node→graph index) and .ptr (graph boundaries)
    automatically during batching. For single-graph inference, we add them manually.
    """
    n = data["node"].num_nodes
    device = data["node"].x_semantic.device if hasattr(data["node"], "x_semantic") else "cpu"

    if not hasattr(data["node"], "batch") or data["node"].batch is None:
        data["node"].batch = torch.zeros(n, dtype=torch.long, device=device)

    if not hasattr(data["node"], "ptr") or data["node"].ptr is None:
        data["node"].ptr = torch.tensor([0, n], dtype=torch.long, device=device)

    return data


# ---------------------------------------------------------------------------
# Coherence: 1 - median(reconstruction_error)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_reconstruction_errors(
    model: RGIN,
    data,  # PyG HeteroData
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """Compute per-node reconstruction error (cosine distance).

    Runs inference with NO masking (all nodes unmasked) — this gives each node
    full structural context, measuring how well structure predicts semantics.

    Returns:
        [N] array of cosine distances ∈ [0, 2]
    """
    data = data.to(device)
    data = _ensure_batched(data)
    n = data["node"].num_nodes

    # No masking — full structural context
    mask = torch.zeros(n, dtype=torch.bool, device=device)

    z_str, z_inv, z_calls, z_imports, z_inherits, g_emb = model(data, mask)

    # Decode: predict semantic embeddings from structural embeddings
    predicted = model.decode(z_str)       # [N, 768]
    actual = data["node"].x_semantic.float()  # [N, 768]

    # Per-node cosine distance: 1 - cos_sim
    cos_sim = F.cosine_similarity(predicted, actual, dim=-1)  # [N]
    reconstruction_error = (1.0 - cos_sim).cpu().numpy()  # [N], ∈ [0, 2]

    return reconstruction_error


def compute_coherence(reconstruction_errors: np.ndarray) -> tuple[float, float]:
    """Compute coherence sub-score.

    coherence = clamp(1 - median(reconstruction_error), 0, 1)

    Returns:
        (coherence, median_error)
    """
    if len(reconstruction_errors) == 0:
        return 1.0, 0.0  # Vacuously correct

    # Guard against NaN from corrupted model output
    if np.any(np.isnan(reconstruction_errors)):
        n_nan = int(np.sum(np.isnan(reconstruction_errors)))
        warnings.warn(
            f"reconstruction_error contains {n_nan}/{len(reconstruction_errors)} NaN values; "
            "filtering them out. This may indicate a corrupted model checkpoint."
        )
        reconstruction_errors = reconstruction_errors[~np.isnan(reconstruction_errors)]
        if len(reconstruction_errors) == 0:
            return 0.0, -1.0  # Sentinel; all NaN means no usable signal

    median_error = float(np.median(reconstruction_errors))
    coherence = max(0.0, min(1.0, 1.0 - median_error))
    return coherence, median_error


# ---------------------------------------------------------------------------
# Flow: cycle_freedom × layer_conformance
# ---------------------------------------------------------------------------

def compute_cycle_freedom(graph: dict) -> tuple[float, int, int]:
    """Compute cycle_freedom = 1 - (nodes_in_nontrivial_SCCs / total_nodes).

    Uses Tarjan's SCC on the directed graph from graph.json.

    Args:
        graph: parsed graph.json with "nodes" and "edges"

    Returns:
        (cycle_freedom, scc_nodes, total_nodes)
    """
    nodes = graph["nodes"]
    edges = graph["edges"]
    n = len(nodes)

    if n == 0:
        return 1.0, 0, 0

    # Build node ID → index mapping
    node_ids = [node["id"] for node in nodes]
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    # Build adjacency list (all directed edges)
    successors = [[] for _ in range(n)]
    for edge in edges:
        src = id_to_idx.get(edge["source"])
        tgt = id_to_idx.get(edge["target"])
        if src is not None and tgt is not None:
            successors[src].append(tgt)

    # Tarjan's SCC (iterative)
    sccs = _tarjan_scc(successors, n)

    # Count nodes in nontrivial SCCs (size >= 2)
    scc_nodes = sum(len(scc) for scc in sccs if len(scc) >= 2)

    cycle_freedom = 1.0 - (scc_nodes / max(n, 1))
    return cycle_freedom, scc_nodes, n


def _tarjan_scc(successors: list[list[int]], n: int) -> list[list[int]]:
    """Tarjan's iterative SCC algorithm. Mirrors the Rust implementation."""
    index_counter = 0
    index = [None] * n  # None = unvisited
    lowlink = [0] * n
    on_stack = [False] * n
    stack = []
    result = []

    for start in range(n):
        if index[start] is not None:
            continue

        index[start] = index_counter
        lowlink[start] = index_counter
        index_counter += 1
        stack.append(start)
        on_stack[start] = True
        work = [(start, 0)]

        while work:
            v, ni = work[-1]
            neighbors = successors[v]

            if ni < len(neighbors):
                work[-1] = (v, ni + 1)
                w = neighbors[ni]
                if index[w] is None:
                    index[w] = index_counter
                    lowlink[w] = index_counter
                    index_counter += 1
                    stack.append(w)
                    on_stack[w] = True
                    work.append((w, 0))
                elif on_stack[w]:
                    lowlink[v] = min(lowlink[v], index[w])
            else:
                if lowlink[v] == index[v]:
                    component = []
                    while True:
                        w = stack.pop()
                        on_stack[w] = False
                        component.append(w)
                        if w == v:
                            break
                    result.append(component)
                work.pop()
                if work:
                    parent = work[-1][0]
                    lowlink[parent] = min(lowlink[parent], lowlink[v])

    return result


@torch.no_grad()
def compute_layer_conformance(
    model: RGIN,
    data,
    z_imports: Tensor,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Compute Phase 3 layer_conformance using direction_surprise.

    layer_conformance = 1 - (Σ_violating max(ds, 0)) / max(Σ_all |ds|, ε)

    Where direction_surprise(u,v) = σ(z_imports(v)^T R z_imports(u)) - σ(z_imports(u)^T R z_imports(v))

    Simplification: we treat all edges with positive direction_surprise as
    violating. The full spec uses a semantic depth probe for layer assignment
    first. This simplification is valid when R has strong asymmetry.
    When R is near-symmetric (asymmetry < 0.1), direction_surprise contains
    no directional information and we fall back to 1.0 (per spec recommendation).
    """
    R = model.R.detach()  # [32, 32]

    # Check R asymmetry — if near-symmetric, direction_surprise is unreliable
    R_np = R.cpu().numpy()
    r_norm = float(np.linalg.norm(R_np))
    r_asym = float(np.linalg.norm(R_np - R_np.T) / r_norm) if r_norm > 1e-8 else 0.0
    if r_asym < R_ASYMMETRY_THRESHOLD:
        warnings.warn(
            f"R asymmetry {r_asym:.4f} < {R_ASYMMETRY_THRESHOLD}: "
            "direction_surprise unreliable, falling back to layer_conformance=1.0"
        )
        return 1.0

    # Get call edges
    call_key = ("node", "calls", "node")
    if call_key not in data.edge_types:
        return 1.0  # No call edges = perfect conformance
    edge_index = data[call_key].edge_index
    if edge_index.shape[1] == 0:
        return 1.0

    z_imp = z_imports.to(device)
    R = R.to(device)

    src = edge_index[0]  # [E]
    tgt = edge_index[1]  # [E]

    z_u = z_imp[src]  # [E, 32]
    z_v = z_imp[tgt]  # [E, 32]

    # Forward direction score: σ(z_u^T R z_v)
    forward_logits = (z_u @ R * z_v).sum(dim=-1)  # [E]
    forward_score = torch.sigmoid(forward_logits)

    # Reverse direction score: σ(z_v^T R z_u)
    reverse_logits = (z_v @ R * z_u).sum(dim=-1)  # [E]
    reverse_score = torch.sigmoid(reverse_logits)

    # direction_surprise = reverse_score - forward_score
    # Positive = model thinks reverse is more likely = against learned flow
    direction_surprise = (reverse_score - forward_score).cpu().numpy()  # [E]

    # Violating edges: direction_surprise > 0
    violating_sum = float(np.sum(np.maximum(direction_surprise, 0)))
    total_abs_sum = float(np.sum(np.abs(direction_surprise)))

    eps = 1e-8
    layer_conformance = 1.0 - (violating_sum / max(total_abs_sum, eps))

    return max(0.0, min(1.0, layer_conformance))


def compute_flow(
    cycle_freedom: float,
    layer_conformance: float,
) -> float:
    """flow = cycle_freedom × layer_conformance. Both ∈ [0, 1]."""
    return cycle_freedom * layer_conformance


# ---------------------------------------------------------------------------
# THS: the combined score
# ---------------------------------------------------------------------------

def compute_ths(coherence: float, flow: float, alpha: float = 0.7) -> float:
    """THS = coherence^α × flow^(1-α).

    Geometric mean with α weighting. Zero in either dimension tanks the score.
    """
    # Guard against NaN from 0^0 or negative^fractional
    if coherence <= 0.0 or flow <= 0.0:
        return 0.0
    return (coherence ** alpha) * (flow ** (1.0 - alpha))


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_health(
    model: RGIN,
    data,  # PyG HeteroData
    graph: dict,  # Parsed graph.json
    alpha: float = 0.7,
    device: torch.device = torch.device("cpu"),
) -> HealthScore:
    """Compute the full Topo Health Score for a single codebase.

    Args:
        model: Trained R-GIN model (eval mode)
        data: PyG HeteroData with node features and edge indices
        graph: Parsed graph.json with "nodes" and "edges"
        alpha: Coherence weight (default 0.7)
        device: torch device

    Returns:
        HealthScore with all sub-components
    """
    data = data.to(device)
    data = _ensure_batched(data)
    n = data["node"].num_nodes

    # Empty graph edge case
    if n == 0:
        return HealthScore(
            topo_health_score=1.0, coherence=1.0, flow=1.0,
            cycle_freedom=1.0, layer_conformance=1.0,
            n_nodes=0, n_edges=0,
            median_reconstruction_error=0.0, r_asymmetry=0.0, alpha=alpha,
        )

    # --- Run model inference (unmasked) ---
    mask = torch.zeros(n, dtype=torch.bool, device=device)
    z_str, z_inv, z_calls, z_imports, z_inherits, g_emb = model(data, mask)

    # --- Coherence ---
    predicted = model.decode(z_str)
    actual = data["node"].x_semantic.float()
    cos_sim = F.cosine_similarity(predicted, actual, dim=-1)
    reconstruction_errors = (1.0 - cos_sim).cpu().numpy()
    coherence, median_error = compute_coherence(reconstruction_errors)

    # --- Flow: cycle_freedom ---
    cycle_freedom, scc_nodes, total_nodes = compute_cycle_freedom(graph)

    # --- Flow: layer_conformance ---
    layer_conformance = compute_layer_conformance(
        model, data, z_imports, device=device,
    )

    # --- Flow ---
    flow = compute_flow(cycle_freedom, layer_conformance)

    # --- THS ---
    ths = compute_ths(coherence, flow, alpha)

    # --- R asymmetry (diagnostic) ---
    R_np = model.R.detach().cpu().numpy()
    r_norm = float(np.linalg.norm(R_np))
    r_asymmetry = float(np.linalg.norm(R_np - R_np.T) / r_norm) if r_norm > 1e-8 else 0.0

    # Count total edges (robust against missing edge stores)
    n_edges = 0
    for rel in ["calls", "imports", "inherits"]:
        key = ("node", rel, "node")
        if key in data.edge_types and hasattr(data[key], "edge_index"):
            n_edges += data[key].edge_index.shape[1]

    return HealthScore(
        topo_health_score=round(ths, 4),
        coherence=round(coherence, 4),
        flow=round(flow, 4),
        cycle_freedom=round(cycle_freedom, 4),
        layer_conformance=round(layer_conformance, 4),
        n_nodes=n,
        n_edges=n_edges,
        median_reconstruction_error=round(median_error, 4),
        r_asymmetry=round(r_asymmetry, 4),
        alpha=alpha,
    )
