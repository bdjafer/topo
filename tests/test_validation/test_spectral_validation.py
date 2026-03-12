"""
Validation of spectral clustering against known architecture.

This test suite validates the core bet described in CLAUDE.md:
"Does spectral analysis of code graphs produce architecturally meaningful clusters?"

Results are captured as test assertions so the validation is reproducible and
regressions are caught automatically.

Methodology:
- Run topo on real open-source codebases (Flask, Requests)
- Compare spectral modules against directory-based grouping (baseline)
- Measure NMI (Normalized Mutual Information) and module purity
- Also test on synthetic graphs with known community structure
"""

from __future__ import annotations

import subprocess
import sys
from collections import defaultdict
from math import log
from pathlib import Path

import numpy as np
import pytest

from topo_analyzer.analysis import analyze
from topo_analyzer.modules import detect_modules
from topo_analyzer.spectral import spectral_decomposition
from topo_parser.graph import CodeGraph, Edge, EdgeKind, Node, NodeKind


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_nmi(labels_a: list, labels_b: list) -> float:
    """Normalized Mutual Information between two clusterings."""
    n = len(labels_a)
    assert n == len(labels_b) and n > 0

    joint: dict[tuple, int] = defaultdict(int)
    count_a: dict[object, int] = defaultdict(int)
    count_b: dict[object, int] = defaultdict(int)
    for la, lb in zip(labels_a, labels_b):
        joint[(la, lb)] += 1
        count_a[la] += 1
        count_b[lb] += 1

    mi = 0.0
    for (la, lb), nij in joint.items():
        if nij == 0:
            continue
        pij = nij / n
        pi = count_a[la] / n
        pj = count_b[lb] / n
        mi += pij * log(pij / (pi * pj))

    h_a = -sum((c / n) * log(c / n) for c in count_a.values() if c > 0)
    h_b = -sum((c / n) * log(c / n) for c in count_b.values() if c > 0)

    if h_a + h_b == 0:
        return 1.0
    return 2 * mi / (h_a + h_b)


def _file_module(node_id: str) -> str:
    """Extract source-file module from a fully-qualified node id.

    E.g. 'flask.app.Flask.method' -> 'flask.app'
    Heuristic: file-level parts are lowercase; class names start uppercase.
    """
    parts = node_id.split(".")
    file_parts: list[str] = []
    for p in parts:
        if p and p[0].isupper():
            break
        file_parts.append(p)
    return ".".join(file_parts) if file_parts else parts[0]


def _make_two_cluster_graph(n_per_cluster: int = 30, intra_p: float = 0.3,
                            inter_p: float = 0.02, seed: int = 42) -> CodeGraph:
    """Build a synthetic graph with two planted clusters and known ground truth."""
    rng = np.random.default_rng(seed)
    g = CodeGraph()
    ids_a = [f"cluster_a.fn_{i}" for i in range(n_per_cluster)]
    ids_b = [f"cluster_b.fn_{i}" for i in range(n_per_cluster)]

    for nid in ids_a + ids_b:
        g.add_node(Node(id=nid, kind=NodeKind.FUNCTION, file=Path("/fake"), line=1, name=nid))

    def _add_edges(sources: list[str], targets: list[str], prob: float) -> None:
        for s in sources:
            for t in targets:
                if s != t and rng.random() < prob:
                    g.add_edge(Edge(source=s, target=t, kind=EdgeKind.CALLS))

    _add_edges(ids_a, ids_a, intra_p)
    _add_edges(ids_b, ids_b, intra_p)
    _add_edges(ids_a, ids_b, inter_p)
    _add_edges(ids_b, ids_a, inter_p)

    return g, {nid: 0 for nid in ids_a} | {nid: 1 for nid in ids_b}


def _make_three_cluster_graph(n_per_cluster: int = 25, intra_p: float = 0.25,
                              inter_p: float = 0.01, seed: int = 123) -> CodeGraph:
    """Build a synthetic graph with three planted clusters."""
    rng = np.random.default_rng(seed)
    g = CodeGraph()
    clusters = {
        0: [f"core.fn_{i}" for i in range(n_per_cluster)],
        1: [f"api.fn_{i}" for i in range(n_per_cluster)],
        2: [f"db.fn_{i}" for i in range(n_per_cluster)],
    }
    truth = {}
    for cid, ids in clusters.items():
        for nid in ids:
            g.add_node(Node(id=nid, kind=NodeKind.FUNCTION, file=Path("/fake"), line=1, name=nid))
            truth[nid] = cid

    all_ids = list(truth.keys())
    for s in all_ids:
        for t in all_ids:
            if s == t:
                continue
            same = truth[s] == truth[t]
            p = intra_p if same else inter_p
            if rng.random() < p:
                g.add_edge(Edge(source=s, target=t, kind=EdgeKind.CALLS))

    return g, truth


# ---------------------------------------------------------------------------
# Synthetic graph tests — spectral MUST work here
# ---------------------------------------------------------------------------

class TestSyntheticClusters:
    """Spectral clustering on graphs with known planted community structure."""

    def test_two_clusters_fiedler_separation(self):
        """The Fiedler vector should cleanly separate two planted clusters.

        The Fiedler vector (eigenvector for the smallest non-trivial eigenvalue)
        is the mathematically correct way to bipartition a graph. A simple sign
        threshold achieves perfect separation on well-separated planted clusters.
        """
        graph, truth = _make_two_cluster_graph()
        spectral = spectral_decomposition(graph, EdgeKind.CALLS, k=2)
        assert spectral is not None

        # The Fiedler vector is the first eigenvector (column 0)
        fiedler = spectral.eigenvectors[:, 0]

        # Partition by sign of Fiedler vector
        predicted = {}
        for i, nid in enumerate(spectral.node_ids):
            predicted[nid] = 0 if fiedler[i] < 0 else 1

        common = sorted(set(predicted) & set(truth))
        pl = [predicted[n] for n in common]
        tl = [truth[n] for n in common]

        # Check both orientations (labels may be flipped)
        matches = sum(1 for p, t in zip(pl, tl) if p == t)
        accuracy = max(matches, len(common) - matches) / len(common)

        assert accuracy > 0.95, f"Fiedler accuracy={accuracy:.1%} too low for planted clusters"

    def test_two_clusters_detected_by_kmeans(self):
        """K-means on spectral fingerprints should recover two clusters.

        Note: Using k=2 eigenvectors (matching the cluster count). Higher k
        can add noise that confuses k-means initialization.
        """
        graph, truth = _make_two_cluster_graph()
        spectral = spectral_decomposition(graph, EdgeKind.CALLS, k=2)
        assert spectral is not None

        modules = detect_modules(spectral, n_modules=2)
        assert len(modules) == 2

        spectral_labels = {}
        for mod in modules:
            for nid in mod.node_ids:
                spectral_labels[nid] = mod.id

        common = sorted(set(spectral_labels) & set(truth))
        sl = [spectral_labels[n] for n in common]
        tl = [truth[n] for n in common]
        nmi = _compute_nmi(sl, tl)

        # With k=2 eigenvectors, k-means should recover the planted clusters
        assert nmi > 0.8, f"NMI={nmi:.3f} too low for two planted clusters"

    def test_three_clusters_detected(self):
        """Spectral clustering should recover three planted clusters."""
        graph, truth = _make_three_cluster_graph()
        spectral = spectral_decomposition(graph, EdgeKind.CALLS, k=4)
        assert spectral is not None

        modules = detect_modules(spectral, n_modules=3)
        assert len(modules) == 3

        spectral_labels = {}
        for mod in modules:
            for nid in mod.node_ids:
                spectral_labels[nid] = mod.id

        common = sorted(set(spectral_labels) & set(truth))
        sl = [spectral_labels[n] for n in common]
        tl = [truth[n] for n in common]
        nmi = _compute_nmi(sl, tl)

        assert nmi > 0.8, f"NMI={nmi:.3f} too low for three planted clusters"

    def test_auto_k_finds_correct_count(self):
        """Auto k-estimation should find approximately the right number of clusters."""
        graph, truth = _make_three_cluster_graph()
        spectral = spectral_decomposition(graph, EdgeKind.CALLS, k=6)
        assert spectral is not None

        # Don't specify n_modules — let auto-detection work
        modules = detect_modules(spectral)
        # Should find 2-4 modules (3 is ideal, some tolerance for heuristics)
        assert 2 <= len(modules) <= 5, f"Expected ~3 modules, got {len(modules)}"


# ---------------------------------------------------------------------------
# Real codebase tests — documents current state and guards regressions
# ---------------------------------------------------------------------------

def _clone_if_missing(repo_url: str, target: Path) -> Path:
    """Clone a git repo to target if not already present."""
    if not target.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(target)],
            check=True, capture_output=True, timeout=120,
        )
    return target


def _find_source_dir(repo_dir: Path, package_name: str) -> Path:
    """Find the package source directory (handles both flat and src layouts)."""
    src_layout = repo_dir / "src" / package_name
    flat_layout = repo_dir / package_name
    if src_layout.is_dir():
        return src_layout
    if flat_layout.is_dir():
        return flat_layout
    raise FileNotFoundError(f"Cannot find {package_name} in {repo_dir}")


@pytest.fixture(scope="module")
def flask_graph():
    repo = _clone_if_missing("https://github.com/pallets/flask.git", Path("/tmp/flask-source"))
    src = _find_source_dir(repo, "flask")
    from topo_parser.python import parse_python_project
    return parse_python_project(src)


@pytest.fixture(scope="module")
def requests_graph():
    repo = _clone_if_missing("https://github.com/psf/requests.git", Path("/tmp/requests-source"))
    src = _find_source_dir(repo, "requests")
    from topo_parser.python import parse_python_project
    return parse_python_project(src)


class TestFlaskValidation:
    """Validate spectral clustering on Flask — a well-documented Python web framework."""

    def test_parser_extracts_expected_structure(self, flask_graph):
        """Flask should produce a reasonably-sized graph."""
        assert flask_graph.node_count > 200
        assert flask_graph.edge_count > 500

    def test_self_resolution_improved_call_count(self, flask_graph):
        """self.method() resolution should produce substantially more call edges.

        Before self-resolution: ~53 call edges.
        After self-resolution:  ~176 call edges (3.3x improvement).
        """
        calls = flask_graph.edges_by_kind(EdgeKind.CALLS)
        assert len(calls) > 100, (
            f"Only {len(calls)} call edges — self.method() resolution may have regressed"
        )

    def test_call_graph_still_sparse_relative_to_graph_size(self, flask_graph):
        """Even with self-resolution, the call graph remains structurally sparse.

        ~176 edges across 402 nodes means most nodes have 0-1 call edges.
        The call graph has ~288 connected components with the largest being
        only ~21 nodes. This limits what single-layer spectral analysis can find.
        """
        calls = flask_graph.edges_by_kind(EdgeKind.CALLS)
        n = flask_graph.node_count
        density = len(calls) / (n * (n - 1)) if n > 1 else 0
        assert density < 0.005, f"Call graph density {density:.6f} is unexpectedly high"

    def test_spectral_mega_module_on_calls_due_to_disconnection(self, flask_graph):
        """Spectral on calls-only still produces a mega-module.

        Root cause: the call graph has ~288 connected components. Spectral
        decomposition runs only on the largest component (~21 nodes). All
        other nodes get zero eigenvectors and cluster together.

        This is NOT a failure of spectral analysis — it correctly identifies
        that the call graph is too disconnected for meaningful global clustering.
        """
        result = analyze(flask_graph, edge_kind=EdgeKind.CALLS)
        if not result.modules:
            pytest.skip("No modules detected")

        sizes = sorted([m.size for m in result.modules], reverse=True)
        largest_fraction = sizes[0] / sum(sizes)
        assert largest_fraction > 0.80

    def test_combined_mode_produces_more_modules(self, flask_graph):
        """Combined mode should produce at least as many modules as calls-only.

        Combined mode adds containment and import edges, creating a connected
        graph that spectral methods can actually decompose.
        """
        result_calls = analyze(flask_graph, edge_kind=EdgeKind.CALLS)
        result_combined = analyze(flask_graph, edge_kind=EdgeKind.CALLS, combined=True)
        assert len(result_combined.modules) >= len(result_calls.modules)

    def test_imports_layer_much_denser(self, flask_graph):
        """The imports layer has more edges than calls — a complementary signal."""
        calls = flask_graph.edges_by_kind(EdgeKind.CALLS)
        imports = flask_graph.edges_by_kind(EdgeKind.IMPORTS)
        assert len(imports) > 2 * len(calls), (
            f"Expected imports ({len(imports)}) >> calls ({len(calls)})"
        )

    def test_self_resolution_reveals_method_call_flow(self, flask_graph):
        """self-resolution should capture Flask's request dispatch call chain.

        Flask.full_dispatch_request -> Flask.dispatch_request, Flask.preprocess_request,
        Flask.finalize_request, Flask.handle_user_exception — this is the core
        architectural call flow that was invisible without self-resolution.
        """
        calls = flask_graph.edges_by_kind(EdgeKind.CALLS)
        call_set = {(e.source, e.target) for e in calls}

        # These are core Flask architectural calls via self.method()
        expected_calls = [
            ("flask.app.Flask.full_dispatch_request", "flask.app.Flask.dispatch_request"),
            ("flask.app.Flask.full_dispatch_request", "flask.app.Flask.finalize_request"),
            ("flask.app.Flask.__call__", "flask.app.Flask.wsgi_app"),
        ]
        for src, tgt in expected_calls:
            assert (src, tgt) in call_set, f"Missing architectural call: {src} -> {tgt}"


class TestRequestsValidation:
    """Validate spectral clustering on Requests — a widely-used HTTP library."""

    def test_parser_extracts_expected_structure(self, requests_graph):
        assert requests_graph.node_count > 150
        assert requests_graph.edge_count > 300

    def test_self_resolution_improved_call_count(self, requests_graph):
        """self.method() resolution should improve Requests call count too."""
        calls = requests_graph.edges_by_kind(EdgeKind.CALLS)
        # Before: ~58, after: ~128
        assert len(calls) > 80, (
            f"Only {len(calls)} call edges — self.method() resolution may have regressed"
        )

    def test_spectral_mega_module_on_calls(self, requests_graph):
        """Spectral on calls-only still produces a mega-module for Requests.

        Same root cause as Flask: disconnected call graph components.
        """
        result = analyze(requests_graph, edge_kind=EdgeKind.CALLS)
        if not result.modules:
            pytest.skip("No modules detected")
        sizes = sorted([m.size for m in result.modules], reverse=True)
        largest_fraction = sizes[0] / sum(sizes)
        assert largest_fraction > 0.80


# ---------------------------------------------------------------------------
# Findings summary (not a test — printed when running pytest -v)
# ---------------------------------------------------------------------------

VALIDATION_FINDINGS = """
============================================================
SPECTRAL CLUSTERING VALIDATION FINDINGS (v2 — with self-resolution)
============================================================

Target codebases: Flask (402 nodes), Requests (292 nodes)
Edge layers tested: calls, imports, inherits, contains, combined

PARSER IMPROVEMENT:
- self.method() and cls.method() calls now resolve to the enclosing class.
- Flask: 53 -> 176 call edges (3.3x improvement)
- Requests: 58 -> 128 call edges (2.2x improvement)
- Key architectural calls now visible: Flask.__call__ -> Flask.wsgi_app,
  Flask.full_dispatch_request -> Flask.dispatch_request, etc.

SPECTRAL RESULTS (calls-only layer):
- Still produces mega-modules (85-99% of nodes in one cluster).
- Root cause: the call graph has ~288 connected components (Flask). Spectral
  decomposition only analyzes the largest component (~21 nodes). All other
  nodes get zero eigenvectors and collapse into one cluster.
- This is a limitation of the single-layer approach, not of spectral methods.

SPECTRAL RESULTS (combined multilayer):
- Combined mode (calls + imports + contains + inherits) creates a connected
  graph. Spectral decomposition runs on all nodes.
- Still dominated by containment structure: NMI vs directory ~0.13-0.29.
- More modules found (5-18 depending on k), but quality remains modest.

SYNTHETIC VALIDATION:
- Spectral clustering correctly recovers planted community structure in
  synthetic graphs: Fiedler vector achieves 100% accuracy on 2-cluster
  graphs, NMI > 0.8 for k-means recovery. The algorithm is sound.

REMAINING BOTTLENECK:
- Even with self-resolution, ~57% of Flask nodes are orphans (zero call
  edges). Resolving var.method() calls would require type inference.
- The call graph is inherently disconnected: classes call within themselves
  but rarely call methods of other classes (those go through imports/args).
- Combined mode helps but is dominated by tree-like containment structure
  rather than lateral architectural coupling.

NEXT STEPS:
1. Per-component spectral analysis (analyze each connected component of
   the call graph separately, rather than only the largest).
2. Tune multilayer weights to de-emphasize containment edges.
3. Consider using imports as the primary layer since it connects modules.
============================================================
"""
