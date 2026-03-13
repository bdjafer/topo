#!/usr/bin/env python3
"""Generate V1 benchmark datasets from existing test fixtures.

Run once:  uv run python scripts/generate_v1_datasets.py
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from topo_parser.graph import CodeGraph, Edge, EdgeKind, Node, NodeKind
from topo_parser.python import parse_python_project
from topo_benchmark.codegraph_io import save_graph, serialize_graph

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "validation"
DATASET_ROOT = REPO_ROOT / "benchmark" / "datasets"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def _copy_graph(graph: CodeGraph) -> CodeGraph:
    """Deep copy a CodeGraph."""
    new = CodeGraph()
    for node in graph.nodes.values():
        new.add_node(node)
    for edge in graph.edges:
        new.add_edge(edge)
    return new


def _add_edge(graph: CodeGraph, source: str, target: str, kind: EdgeKind) -> CodeGraph:
    """Return a copy with one edge added."""
    g = _copy_graph(graph)
    g.add_edge(Edge(source=source, target=target, kind=kind))
    return g


def _add_edges(graph: CodeGraph, edges: list[tuple[str, str, EdgeKind]]) -> CodeGraph:
    """Return a copy with multiple edges added."""
    g = _copy_graph(graph)
    for src, tgt, kind in edges:
        g.add_edge(Edge(source=src, target=tgt, kind=kind))
    return g


def _add_node_and_edges(
    graph: CodeGraph,
    node: Node,
    edges: list[tuple[str, str, EdgeKind]],
) -> CodeGraph:
    """Return a copy with a new node and edges."""
    g = _copy_graph(graph)
    g.add_node(node)
    for src, tgt, kind in edges:
        g.add_edge(Edge(source=src, target=tgt, kind=kind))
    return g


def _remove_edges_matching(
    graph: CodeGraph,
    source: str | None = None,
    target: str | None = None,
) -> CodeGraph:
    """Return a copy without edges matching the given source/target."""
    g = CodeGraph()
    for node in graph.nodes.values():
        g.add_node(node)
    for edge in graph.edges:
        if source and edge.source == source:
            continue
        if target and edge.target == target:
            continue
        g.add_edge(edge)
    return g


# ---------------------------------------------------------------------------
# Parse base fixture
# ---------------------------------------------------------------------------

def parse_fixture(name: str) -> CodeGraph:
    return parse_python_project(FIXTURE_ROOT / name)


# ---------------------------------------------------------------------------
# Mutation cases
# ---------------------------------------------------------------------------

def generate_mutations(clean: CodeGraph) -> None:
    """Generate all 12 mutation cases from the clean layered_app graph."""
    out = DATASET_ROOT / "mutations"

    # Node IDs from layered_app
    # api layer: api.routes.submit_order, api.routes.get_order, api.serializers.serialize_order
    # core layer: core.service.create_order, core.service.fetch_order, core.rules.normalize_order
    # data layer: data.store.save_order, data.store.load_order, data.audit.record_event

    # --- revdep_light ---
    # Signal: cross_package_dep_count increases (data->api appears)
    case_dir = out / "revdep_light"
    mutated = _add_edge(clean, "data.store.load_order", "api.serializers.serialize_order", EdgeKind.CALLS)
    repaired = _copy_graph(clean)
    save_graph(clean, case_dir / "variants" / "clean.json")
    save_graph(mutated, case_dir / "variants" / "mutated.json")
    save_graph(repaired, case_dir / "variants" / "repaired.json")
    _write_json(case_dir / "expectations.json", {
        "ordering": [["clean", "mutated"], ["repaired", "mutated"]],
        "required_expectations": [
            {"variants": ["clean", "mutated"], "signal": "cross_package_dep_count", "direction": "higher_in_second"},
            {"variants": ["clean", "mutated"], "signal": "has_cross_package_dep", "signal_args": {"source_pkg": "data", "target_pkg": "api"}, "direction": "true_in_second"},
        ],
        "mutated_region": {"nodes": ["data.store", "api.serializers"]},
    })
    _write_json(case_dir / "metadata.json", {"split": "public", "level": "module", "description": "Single reverse calls edge across module boundary"})

    # --- revdep_balanced ---
    case_dir = out / "revdep_balanced"
    mutated = _add_edges(clean, [
        ("data.store.load_order", "api.serializers.serialize_order", EdgeKind.CALLS),
        ("data.store.save_order", "api.routes.submit_order", EdgeKind.CALLS),
        ("data.audit.record_event", "api.routes.get_order", EdgeKind.CALLS),
    ])
    save_graph(clean, case_dir / "variants" / "clean.json")
    save_graph(mutated, case_dir / "variants" / "mutated.json")
    _write_json(case_dir / "expectations.json", {
        "ordering": [["clean", "mutated"]],
        "required_expectations": [
            {"variants": ["clean", "mutated"], "signal": "cross_package_dep_count", "direction": "higher_in_second"},
            {"variants": ["clean", "mutated"], "signal": "has_cross_package_dep", "signal_args": {"source_pkg": "data", "target_pkg": "api"}, "direction": "true_in_second"},
        ],
        "mutated_region": {"nodes": ["data.store", "data.audit", "api.serializers", "api.routes"]},
    })
    _write_json(case_dir / "metadata.json", {"split": "public", "level": "module", "description": "Multiple reverse edges making boundary nearly bidirectional"})

    # --- revdep_multilayer ---
    case_dir = out / "revdep_multilayer"
    mutated = _add_edges(clean, [
        ("data.store.load_order", "api.serializers.serialize_order", EdgeKind.CALLS),
        ("data.store.load_order", "api.serializers.serialize_order", EdgeKind.IMPORTS),
    ])
    repaired = _copy_graph(clean)
    save_graph(clean, case_dir / "variants" / "clean.json")
    save_graph(mutated, case_dir / "variants" / "mutated.json")
    save_graph(repaired, case_dir / "variants" / "repaired.json")
    _write_json(case_dir / "expectations.json", {
        "ordering": [["clean", "mutated"], ["repaired", "mutated"]],
        "required_expectations": [
            {"variants": ["clean", "mutated"], "signal": "cross_package_dep_count", "direction": "higher_in_second"},
            {"variants": ["clean", "mutated"], "signal": "has_cross_package_dep", "signal_args": {"source_pkg": "data", "target_pkg": "api"}, "direction": "true_in_second"},
        ],
        "mutated_region": {"nodes": ["data.store", "api.serializers"]},
    })
    _write_json(case_dir / "metadata.json", {"split": "public", "level": "module", "description": "Reverse calls + imports edges across same boundary"})

    # --- cycle_two_module ---
    # Signal: cycle_member finding appears, cross_package_dep for core->api appears
    case_dir = out / "cycle_two_module"
    mutated = _add_edge(clean, "core.service.create_order", "api.routes.submit_order", EdgeKind.CALLS)
    repaired = _copy_graph(clean)
    save_graph(clean, case_dir / "variants" / "clean.json")
    save_graph(mutated, case_dir / "variants" / "mutated.json")
    save_graph(repaired, case_dir / "variants" / "repaired.json")
    _write_json(case_dir / "expectations.json", {
        "ordering": [["clean", "mutated"], ["repaired", "mutated"]],
        "required_expectations": [
            {"variants": ["clean", "mutated"], "signal": "has_finding", "signal_args": {"kind": "cycle_member"}, "direction": "present_in_second_not_first"},
            {"variants": ["clean", "mutated"], "signal": "cross_package_dep_count", "direction": "higher_in_second"},
        ],
        "mutated_region": {"nodes": ["core.service", "api.routes"]},
    })
    _write_json(case_dir / "metadata.json", {"split": "public", "level": "module", "description": "Back edge closing a 2-module SCC"})

    # --- cycle_three_module_ring ---
    case_dir = out / "cycle_three_module_ring"
    mutated = _add_edge(clean, "data.store.save_order", "api.routes.submit_order", EdgeKind.CALLS)
    save_graph(clean, case_dir / "variants" / "clean.json")
    save_graph(mutated, case_dir / "variants" / "mutated.json")
    _write_json(case_dir / "expectations.json", {
        "ordering": [["clean", "mutated"]],
        "required_expectations": [
            {"variants": ["clean", "mutated"], "signal": "has_finding", "signal_args": {"kind": "cycle_member"}, "direction": "present_in_second_not_first"},
            {"variants": ["clean", "mutated"], "signal": "cross_package_dep_count", "direction": "higher_in_second"},
        ],
        "mutated_region": {"nodes": ["data.store", "api.routes"]},
    })
    _write_json(case_dir / "metadata.json", {"split": "public", "level": "module", "description": "data→api edge closing a 3-module ring (api→core→data→api)"})

    # --- cycle_multilayer ---
    case_dir = out / "cycle_multilayer"
    mutated = _add_edges(clean, [
        ("data.store.load_order", "core.service.fetch_order", EdgeKind.IMPORTS),
        ("core.rules.normalize_order", "api.routes.get_order", EdgeKind.CALLS),
    ])
    repaired = _copy_graph(clean)
    save_graph(clean, case_dir / "variants" / "clean.json")
    save_graph(mutated, case_dir / "variants" / "mutated.json")
    save_graph(repaired, case_dir / "variants" / "repaired.json")
    _write_json(case_dir / "expectations.json", {
        "ordering": [["clean", "mutated"], ["repaired", "mutated"]],
        "required_expectations": [
            {"variants": ["clean", "mutated"], "signal": "cross_package_dep_count", "direction": "higher_in_second"},
        ],
        "mutated_region": {"nodes": ["data.store", "core.service", "core.rules", "api.routes"]},
    })
    _write_json(case_dir / "metadata.json", {"split": "public", "level": "module", "description": "Cycle visible only when calls+imports are combined"})

    # --- bridge_connector ---
    # Signal: largest_module_ratio increases, module_separation finding appears
    case_dir = out / "bridge_connector"
    connector = Node(id="bridge.connector.relay", kind=NodeKind.FUNCTION, file=Path("bridge/connector.py"), line=1, name="relay")
    mutated = _add_node_and_edges(clean, connector, [
        ("api.routes.submit_order", "bridge.connector.relay", EdgeKind.CALLS),
        ("bridge.connector.relay", "data.store.save_order", EdgeKind.CALLS),
        ("api.routes.get_order", "bridge.connector.relay", EdgeKind.CALLS),
        ("bridge.connector.relay", "data.store.load_order", EdgeKind.CALLS),
    ])
    repaired = _copy_graph(clean)
    save_graph(clean, case_dir / "variants" / "clean.json")
    save_graph(mutated, case_dir / "variants" / "mutated.json")
    save_graph(repaired, case_dir / "variants" / "repaired.json")
    _write_json(case_dir / "expectations.json", {
        "ordering": [["clean", "mutated"], ["repaired", "mutated"]],
        "required_expectations": [
            {"variants": ["clean", "mutated"], "signal": "largest_module_ratio", "direction": "higher_in_second", "margin": 0.01},
            {"variants": ["clean", "mutated"], "signal": "cross_package_dep_count", "direction": "higher_in_second"},
        ],
        "mutated_region": {"nodes": ["bridge.connector"]},
    })
    _write_json(case_dir / "metadata.json", {"split": "public", "level": "module", "description": "Connector node bridging two separate regions"})

    # --- bridge_hub_escalation ---
    case_dir = out / "bridge_hub_escalation"
    connector = Node(id="bridge.connector.relay", kind=NodeKind.FUNCTION, file=Path("bridge/connector.py"), line=1, name="relay")
    mutated = _add_node_and_edges(clean, connector, [
        ("api.routes.submit_order", "bridge.connector.relay", EdgeKind.CALLS),
        ("api.routes.get_order", "bridge.connector.relay", EdgeKind.CALLS),
        ("bridge.connector.relay", "data.store.save_order", EdgeKind.CALLS),
        ("bridge.connector.relay", "data.store.load_order", EdgeKind.CALLS),
        ("bridge.connector.relay", "core.service.create_order", EdgeKind.CALLS),
        ("bridge.connector.relay", "core.rules.normalize_order", EdgeKind.CALLS),
    ])
    save_graph(clean, case_dir / "variants" / "clean.json")
    save_graph(mutated, case_dir / "variants" / "mutated.json")
    _write_json(case_dir / "expectations.json", {
        "ordering": [["clean", "mutated"]],
        "required_expectations": [
            {"variants": ["clean", "mutated"], "signal": "largest_module_ratio", "direction": "higher_in_second", "margin": 0.01},
            {"variants": ["clean", "mutated"], "signal": "cross_package_dep_count", "direction": "higher_in_second"},
        ],
        "mutated_region": {"nodes": ["bridge.connector"]},
    })
    _write_json(case_dir / "metadata.json", {"split": "public", "level": "module", "description": "Connector touching all three regions, becoming a hub"})

    # --- bridge_multilayer ---
    case_dir = out / "bridge_multilayer"
    connector = Node(id="bridge.connector.relay", kind=NodeKind.FUNCTION, file=Path("bridge/connector.py"), line=1, name="relay")
    mutated = _add_node_and_edges(clean, connector, [
        ("api.routes.submit_order", "bridge.connector.relay", EdgeKind.CALLS),
        ("bridge.connector.relay", "data.store.save_order", EdgeKind.IMPORTS),
        ("api.routes.get_order", "bridge.connector.relay", EdgeKind.IMPORTS),
        ("bridge.connector.relay", "data.store.load_order", EdgeKind.CALLS),
    ])
    repaired = _copy_graph(clean)
    save_graph(clean, case_dir / "variants" / "clean.json")
    save_graph(mutated, case_dir / "variants" / "mutated.json")
    save_graph(repaired, case_dir / "variants" / "repaired.json")
    _write_json(case_dir / "expectations.json", {
        "ordering": [["clean", "mutated"], ["repaired", "mutated"]],
        "required_expectations": [
            {"variants": ["clean", "mutated"], "signal": "cross_package_dep_count", "direction": "higher_in_second"},
        ],
        "mutated_region": {"nodes": ["bridge.connector"]},
    })
    _write_json(case_dir / "metadata.json", {"split": "public", "level": "module", "description": "Bridge via mixed calls/imports edges"})

    # --- boundary_erosion_light ---
    # Signal: cross_package_dep_count increases (api->data appears)
    case_dir = out / "boundary_erosion_light"
    mutated = _add_edges(clean, [
        ("api.routes.submit_order", "data.audit.record_event", EdgeKind.CALLS),
        ("api.routes.get_order", "data.store.load_order", EdgeKind.CALLS),
    ])
    repaired = _copy_graph(clean)
    save_graph(clean, case_dir / "variants" / "clean.json")
    save_graph(mutated, case_dir / "variants" / "mutated.json")
    save_graph(repaired, case_dir / "variants" / "repaired.json")
    _write_json(case_dir / "expectations.json", {
        "ordering": [["clean", "mutated"], ["repaired", "mutated"]],
        "required_expectations": [
            {"variants": ["clean", "mutated"], "signal": "cross_package_dep_count", "direction": "higher_in_second"},
        ],
        "mutated_region": {"nodes": ["api.routes", "data.audit", "data.store"]},
    })
    _write_json(case_dir / "metadata.json", {"split": "public", "level": "module", "description": "Sparse cross edges between api and data"})

    # --- boundary_merge_full ---
    case_dir = out / "boundary_merge_full"
    mutated = _add_edges(clean, [
        ("api.routes.submit_order", "data.store.save_order", EdgeKind.CALLS),
        ("api.routes.submit_order", "data.audit.record_event", EdgeKind.CALLS),
        ("api.routes.get_order", "data.store.load_order", EdgeKind.CALLS),
        ("api.routes.get_order", "data.audit.record_event", EdgeKind.CALLS),
        ("api.serializers.serialize_order", "data.store.load_order", EdgeKind.CALLS),
        ("data.store.save_order", "api.serializers.serialize_order", EdgeKind.CALLS),
        ("data.store.load_order", "api.serializers.serialize_order", EdgeKind.CALLS),
        ("data.audit.record_event", "api.routes.submit_order", EdgeKind.CALLS),
    ])
    save_graph(clean, case_dir / "variants" / "clean.json")
    save_graph(mutated, case_dir / "variants" / "mutated.json")
    _write_json(case_dir / "expectations.json", {
        "ordering": [["clean", "mutated"]],
        "required_expectations": [
            {"variants": ["clean", "mutated"], "signal": "cross_package_dep_count", "direction": "higher_in_second"},
        ],
        "mutated_region": {"nodes": ["api.routes", "api.serializers", "data.store", "data.audit"]},
    })
    _write_json(case_dir / "metadata.json", {"split": "public", "level": "module", "description": "Enough cross edges to collapse two clusters"})

    # --- utility_misplaced (symbol-level) ---
    case_dir = out / "utility_misplaced"
    # At symbol level, normalize_order is a utility (leaf-like).
    # Mutation adds cross-module edges making it a bridge.
    mutated = _add_edges(clean, [
        ("api.routes.submit_order", "core.rules.normalize_order", EdgeKind.CALLS),
        ("core.rules.normalize_order", "data.store.save_order", EdgeKind.CALLS),
        ("core.rules.normalize_order", "api.serializers.serialize_order", EdgeKind.CALLS),
    ])
    repaired = _copy_graph(clean)
    save_graph(clean, case_dir / "variants" / "clean.json")
    save_graph(mutated, case_dir / "variants" / "mutated.json")
    save_graph(repaired, case_dir / "variants" / "repaired.json")
    _write_json(case_dir / "expectations.json", {
        "ordering": [["clean", "mutated"], ["repaired", "mutated"]],
        "required_expectations": [
            {"variants": ["clean", "mutated"], "signal": "cross_package_dep_count", "direction": "higher_in_second"},
        ],
        "mutated_region": {"nodes": ["core.rules.normalize_order"]},
    })
    _write_json(case_dir / "metadata.json", {"split": "public", "level": "symbol", "description": "Utility node moved into cross-module flow path"})


# ---------------------------------------------------------------------------
# Architecture recovery dataset
# ---------------------------------------------------------------------------

def generate_architecture(clean: CodeGraph) -> None:
    """Generate architecture recovery gold labels from clean layered_app."""
    out = DATASET_ROOT / "architecture" / "layered_app"

    save_graph(clean, out / "graph.json")

    # Gold labels at MODULE level: use module-level IDs (2-part dotted paths)
    # After projection, nodes are: api.routes, api.serializers, core.service, core.rules, data.store, data.audit
    # Plus package nodes: api, core, data (which get marked unassigned)
    included_nodes = {}
    for node_id in clean.nodes:
        parts = node_id.split(".")
        pkg = parts[0]
        # Only include module-level nodes (2-part paths like api.routes)
        if len(parts) == 2:
            included_nodes[node_id] = pkg

    _write_json(out / "labels.json", {
        "analysis_level": "module",
        "included_nodes": included_nodes,
        "excluded_nodes": [],
    })
    _write_json(out / "metadata.json", {
        "split": "public",
        "description": "Clean layered app with api/core/data packages",
        "label_provenance": "Package structure matches documented architecture",
    })


# ---------------------------------------------------------------------------
# Stability dataset
# ---------------------------------------------------------------------------

def generate_stability(clean: CodeGraph) -> None:
    """Generate stability perturbation cases."""
    out = DATASET_ROOT / "stability" / "layered_app_perturbations"

    save_graph(clean, out / "base_graph.json")

    # Perturbation 1: rename nodes (simulates refactoring)
    renamed = CodeGraph()
    node_mapping: dict[str, str] = {}
    for node in clean.nodes.values():
        # Rename by appending _v2 to function names
        new_name = node.name + "_v2"
        new_id = node.id.rsplit(".", 1)[0] + "." + new_name
        node_mapping[node.id] = new_id
        renamed.add_node(Node(
            id=new_id, kind=node.kind, file=node.file, line=node.line, name=new_name,
        ))
    for edge in clean.edges:
        renamed.add_edge(Edge(
            source=node_mapping[edge.source],
            target=node_mapping[edge.target],
            kind=edge.kind,
        ))
    save_graph(renamed, out / "perturbations" / "rename.json")

    # Perturbation 2: add dead leaf helpers
    with_leaves = _copy_graph(clean)
    for pkg in ["api", "core", "data"]:
        leaf = Node(
            id=f"{pkg}.helpers.noop",
            kind=NodeKind.FUNCTION,
            file=Path(f"{pkg}/helpers.py"),
            line=1,
            name="noop",
        )
        with_leaves.add_node(leaf)
    save_graph(with_leaves, out / "perturbations" / "add_dead_leaf.json")
    # Dead leaf nodes have no mapping — they are new
    dead_leaf_mapping = {nid: nid for nid in clean.nodes}

    _write_json(out / "node_mapping.json", {
        "rename": node_mapping,
        "add_dead_leaf": dead_leaf_mapping,
    })
    _write_json(out / "metadata.json", {
        "split": "public",
        "description": "Structure-preserving perturbations of layered_app",
        "perturbation_families": ["rename", "add_dead_leaf"],
    })


# ---------------------------------------------------------------------------
# Anomaly dataset
# ---------------------------------------------------------------------------

def generate_anomalies(clean: CodeGraph) -> None:
    """Generate anomaly gold labels from mutation-based cases."""
    # Gold regions use MODULE-LEVEL node IDs (after projection)
    # Reuse reverse dep as anomaly case
    out = DATASET_ROOT / "anomalies" / "reverse_dep"
    mutated = _add_edge(clean, "data.store.load_order", "api.serializers.serialize_order", EdgeKind.CALLS)
    save_graph(mutated, out / "graph.json")
    _write_json(out / "gold.json", {
        "anomalies": [
            {
                "kind": "cross_module",
                "region_nodes": ["data.store", "api.serializers"],
            }
        ],
    })
    _write_json(out / "metadata.json", {"split": "public", "description": "Reverse dependency anomaly"})

    # Cycle anomaly case (3-module ring)
    out = DATASET_ROOT / "anomalies" / "cycle"
    mutated = _add_edge(clean, "data.store.save_order", "api.routes.submit_order", EdgeKind.CALLS)
    save_graph(mutated, out / "graph.json")
    _write_json(out / "gold.json", {
        "anomalies": [
            {
                "kind": "cycle_member",
                "region_nodes": ["api.routes", "core.service", "data.store"],
            }
        ],
    })
    _write_json(out / "metadata.json", {"split": "public", "description": "Three-module cycle anomaly"})

    # Boundary erosion anomaly
    out = DATASET_ROOT / "anomalies" / "boundary_erosion"
    mutated = _add_edges(clean, [
        ("api.routes.submit_order", "data.audit.record_event", EdgeKind.CALLS),
        ("api.routes.get_order", "data.store.load_order", EdgeKind.CALLS),
    ])
    save_graph(mutated, out / "graph.json")
    _write_json(out / "gold.json", {
        "anomalies": [
            {
                "kind": "cross_module",
                "region_nodes": ["api.routes", "data.audit", "data.store"],
            }
        ],
    })
    _write_json(out / "metadata.json", {"split": "public", "description": "Boundary erosion anomaly"})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Parsing layered_app fixture...")
    clean = parse_fixture("layered_app")
    print(f"  {clean.node_count} nodes, {clean.edge_count} edges")

    print("Generating mutation datasets (12 cases)...")
    generate_mutations(clean)

    print("Generating architecture recovery dataset...")
    generate_architecture(clean)

    print("Generating stability dataset...")
    generate_stability(clean)

    print("Generating anomaly datasets...")
    generate_anomalies(clean)

    print("Done! Datasets written to benchmark/datasets/")


if __name__ == "__main__":
    main()
