"""
JSON serialization for structural analysis results.

Converts Python StructuralAnalysis objects into dicts matching
schemas/analysis.schema.json (v3). Formatting/presentation has moved
to topo-formatter.
"""

from __future__ import annotations

from topo_analyzer.analysis import _module_label


def to_dict(analysis) -> dict:
    """Convert the full analysis result into a JSON-serializable dict.

    Matches schemas/analysis.schema.json (v3).
    """
    return {
        "scope": {
            "level": analysis.projection.config.level.value,
            "edge_kinds": [kind.value for kind in analysis.projection.config.edge_kinds],
            "internal_only": analysis.projection.config.internal_only,
            "roots": analysis.projection.config.scope_labels,
        },
        "coverage": {
            "analyzed_nodes": analysis.graph.node_count,
            "analyzed_edges": analysis.graph.edge_count,
            "parsed_nodes": analysis.raw_graph.node_count,
            "parsed_edges": analysis.raw_graph.edge_count,
        },
        "spectral": {
            "fiedler_value": analysis.spectral.fiedler_value,
            "eigenvalues": analysis.spectral.eigenvalues.tolist(),
            "nodes_covered": analysis.spectral.analyzed_node_count,
            "coverage_ratio": round(analysis.spectral.coverage_ratio, 4),
            "components": analysis.spectral.component_count,
            "largest_component_ratio": round(analysis.spectral.largest_component_ratio, 4),
        } if analysis.spectral else None,
        "architecture": {
            "modules": [
                {
                    "id": module.id,
                    "size": module.size,
                    "members": module.node_ids,
                    "label": _module_label(module),
                    "cohesion": round(module.cohesion, 4) if module.cohesion is not None else None,
                    "separation": round(module.separation, 4) if module.separation is not None else None,
                    "confidence": round(module.confidence, 4),
                    "unassigned": module.unassigned,
                }
                for module in analysis.modules
            ],
            "dependencies": _build_module_dependencies(analysis),
            "silhouette": round(analysis.module_detection.silhouette, 4)
            if analysis.module_detection.silhouette is not None else None,
            "package_fallback": analysis.module_detection.package_fallback,
        },
        "roles": [
            {
                "node_id": role.node_id,
                "role": role.role.value,
                "degree": role.degree,
                "betweenness": round(role.betweenness, 4),
                "in_degree": role.in_degree,
                "out_degree": role.out_degree,
                "anchor": _first_anchor(analysis.projection.anchors_for([role.node_id], limit=1)),
            }
            for role in analysis.roles
        ],
        "issues": [
            {
                "id": finding.id,
                "kind": finding.kind,
                "title": finding.title,
                "description": finding.description,
                "severity": round(finding.severity, 2),
                "severity_label": finding.severity_label,
                "confidence": round(finding.confidence, 2),
                "confidence_label": finding.confidence_label,
                "anchors": [anchor.to_dict() for anchor in finding.anchors],
            }
            for finding in analysis.findings
        ],
        "health": {
            "modularity_q": analysis.health.modularity_q,
        } if analysis.health else None,
    }


def _build_module_dependencies(analysis) -> list[dict]:
    """Aggregate cross-module edges into directed dependency records."""
    node_to_module: dict[str, int] = {}
    for module in analysis.modules:
        for node_id in module.node_ids:
            node_to_module[node_id] = module.id

    dep_map: dict[tuple[int, int], dict] = {}
    for edge in analysis.graph.edges:
        src_mod = node_to_module.get(edge.source)
        tgt_mod = node_to_module.get(edge.target)
        if src_mod is None or tgt_mod is None or src_mod == tgt_mod:
            continue
        key = (src_mod, tgt_mod)
        if key not in dep_map:
            dep_map[key] = {"weight": 0, "edge_kinds": {}}
        dep_map[key]["weight"] += 1
        kind_str = edge.kind.value
        dep_map[key]["edge_kinds"][kind_str] = dep_map[key]["edge_kinds"].get(kind_str, 0) + 1

    return [
        {"source": src, "target": tgt, "weight": info["weight"], "edge_kinds": info["edge_kinds"]}
        for (src, tgt), info in sorted(dep_map.items())
    ]


def _first_anchor(anchors: list) -> dict | None:
    """Return the first anchor as a dict, or None."""
    if anchors:
        return anchors[0].to_dict()
    return None
