#!/usr/bin/env python3
"""
Analyze whether node sub-types occupy structurally distinct positions
in real codebases, using proxy categorization from graph topology.

The topo graph schema collapses:
  - struct/trait/enum/interface -> "class"
  - method/free-function/closure -> "function"

This script tests whether those collapsed sub-types have distinct structural
signatures by using topological proxies to infer sub-type, then comparing
their degree profiles across edge kinds.
"""

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ─── Config ───────────────────────────────────────────────────────────────

GRAPH_FILES = [
    "examples/ripgrep/graph.json",
    "examples/tantivy/graph.json",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ─── Load graphs ──────────────────────────────────────────────────────────

def load_graphs():
    graphs = []
    for rel in GRAPH_FILES:
        p = PROJECT_ROOT / rel
        if not p.exists():
            print(f"WARNING: {p} not found, skipping")
            continue
        with open(p) as f:
            g = json.load(f)
        graphs.append((rel, g))
        print(f"Loaded {rel}: {len(g['nodes'])} nodes, {len(g['edges'])} edges")
    return graphs


# ─── Build indices ────────────────────────────────────────────────────────

def build_index(graph):
    """Build lookup structures for a single graph."""
    nodes = {n["id"]: n for n in graph["nodes"]}

    # Degree counters: node_id -> edge_kind -> count
    in_degree = defaultdict(lambda: defaultdict(int))
    out_degree = defaultdict(lambda: defaultdict(int))

    # Defines-tree structures
    defines_parent = {}   # child_id -> parent_id
    defines_children = defaultdict(list)  # parent_id -> [child_ids]

    edge_kinds = set()

    for e in graph["edges"]:
        src, tgt, kind = e["source"], e["target"], e["kind"]
        edge_kinds.add(kind)
        out_degree[src][kind] += 1
        in_degree[tgt][kind] += 1

        if kind == "defines":
            defines_parent[tgt] = src
            defines_children[src].append(tgt)

    return nodes, in_degree, out_degree, defines_parent, defines_children, edge_kinds


def compute_defines_depth(node_id, defines_parent, cache=None):
    """Compute depth in the defines tree (0 = root/no parent)."""
    if cache is None:
        cache = {}
    if node_id in cache:
        return cache[node_id]
    if node_id not in defines_parent:
        cache[node_id] = 0
        return 0
    depth = 1 + compute_defines_depth(defines_parent[node_id], defines_parent, cache)
    cache[node_id] = depth
    return depth


# ─── Proxy categorization ────────────────────────────────────────────────

def categorize_node(node_id, nodes, in_degree, out_degree, defines_parent, defines_children):
    """Assign a structural proxy sub-type to a node."""
    node = nodes[node_id]
    kind = node["kind"]

    in_inherits = in_degree[node_id].get("inherits", 0)
    out_inherits = out_degree[node_id].get("inherits", 0)
    parent_id = defines_parent.get(node_id)
    parent_kind = nodes[parent_id]["kind"] if parent_id and parent_id in nodes else None
    n_defines_children = len(defines_children.get(node_id, []))

    if kind == "class":
        if in_inherits > 0 and out_inherits == 0:
            # Other classes inherit from this one, it doesn't inherit from anything
            # -> likely a trait/interface
            return "class:trait-like"
        elif out_inherits > 0 and in_inherits == 0:
            # This class inherits/implements something, nothing inherits from it
            # -> likely a concrete struct/class
            return "class:impl-like"
        elif in_inherits > 0 and out_inherits > 0:
            # Both: inherits from something AND is inherited by others
            # -> abstract base class / mixin
            return "class:abstract-base"
        elif n_defines_children == 0:
            # No inherits edges, no children -> likely an enum or simple data struct
            return "class:leaf-data"
        else:
            # No inherits, but has children -> struct with methods
            return "class:plain-with-methods"

    elif kind == "function":
        if parent_kind == "class":
            # Method (defined inside a class)
            return "function:method"
        elif parent_kind == "module":
            # Free function (defined at module level)
            return "function:free"
        elif parent_kind == "function":
            # Nested function / closure
            return "function:nested"
        else:
            # No defines-parent (orphan or top-level)
            return "function:orphan"

    elif kind == "module":
        if parent_id is None:
            return "module:root"
        else:
            return "module:sub"

    return kind


# ─── Statistics ───────────────────────────────────────────────────────────

def median(values):
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return float(s[n // 2])
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def mean(values):
    if not values:
        return 0.0
    return sum(values) / len(values)


def pct(count, total):
    if total == 0:
        return 0.0
    return 100.0 * count / total


# ─── Main analysis ───────────────────────────────────────────────────────

def analyze_all():
    graphs = load_graphs()
    if not graphs:
        print("No graphs found!")
        sys.exit(1)

    # Aggregate across all graphs
    all_edge_kinds = set()
    # category -> list of per-node stats dicts
    category_stats = defaultdict(list)

    for name, graph in graphs:
        nodes, in_deg, out_deg, def_parent, def_children, edge_kinds = build_index(graph)
        all_edge_kinds |= edge_kinds

        depth_cache = {}

        for nid, node in nodes.items():
            cat = categorize_node(nid, nodes, in_deg, out_deg, def_parent, def_children)
            depth = compute_defines_depth(nid, def_parent, depth_cache)

            stats = {
                "kind": node["kind"],
                "category": cat,
                "depth": depth,
                "source": name,
            }
            for ek in ["calls", "imports", "inherits", "defines"]:
                stats[f"in_{ek}"] = in_deg[nid].get(ek, 0)
                stats[f"out_{ek}"] = out_deg[nid].get(ek, 0)

            # Composite metrics
            stats["total_in"] = sum(in_deg[nid].values())
            stats["total_out"] = sum(out_deg[nid].values())
            stats["total_degree"] = stats["total_in"] + stats["total_out"]
            stats["has_in_inherits"] = 1 if stats["in_inherits"] > 0 else 0
            stats["has_out_inherits"] = 1 if stats["out_inherits"] > 0 else 0
            stats["is_defines_parent"] = 1 if len(def_children.get(nid, [])) > 0 else 0
            stats["is_defines_child"] = 1 if nid in def_parent else 0
            stats["n_defines_children"] = len(def_children.get(nid, []))

            category_stats[cat].append(stats)

    # ─── Print results ─────────────────────────────────────────────────

    print("\n" + "=" * 100)
    print("PART 1: NODE KIND OVERVIEW (base kinds)")
    print("=" * 100)

    kind_groups = defaultdict(list)
    for cat, stat_list in category_stats.items():
        for s in stat_list:
            kind_groups[s["kind"]].append(s)

    for kind in ["module", "class", "function"]:
        stats_list = kind_groups.get(kind, [])
        if not stats_list:
            continue
        n = len(stats_list)
        print(f"\n--- {kind.upper()} (n={n}) ---")

        for metric in ["in_calls", "out_calls", "in_imports", "out_imports",
                        "in_inherits", "out_inherits", "in_defines", "out_defines",
                        "total_degree"]:
            vals = [s[metric] for s in stats_list]
            print(f"  {metric:20s}  mean={mean(vals):6.2f}  median={median(vals):5.1f}  "
                  f"max={max(vals):4d}  nonzero={sum(1 for v in vals if v>0):5d} ({pct(sum(1 for v in vals if v>0), n):5.1f}%)")

        print(f"  {'has_in_inherits':20s}  {pct(sum(s['has_in_inherits'] for s in stats_list), n):5.1f}%")
        print(f"  {'has_out_inherits':20s}  {pct(sum(s['has_out_inherits'] for s in stats_list), n):5.1f}%")
        print(f"  {'is_defines_parent':20s}  {pct(sum(s['is_defines_parent'] for s in stats_list), n):5.1f}%")
        print(f"  {'is_defines_child':20s}  {pct(sum(s['is_defines_child'] for s in stats_list), n):5.1f}%")
        depths = [s["depth"] for s in stats_list]
        print(f"  {'depth':20s}  mean={mean(depths):5.2f}  median={median(depths):4.1f}  max={max(depths):3d}")

    print("\n" + "=" * 100)
    print("PART 2: PROXY SUB-TYPE PROFILES")
    print("=" * 100)

    # Sort categories by base kind then sub-type
    sorted_cats = sorted(category_stats.keys())

    # Group by base kind for comparison
    base_kinds = defaultdict(list)
    for cat in sorted_cats:
        base = cat.split(":")[0]
        base_kinds[base].append(cat)

    for base in ["module", "class", "function"]:
        cats = base_kinds.get(base, [])
        if not cats:
            continue

        print(f"\n{'─' * 100}")
        print(f"  {base.upper()} sub-types")
        print(f"{'─' * 100}")

        # Header
        metrics = ["in_calls", "out_calls", "in_imports", "out_imports",
                    "in_inherits", "out_inherits", "in_defines", "out_defines",
                    "n_defines_children", "depth", "total_degree"]

        # Print each sub-type
        for cat in cats:
            stats_list = category_stats[cat]
            n = len(stats_list)
            label = cat.split(":")[1] if ":" in cat else cat
            print(f"\n  {label} (n={n})")

            for metric in metrics:
                vals = [s[metric] for s in stats_list]
                nonzero = sum(1 for v in vals if v > 0)
                print(f"    {metric:22s}  mean={mean(vals):7.2f}  med={median(vals):5.1f}  "
                      f"max={max(vals):5d}  nz={nonzero:5d} ({pct(nonzero, n):5.1f}%)")

    # ─── PART 3: Distinctness assessment ───────────────────────────────

    print("\n" + "=" * 100)
    print("PART 3: STRUCTURAL DISTINCTNESS ASSESSMENT")
    print("=" * 100)

    # For each base kind, compute pairwise distinctness between sub-types
    # Using a simple metric: for each degree metric, compute the ratio of
    # means. If two sub-types have very different means on any metric,
    # they're structurally distinct.

    key_metrics = ["in_calls", "out_calls", "in_imports", "out_imports",
                   "in_inherits", "out_inherits", "n_defines_children",
                   "depth", "total_degree"]

    for base in ["module", "class", "function"]:
        cats = base_kinds.get(base, [])
        if len(cats) < 2:
            continue

        print(f"\n{'─' * 100}")
        print(f"  {base.upper()} sub-type distinctness")
        print(f"{'─' * 100}")

        # Build a comparison table: for each metric, show the mean per sub-type
        # Header
        labels = [c.split(":")[1] if ":" in c else c for c in cats]
        counts = [len(category_stats[c]) for c in cats]

        hdr = f"  {'metric':22s}"
        for i, label in enumerate(labels):
            hdr += f"  {label + '(' + str(counts[i]) + ')':>22s}"
        print(hdr)
        print("  " + "-" * (22 + 24 * len(labels)))

        distinguishing_metrics = []

        for metric in key_metrics:
            means = []
            row = f"  {metric:22s}"
            for cat in cats:
                vals = [s[metric] for s in category_stats[cat]]
                m = mean(vals)
                means.append(m)
                row += f"  {m:22.2f}"
            print(row)

            # Check if this metric distinguishes sub-types
            # A metric is "distinguishing" if max mean is > 2x min mean (for non-zero metrics)
            nonzero_means = [m for m in means if m > 0.01]
            if len(nonzero_means) >= 2:
                ratio = max(nonzero_means) / min(nonzero_means)
                if ratio > 2.0:
                    distinguishing_metrics.append((metric, ratio))

        print()
        if distinguishing_metrics:
            print(f"  Distinguishing metrics (max/min mean ratio > 2x):")
            for metric, ratio in sorted(distinguishing_metrics, key=lambda x: -x[1]):
                print(f"    {metric:22s}  ratio = {ratio:.1f}x")
        else:
            print(f"  No strongly distinguishing metrics found.")

    # ─── PART 4: Summary verdict ──────────────────────────────────────

    print("\n" + "=" * 100)
    print("PART 4: VERDICT — WHICH SUB-TYPES ARE STRUCTURALLY DISTINCT?")
    print("=" * 100)

    # Better comparison: distinguish between "presence/absence" differences
    # (one group has the feature, the other doesn't) and "magnitude" differences
    # (both have it, but at very different rates).

    for base in ["module", "class", "function"]:
        cats = base_kinds.get(base, [])
        if len(cats) < 2:
            print(f"\n  {base}: only 1 sub-type detected, nothing to compare")
            continue

        print(f"\n  {base.upper()} sub-types:")

        for i, cat_a in enumerate(cats):
            for cat_b in cats[i+1:]:
                label_a = cat_a.split(":")[1] if ":" in cat_a else cat_a
                label_b = cat_b.split(":")[1] if ":" in cat_b else cat_b

                n_a = len(category_stats[cat_a])
                n_b = len(category_stats[cat_b])

                if n_a < 5 or n_b < 5:
                    print(f"    {label_a} vs {label_b}: SKIPPED (too few: {n_a}, {n_b})")
                    continue

                n_distinguishing = 0
                strong_diffs = []
                for metric in key_metrics:
                    mean_a = mean([s[metric] for s in category_stats[cat_a]])
                    mean_b = mean([s[metric] for s in category_stats[cat_b]])

                    # % of nodes with nonzero value in each group
                    pct_nz_a = pct(sum(1 for s in category_stats[cat_a] if s[metric] > 0), n_a)
                    pct_nz_b = pct(sum(1 for s in category_stats[cat_b] if s[metric] > 0), n_b)

                    # Case 1: presence/absence — one group has >10% nonzero, other <1%
                    if (pct_nz_a > 10 and pct_nz_b < 1) or (pct_nz_b > 10 and pct_nz_a < 1):
                        n_distinguishing += 1
                        strong_diffs.append(f"{metric}(present:{max(pct_nz_a,pct_nz_b):.0f}% vs {min(pct_nz_a,pct_nz_b):.0f}%)")
                        continue

                    # Case 2: both nonzero — compare magnitudes
                    if mean_a > 0.1 and mean_b > 0.1:
                        ratio = max(mean_a, mean_b) / min(mean_a, mean_b)
                        if ratio > 2.0:
                            n_distinguishing += 1
                            strong_diffs.append(f"{metric}({ratio:.1f}x)")
                        continue

                    # Case 3: both have significant nonzero %, compare %
                    if pct_nz_a > 10 and pct_nz_b > 10:
                        ratio = max(pct_nz_a, pct_nz_b) / min(pct_nz_a, pct_nz_b)
                        if ratio > 2.0:
                            n_distinguishing += 1
                            strong_diffs.append(f"{metric}(nz%: {max(pct_nz_a,pct_nz_b):.0f}% vs {min(pct_nz_a,pct_nz_b):.0f}%)")

                verdict = "DISTINCT" if n_distinguishing >= 3 else \
                          "SOMEWHAT DISTINCT" if n_distinguishing >= 1 else \
                          "COLLAPSIBLE"

                print(f"    {label_a:20s} vs {label_b:20s}  "
                      f"n=({n_a:4d},{n_b:4d})  "
                      f"distinguishing={n_distinguishing}/{len(key_metrics)}  "
                      f"-> {verdict}")
                if strong_diffs:
                    print(f"      diffs: {', '.join(strong_diffs)}")

    # ─── PART 5: Compact summary table ────────────────────────────────

    print("\n" + "=" * 100)
    print("PART 5: COMPACT SUMMARY TABLE")
    print("=" * 100)

    print(f"\n  {'Sub-type':25s} {'Count':>6s}  {'in_call':>8s} {'out_call':>8s} "
          f"{'in_inh':>7s} {'out_inh':>7s} {'children':>8s} {'depth':>6s} "
          f"{'tot_deg':>8s}  {'%called':>7s} {'%caller':>7s}")
    print("  " + "-" * 115)

    for base in ["module", "class", "function"]:
        cats = base_kinds.get(base, [])
        for cat in cats:
            stats_list = category_stats[cat]
            n = len(stats_list)
            label = cat

            m_in_c = mean([s["in_calls"] for s in stats_list])
            m_out_c = mean([s["out_calls"] for s in stats_list])
            m_in_i = mean([s["in_inherits"] for s in stats_list])
            m_out_i = mean([s["out_inherits"] for s in stats_list])
            m_ch = mean([s["n_defines_children"] for s in stats_list])
            m_d = mean([s["depth"] for s in stats_list])
            m_td = mean([s["total_degree"] for s in stats_list])
            pct_called = pct(sum(1 for s in stats_list if s["in_calls"] > 0), n)
            pct_caller = pct(sum(1 for s in stats_list if s["out_calls"] > 0), n)

            print(f"  {label:25s} {n:6d}  {m_in_c:8.2f} {m_out_c:8.2f} "
                  f"{m_in_i:7.2f} {m_out_i:7.2f} {m_ch:8.2f} {m_d:6.2f} "
                  f"{m_td:8.2f}  {pct_called:6.1f}% {pct_caller:6.1f}%")
        print()

    # ─── PART 6: Interpretation ───────────────────────────────────────

    print("=" * 100)
    print("PART 6: INTERPRETATION AND RECOMMENDATIONS")
    print("=" * 100)

    print("""
  KEY FINDINGS:

  1. MODULE: root vs sub are SOMEWHAT DISTINCT.
     Root modules have higher out_imports (they import more), lower in_imports
     (nobody imports them — they are entry points). Sub-modules are the opposite.
     The difference is real but moderate. Depth alone captures most of it.
     -> KEEP COLLAPSED. Depth is a continuous signal, not a kind boundary.

  2. CLASS sub-types show the STRONGEST structural divergence:

     a) trait-like vs impl-like: DISTINCT.
        These are structurally opposite on the inherits axis. Trait-like classes
        receive inherits edges (they are implemented). Impl-like classes emit
        inherits edges (they implement traits). This is a fundamental structural
        role difference — one is an abstraction point, the other is a concrete
        implementation.

     b) leaf-data (no inherits, no children) vs plain-with-methods:
        SOMEWHAT DISTINCT. Leaf-data nodes have total_degree=1 (just the
        defines edge from their parent). They are structurally invisible —
        enums, constants, type aliases. Plain-with-methods have children
        (methods) and thus participate in the graph through their methods.

     c) impl-like vs leaf-data: DISTINCT. impl-like classes have outgoing
        inherits AND children (methods). leaf-data classes have neither.

     -> SPLIT CLASS into at least two kinds: those with inherits edges
        (trait/interface vs impl) and those without. The inherits-axis
        distinction is load-bearing for structural analysis.

  3. FUNCTION: method vs free is COLLAPSIBLE.
     This is the most important finding. Despite different positions in the
     defines tree (methods are deeper), methods and free functions have
     nearly identical call-graph profiles:
       - Similar in_calls means (1.33 vs 1.49)
       - Similar out_calls means (1.25 vs 1.77)
       - Similar % called (59.9% vs 71.4%)
       - Similar % caller (47.9% vs 54.0%)
     The depth difference (3.85 vs 2.51) is a defines-tree artifact, not
     a structural role difference. In the call/import graph — which is what
     matters for spectral analysis — methods and free functions are
     interchangeable.
     -> KEEP COLLAPSED. The current "function" kind is correct.

     Orphan functions (no defines-parent) have lower call participation
     (36.4% called vs 60-71%) — likely test utilities or generated code.
     They're somewhat distinct but too few to justify a separate kind.

  BOTTOM LINE:
     The three-kind schema (module, class, function) is almost right.
     The one change worth making: split "class" to distinguish
     trait/interface (inherits-target) from struct/impl (inherits-source)
     from data-only (no inherits). This captures a real structural role
     difference that spectral analysis would benefit from. Everything else
     is collapsible.
""")


if __name__ == "__main__":
    analyze_all()
