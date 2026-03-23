#!/usr/bin/env python3
"""
Report distributional coverage of the current registry.

Reads examples/registry.toml for tag metadata. Optionally checks
examples/*/graph.json for parse status and examples/*/features.npz
for preprocessing status.

Usage:
    python coverage_report.py                 # Full report
    python coverage_report.py --parsed-only   # Only include parsed repos
    python coverage_report.py --json          # Output machine-readable JSON
"""

import argparse
import json
import sys
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples"
REGISTRY = EXAMPLES_DIR / "registry.toml"

# ── Target distributions (from REGISTRY_EXPANSION_PLAN.md §8) ────

DOMAIN_TARGETS = {
    "web": (20, 25),
    "library": (20, 25),
    "cli": (10, 15),
    "data_ml": (10, 15),
    "systems": (8, 12),
    "devops": (5, 8),
    "other": (10, 15),
}

LANGUAGE_TARGETS = {
    "python": (35, 45),
    "rust": (35, 45),
    "typescript": (15, 25),
}

SIZE_TARGETS = {
    "small": (25, 35),
    "medium": (40, 50),
    "large": (20, 30),
}

QUALITY_TARGETS = {
    "clean": (40, 50),
    "mixed": (30, 40),
    "messy": (15, 25),
}


def load_registry() -> list[dict]:
    """Load all registry entries."""
    with open(REGISTRY, "rb") as f:
        reg = tomllib.load(f)
    return reg.get("example", [])


def check_status(name: str) -> dict:
    """Check parse/preprocess status for a repo."""
    repo_dir = EXAMPLES_DIR / name
    status = {
        "has_graph": (repo_dir / "graph.json").exists(),
        "has_features": (repo_dir / "features.npz").exists(),
        "has_metadata": (repo_dir / "metadata.json").exists(),
        "edge_types": 0,
    }
    # Count edge types from graph.json
    graph_path = repo_dir / "graph.json"
    if status["has_graph"]:
        try:
            with open(graph_path) as f:
                graph = json.load(f)
            edge_kinds = {e.get("kind", "unknown") for e in graph.get("edges", [])}
            status["edge_types"] = len(edge_kinds & {"calls", "imports", "inherits"})
        except (json.JSONDecodeError, OSError):
            pass
    return status


def format_pct(count: int, total: int) -> str:
    """Format a count as percentage."""
    if total == 0:
        return "0.0%"
    return f"{count / total * 100:.1f}%"


def check_target(pct: float, low: int, high: int) -> str:
    """Check if a percentage is within target range."""
    if low <= pct <= high:
        return "ok"
    elif pct < low:
        return f"low (target: {low}-{high}%)"
    else:
        return f"high (target: {low}-{high}%)"


def generate_report(entries: list[dict], parsed_only: bool = False) -> dict:
    """Generate a full coverage report."""
    # Annotate entries with parse/preprocess status
    for entry in entries:
        status = check_status(entry["name"])
        entry["_status"] = status

    # Filter to parsed-only if requested
    if parsed_only:
        entries = [e for e in entries if e["_status"]["has_graph"]]

    total = len(entries)

    # Count parsed/preprocessed
    n_parsed = sum(1 for e in entries if e["_status"]["has_graph"])
    n_preprocessed = sum(1 for e in entries if e["_status"]["has_features"])

    # Pinning status
    n_pinned = sum(1 for e in entries if e.get("commit", "HEAD") != "HEAD")

    # ── Language distribution ──
    lang_counts: dict[str, int] = {}
    for e in entries:
        lang = e.get("language", "unknown").lower()
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

    lang_report = {}
    for lang, (low, high) in LANGUAGE_TARGETS.items():
        count = lang_counts.get(lang, 0)
        pct = count / total * 100 if total > 0 else 0
        lang_report[lang] = {
            "count": count,
            "pct": round(pct, 1),
            "status": check_target(pct, low, high),
        }
    # Add any languages not in targets
    for lang, count in lang_counts.items():
        if lang not in lang_report:
            pct = count / total * 100 if total > 0 else 0
            lang_report[lang] = {"count": count, "pct": round(pct, 1), "status": "no_target"}

    # ── Domain distribution ──
    domain_counts: dict[str, int] = {}
    n_tagged_domain = 0
    for e in entries:
        tags = e.get("tags", {})
        domain = tags.get("domain", "")
        if domain:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            n_tagged_domain += 1

    domain_report = {}
    base = n_tagged_domain if n_tagged_domain > 0 else total
    for domain, (low, high) in DOMAIN_TARGETS.items():
        count = domain_counts.get(domain, 0)
        pct = count / base * 100 if base > 0 else 0
        domain_report[domain] = {
            "count": count,
            "pct": round(pct, 1),
            "status": check_target(pct, low, high),
        }
    for domain, count in domain_counts.items():
        if domain not in domain_report:
            pct = count / base * 100 if base > 0 else 0
            domain_report[domain] = {"count": count, "pct": round(pct, 1), "status": "no_target"}

    # ── Size distribution ──
    size_counts: dict[str, int] = {}
    n_tagged_size = 0
    for e in entries:
        tags = e.get("tags", {})
        size = tags.get("size", "")
        if size:
            size_counts[size] = size_counts.get(size, 0) + 1
            n_tagged_size += 1

    size_report = {}
    base = n_tagged_size if n_tagged_size > 0 else total
    for size, (low, high) in SIZE_TARGETS.items():
        count = size_counts.get(size, 0)
        pct = count / base * 100 if base > 0 else 0
        size_report[size] = {
            "count": count,
            "pct": round(pct, 1),
            "status": check_target(pct, low, high),
        }

    # ── Quality distribution ──
    quality_counts: dict[str, int] = {}
    n_tagged_quality = 0
    for e in entries:
        tags = e.get("tags", {})
        quality = tags.get("quality", "")
        if quality:
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
            n_tagged_quality += 1

    quality_report = {}
    base = n_tagged_quality if n_tagged_quality > 0 else total
    for quality, (low, high) in QUALITY_TARGETS.items():
        count = quality_counts.get(quality, 0)
        pct = count / base * 100 if base > 0 else 0
        quality_report[quality] = {
            "count": count,
            "pct": round(pct, 1),
            "status": check_target(pct, low, high),
        }

    # ── Edge type coverage (parsed repos only) ──
    edge_type_counts: dict[int, int] = {3: 0, 2: 0, 1: 0, 0: 0}
    parsed_entries = [e for e in entries if e["_status"]["has_graph"]]
    for e in parsed_entries:
        et = e["_status"]["edge_types"]
        edge_type_counts[min(et, 3)] += 1
    n_parsed_total = len(parsed_entries)
    edge_type_report = {}
    for n_types in [3, 2, 1]:
        count = edge_type_counts[n_types]
        pct = count / n_parsed_total * 100 if n_parsed_total > 0 else 0
        edge_type_report[f"{n_types}/3"] = {"count": count, "pct": round(pct, 1)}

    # ── Tagging completeness ──
    n_with_any_tag = sum(1 for e in entries if e.get("tags"))

    # ── Gaps ──
    gaps = []
    for lang, info in lang_report.items():
        if "low" in info["status"]:
            gaps.append(f"{lang} language is {info['pct']}% — below target")
    for domain, info in domain_report.items():
        if "low" in info["status"]:
            gaps.append(f"{domain} domain is {info['pct']}% — below target")
    for size, info in size_report.items():
        if "low" in info["status"]:
            gaps.append(f"{size} size is {info['pct']}% — below target")
    for quality, info in quality_report.items():
        if "low" in info["status"]:
            gaps.append(f"{quality} quality is {info['pct']}% — below target")
    # Edge type gap: target ≥80% with 3/3 edge types
    if n_parsed_total > 0:
        pct_3of3 = edge_type_report["3/3"]["pct"]
        if pct_3of3 < 80:
            gaps.append(f"3/3 edge types is {pct_3of3}% — below 80% target")

    return {
        "total": total,
        "parsed": n_parsed,
        "preprocessed": n_preprocessed,
        "pinned": n_pinned,
        "tagged": n_with_any_tag,
        "tagged_domain": n_tagged_domain,
        "tagged_size": n_tagged_size,
        "tagged_quality": n_tagged_quality,
        "language": lang_report,
        "domain": domain_report,
        "size": size_report,
        "quality": quality_report,
        "edge_type_coverage": edge_type_report,
        "gaps": gaps,
    }


def print_report(report: dict) -> None:
    """Print a human-readable coverage report."""
    print("Registry Coverage Report")
    print("=" * 60)
    print(f"Total repos:    {report['total']}")
    print(f"Parsed:         {report['parsed']}")
    print(f"Preprocessed:   {report['preprocessed']}")
    print(f"Pinned (SHA):   {report['pinned']}")
    print(f"Tagged:         {report['tagged']} ({report['tagged_domain']} domain, "
          f"{report['tagged_size']} size, {report['tagged_quality']} quality)")

    def print_distribution(title: str, data: dict) -> None:
        print(f"\n{title}:")
        for key, info in sorted(data.items(), key=lambda x: -x[1]["count"]):
            count = info["count"]
            pct = info["pct"]
            status = info["status"]
            marker = "ok" if status == "ok" else "!!" if "low" in status or "high" in status else "  "
            print(f"  {key:<15} {count:>4}  ({pct:>5}%)  [{marker}] {status}")

    print_distribution("Language Distribution", report["language"])
    print_distribution("Domain Distribution", report["domain"])

    if report["tagged_size"] > 0:
        print_distribution("Size Distribution", report["size"])
    else:
        print("\nSize Distribution: no repos tagged with size yet")

    if report["tagged_quality"] > 0:
        print_distribution("Quality Distribution", report["quality"])
    else:
        print("\nQuality Distribution: no repos tagged with quality yet")

    # Edge type coverage
    etc = report.get("edge_type_coverage", {})
    if any(info["count"] > 0 for info in etc.values()):
        print(f"\nEdge Type Coverage (parsed repos, target: >=80% with 3/3):")
        for key in ["3/3", "2/3", "1/3"]:
            if key in etc:
                info = etc[key]
                print(f"  {key} edge types: {info['count']:>4}  ({info['pct']:>5}%)")
    else:
        print("\nEdge Type Coverage: no parsed repos yet")

    if report["gaps"]:
        print(f"\nGAPS ({len(report['gaps'])}):")
        for gap in report["gaps"]:
            print(f"  !! {gap}")
    else:
        print("\nNo gaps detected — all targets met!")


def main():
    parser = argparse.ArgumentParser(
        description="Report distributional coverage of the current registry."
    )
    parser.add_argument("--parsed-only", action="store_true",
                        help="Only include repos with parsed graph.json")
    parser.add_argument("--json", action="store_true",
                        help="Output machine-readable JSON")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write JSON report to file")
    args = parser.parse_args()

    if not REGISTRY.exists():
        print(f"ERROR: Registry not found at {REGISTRY}", file=sys.stderr)
        sys.exit(1)

    entries = load_registry()
    report = generate_report(entries, parsed_only=args.parsed_only)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)

    # Write JSON report if --output or always to default location
    output_path = args.output or EXAMPLES_DIR / "coverage_report.json"
    output_path.write_text(json.dumps(report, indent=2))
    if not args.json:
        print(f"\nJSON report: {output_path}")


if __name__ == "__main__":
    main()
