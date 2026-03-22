#!/usr/bin/env python3
"""
Harvest corpus: clone repos from corpus_manifest.json and parse them with topo.

Usage:
    python3 benchmark/scripts/harvest_corpus.py                  # All repos
    python3 benchmark/scripts/harvest_corpus.py flask requests    # Specific repos
    python3 benchmark/scripts/harvest_corpus.py --stats           # Show corpus stats

Cached parsed graphs are stored in benchmark/corpus/<name>/graph.json (gitignored).
"""

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
MANIFEST = PROJECT_ROOT / "benchmark" / "corpus_manifest.json"
CORPUS_DIR = PROJECT_ROOT / "benchmark" / "corpus"
CLONE_DIR = Path("/tmp/topo-corpus")


def load_manifest() -> list[dict]:
    with open(MANIFEST) as f:
        return json.load(f)


def clone_repo(entry: dict) -> Path | None:
    """Clone repo to temp dir. Returns clone path or None on failure."""
    name = entry["name"]
    clone_path = CLONE_DIR / name

    if clone_path.exists():
        print(f"  Using cached clone: {clone_path}")
        return clone_path

    url = entry["url"]
    commit = entry.get("commit", "HEAD")

    try:
        if commit == "HEAD":
            subprocess.run(
                ["git", "clone", "--depth", "1", url, str(clone_path)],
                capture_output=True, timeout=120, check=True,
            )
        else:
            subprocess.run(
                ["git", "clone", "--depth", "50", url, str(clone_path)],
                capture_output=True, timeout=120, check=True,
            )
            subprocess.run(
                ["git", "checkout", commit],
                cwd=clone_path, capture_output=True, timeout=10,
            )
        return clone_path
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"  FAILED to clone {name}: {e}")
        return None


def parse_repo(entry: dict, clone_path: Path) -> bool:
    """Parse repo with topo. Returns True on success."""
    name = entry["name"]
    language = entry["language"]
    entrypoint = entry.get("entrypoint", ".")
    out_dir = CORPUS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "graph.json"

    src_path = clone_path if entrypoint == "." else clone_path / entrypoint

    try:
        result = subprocess.run(
            ["cargo", "run", "-p", "topo-cli", "--", "parse",
             str(src_path), "--language", language, "-o", str(out_path)],
            cwd=PROJECT_ROOT,
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            print(f"  PARSE FAILED for {name}: {result.stderr[:200]}")
            return False

        # Write metadata
        with open(out_path) as f:
            graph = json.load(f)
        meta = {
            "name": name,
            "language": language,
            "url": entry["url"],
            "commit": entry.get("commit", "HEAD"),
            "nodes": len(graph.get("nodes", [])),
            "edges": len(graph.get("edges", [])),
        }
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  OK: {meta['nodes']} nodes, {meta['edges']} edges")
        return True
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT for {name}")
        return False


def show_stats():
    """Show corpus statistics."""
    if not CORPUS_DIR.exists():
        print("No corpus directory found. Run harvest first.")
        return

    total = 0
    by_language: dict[str, int] = {}
    total_nodes = 0
    total_edges = 0

    for meta_path in sorted(CORPUS_DIR.glob("*/metadata.json")):
        with open(meta_path) as f:
            meta = json.load(f)
        total += 1
        lang = meta.get("language", "?")
        by_language[lang] = by_language.get(lang, 0) + 1
        total_nodes += meta.get("nodes", 0)
        total_edges += meta.get("edges", 0)

    print(f"Corpus: {total} repos")
    for lang, count in sorted(by_language.items()):
        print(f"  {lang}: {count}")
    print(f"Total nodes: {total_nodes:,}")
    print(f"Total edges: {total_edges:,}")


def main():
    if "--stats" in sys.argv:
        show_stats()
        return

    manifest = load_manifest()
    targets = sys.argv[1:]

    if targets:
        manifest = [e for e in manifest if e["name"] in targets]
        if not manifest:
            print(f"No matching repos found for: {targets}")
            sys.exit(1)

    CLONE_DIR.mkdir(parents=True, exist_ok=True)
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    ok_count = 0
    fail_count = 0

    for entry in manifest:
        name = entry["name"]

        # Skip if already parsed
        cached = CORPUS_DIR / name / "graph.json"
        if cached.exists() and "--force" not in sys.argv:
            print(f"[skip] {name} (cached)")
            ok_count += 1
            continue

        print(f"[harvest] {name}")
        clone_path = clone_repo(entry)
        if clone_path is None:
            fail_count += 1
            continue

        if parse_repo(entry, clone_path):
            ok_count += 1
        else:
            fail_count += 1

    print(f"\nDone: {ok_count} ok, {fail_count} failed, {len(manifest)} total")


if __name__ == "__main__":
    main()
