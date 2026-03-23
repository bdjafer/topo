#!/usr/bin/env python3
"""Compute real CodeLM semantic embeddings for all parsed repos.

Uses the Jina Embeddings API (jina-embeddings-v2-base-code, 768d)
to embed source code for each node in graph.json.

Outputs embeddings.json per repo, consumed by `topo export-features --embeddings`.

Usage:
    python compute_embeddings.py                      # All parsed repos
    python compute_embeddings.py --repos flask,click   # Specific repos
    python compute_embeddings.py --force               # Re-embed even if cached

Requires JINA_API_KEY in .env or environment.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples"
CLONE_DIR = Path("/tmp/topo-corpus")

API_URL = "https://api.jina.ai/v1/embeddings"
MODEL_NAME = "jina-embeddings-v2-base-code"
EMBED_DIM = 768

# API batching: send up to 128 texts per request to minimize round-trips.
# Each text is truncated to 1500 chars (~375 tokens), so 128 * 375 ≈ 48K tokens/request.
API_BATCH_SIZE = 128
MAX_SOURCE_CHARS = 1500


def load_api_key() -> str:
    """Load Jina API key from .env or environment."""
    key = os.environ.get("JINA_API_KEY")
    if key:
        return key
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("JINA_API_KEY="):
                return line.split("=", 1)[1].strip().strip("'\"")
    print("ERROR: JINA_API_KEY not found in .env or environment", file=sys.stderr)
    sys.exit(1)


def extract_source_text(node: dict, clone_path: Path) -> str:
    """Extract source code text for a node, truncated for API efficiency."""
    file_path_str = node.get("file", "")
    line_start = node.get("line", 1)
    line_end = node.get("line_end", line_start)
    node_id = node.get("id", "unknown")
    kind = node.get("kind", "unknown")
    name = node.get("name", node_id)

    file_path = _resolve_file(file_path_str, clone_path) if file_path_str else None

    source = ""
    if file_path and file_path.exists():
        try:
            with open(file_path, "r", errors="replace") as f:
                all_lines = f.readlines()
            start_idx = max(0, line_start - 1)
            end_idx = min(len(all_lines), line_end)
            source = "".join(all_lines[start_idx:end_idx])
        except (OSError, UnicodeDecodeError):
            pass

    header = f"# {kind}: {node_id}\n"
    if source:
        if len(source) > MAX_SOURCE_CHARS:
            source = source[:MAX_SOURCE_CHARS]
        return header + source
    else:
        return f"{kind} {name} ({node_id})"


def _resolve_file(file_path_str: str, clone_path: Path) -> Path | None:
    """Resolve a node's file path to an actual file on disk."""
    file_path = Path(file_path_str)
    if file_path.exists():
        return file_path
    try:
        rel = file_path.relative_to(CLONE_DIR / clone_path.name)
        candidate = clone_path / rel
        if candidate.exists():
            return candidate
    except ValueError:
        pass
    parts = file_path.parts
    for i, part in enumerate(parts):
        if part == "topo-corpus" and i + 2 < len(parts):
            candidate = clone_path / Path(*parts[i + 2:])
            if candidate.exists():
                return candidate
    return None


def embed_batch(texts: list[str], api_key: str) -> list[list[float]]:
    """Call Jina API for a batch of texts. Returns list of embedding vectors."""
    body = json.dumps({
        "input": texts,
        "model": MODEL_NAME,
        "normalized": True,
        "embedding_type": "float",
    }).encode()

    req = Request(
        API_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "topo-embeddings/1.0",
        },
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            with urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())
            # Sort by index to preserve order
            data = sorted(result["data"], key=lambda d: d["index"])
            return [d["embedding"] for d in data]
        except HTTPError as e:
            if e.code == 429:
                wait = 2 ** attempt * 5
                print(f"\n    Rate limited, waiting {wait}s...", end="", flush=True)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("Max retries exceeded")


def embed_repo(repo_name: str, api_key: str, force: bool = False) -> dict:
    """Compute embeddings for one repo via Jina API."""
    repo_dir = EXAMPLES_DIR / repo_name
    graph_path = repo_dir / "graph.json"
    output_path = repo_dir / "embeddings.json"

    if not graph_path.exists():
        return {"repo": repo_name, "status": "not_parsed"}
    if not force and output_path.exists():
        return {"repo": repo_name, "status": "skipped"}

    clone_path = CLONE_DIR / repo_name
    if not clone_path.exists():
        return {"repo": repo_name, "status": "no_clone"}

    try:
        with open(graph_path) as f:
            graph = json.load(f)
        nodes = graph.get("nodes", [])
        n = len(nodes)
        if n == 0:
            return {"repo": repo_name, "status": "empty"}

        t0 = time.time()
        node_ids = [node["id"] for node in nodes]
        texts = [extract_source_text(node, clone_path) for node in nodes]

        # Embed in batches via API
        all_embeddings = []
        for i in range(0, n, API_BATCH_SIZE):
            batch = texts[i : i + API_BATCH_SIZE]
            embs = embed_batch(batch, api_key)
            all_embeddings.extend(embs)

        embed_time = time.time() - t0

        # Verify
        assert len(all_embeddings) == n
        assert len(all_embeddings[0]) == EMBED_DIM

        # Write embeddings.json
        result = {nid: emb for nid, emb in zip(node_ids, all_embeddings)}
        with open(output_path, "w") as f:
            json.dump(result, f)

        return {
            "repo": repo_name,
            "status": "ok",
            "n_nodes": n,
            "embed_time_s": round(embed_time, 1),
            "file_size_mb": round(output_path.stat().st_size / 1e6, 1),
        }
    except Exception as e:
        return {"repo": repo_name, "status": "error", "error": str(e)[:200]}


def main():
    parser = argparse.ArgumentParser(
        description="Compute CodeLM embeddings via Jina API."
    )
    parser.add_argument(
        "--repos", type=str, default=None,
        help="Comma-separated repo names (default: all with graph.json + clone)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-embed even if embeddings.json exists",
    )
    args = parser.parse_args()

    api_key = load_api_key()
    print(f"Using Jina API ({MODEL_NAME}, {EMBED_DIM}d)")

    if args.repos:
        repo_names = [r.strip() for r in args.repos.split(",")]
    else:
        repo_names = sorted(
            d.name for d in EXAMPLES_DIR.iterdir()
            if d.is_dir()
            and (d / "graph.json").exists()
            and (CLONE_DIR / d.name).exists()
        )

    if not repo_names:
        print("No repos found with both graph.json and cloned source.", file=sys.stderr)
        sys.exit(1)

    print(f"Embedding {len(repo_names)} repos...\n")
    results = []
    total_nodes = 0
    total_time = 0.0

    for i, name in enumerate(repo_names, 1):
        print(f"  [{i}/{len(repo_names)}] {name}", end=" ", flush=True)
        r = embed_repo(name, api_key, force=args.force)
        print(f"→ {r['status']}", end="")
        if r.get("n_nodes"):
            print(f" ({r['n_nodes']} nodes, {r['embed_time_s']}s)", end="")
            total_nodes += r["n_nodes"]
            total_time += r.get("embed_time_s", 0)
        if r.get("error"):
            print(f" ERROR: {r['error']}", end="")
        print()
        results.append(r)

    ok = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors = sum(1 for r in results if r["status"] == "error")

    print(f"\nDone: {ok} embedded, {skipped} skipped, {errors} errors")
    print(f"Total: {total_nodes:,} nodes in {total_time:.0f}s")

    summary_path = EXAMPLES_DIR / "embedding_report.json"
    with open(summary_path, "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "embed_dim": EMBED_DIM,
            "total_repos": len(results),
            "ok": ok,
            "skipped": skipped,
            "errors": errors,
            "total_nodes": total_nodes,
            "total_time_s": round(total_time, 1),
            "repos": sorted(results, key=lambda r: r["repo"]),
        }, f, indent=2)
    print(f"Report: {summary_path}")


if __name__ == "__main__":
    main()
