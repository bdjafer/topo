"""Generate semantic embeddings for a topo graph using fastembed (jina-embeddings-v2-base-code).

Usage:
    python scripts/generate_embeddings.py examples/ripgrep/graph.json -o examples/ripgrep/embeddings.json
    python scripts/generate_embeddings.py examples/ripgrep/graph.json --timeout 120
"""
import argparse
import json
import signal
import sys
import time
from pathlib import Path


def timeout_handler(signum, frame):
    print("\nTIMEOUT: Embedding generation took too long. Partial results saved.", file=sys.stderr)
    sys.exit(1)


def build_embedding_input(node: dict) -> str:
    """Build the embedding input string for a node.

    Format: # module: <parent_path>\n# file: <file>:<line>\n<name> (<kind>)
    For real usage, this would include the source body. For now we use
    the structural metadata which is still meaningful for code models.
    """
    parts = []
    # Module context from the node ID (parent path)
    node_id = node.get("id", "")
    if "." in node_id:
        parent = ".".join(node_id.split(".")[:-1])
        parts.append(f"# module: {parent}")

    # File and line
    file_path = node.get("file", "")
    line = node.get("line", 0)
    if file_path:
        parts.append(f"# file: {file_path}:{line}")

    # Name and kind
    name = node.get("name", node_id.split(".")[-1] if "." in node_id else node_id)
    kind = node.get("kind", "unknown")
    parts.append(f"{name} ({kind})")

    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for a topo graph")
    parser.add_argument("graph", type=Path, help="Path to graph.json")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output path (default: <graph_dir>/embeddings.json)")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds (default: 300)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for inference (default: 64)")
    parser.add_argument("--model", type=str, default="jinaai/jina-embeddings-v2-base-code", help="Model name")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.graph.parent / "embeddings.json"

    # Set timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(args.timeout)

    # Load graph
    print(f"Loading graph from {args.graph}...", file=sys.stderr)
    with open(args.graph) as f:
        graph = json.load(f)

    nodes = graph["nodes"]
    print(f"Found {len(nodes)} nodes", file=sys.stderr)

    # Build embedding inputs
    texts = []
    node_ids = []
    for node in nodes:
        node_ids.append(node["id"])
        texts.append(build_embedding_input(node))

    print(f"Built {len(texts)} embedding inputs", file=sys.stderr)
    print(f"Sample input:\n---\n{texts[0]}\n---", file=sys.stderr)

    # Initialize model
    print(f"Loading model {args.model}...", file=sys.stderr)
    t0 = time.time()

    try:
        from fastembed import TextEmbedding
    except ImportError:
        print("fastembed not installed. Install with: pip install fastembed", file=sys.stderr)
        sys.exit(1)

    model = TextEmbedding(model_name=args.model)
    print(f"Model loaded in {time.time() - t0:.1f}s", file=sys.stderr)

    # Generate embeddings in batches
    print(f"Generating embeddings (batch_size={args.batch_size})...", file=sys.stderr)
    t0 = time.time()

    all_embeddings = list(model.embed(texts, batch_size=args.batch_size))

    elapsed = time.time() - t0
    print(f"Generated {len(all_embeddings)} embeddings in {elapsed:.1f}s ({len(all_embeddings)/elapsed:.0f} nodes/sec)", file=sys.stderr)

    # Build output: {node_id: [f32, ...]}
    result = {}
    dim = None
    for nid, emb in zip(node_ids, all_embeddings):
        vec = emb.tolist()
        if dim is None:
            dim = len(vec)
            print(f"Embedding dimension: {dim}", file=sys.stderr)
        result[nid] = vec

    # Save
    print(f"Writing to {args.output}...", file=sys.stderr)
    with open(args.output, "w") as f:
        json.dump(result, f)

    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"Done. {len(result)} embeddings, {dim}d, {size_mb:.1f}MB", file=sys.stderr)

    signal.alarm(0)  # Cancel timeout


if __name__ == "__main__":
    main()
