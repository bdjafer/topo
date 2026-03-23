#!/usr/bin/env python3
"""Generate synthetic semantic embeddings for training validation.

Since CodeLM embeddings haven't been computed yet, we generate
structurally-correlated synthetic embeddings that give the
reconstruction loss something meaningful to learn from.

Strategy: Nodes that are structurally similar (connected, same type)
get similar embeddings. This creates a learnable signal that
tests whether the model can predict semantics from structure.
"""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples"


def generate_for_repo(
    repo_dir: Path,
    type_centroids: np.ndarray,
    spectral_proj: np.ndarray,
    tree_proj: np.ndarray,
    repo_seed: int = 42,
) -> None:
    """Generate synthetic semantic embeddings for one repo.

    Uses SHARED type centroids and projections across all repos
    so the model can generalize. Only per-node noise varies.
    """
    npz_path = repo_dir / "features.npz"
    meta_path = repo_dir / "features.meta.json"

    if not npz_path.exists():
        return

    meta = json.loads(meta_path.read_text())
    n = meta["n_nodes"]

    with np.load(npz_path) as arrays:
        data = dict(arrays)

    # 1. Base: shared type centroids (same across all repos)
    node_types = data["node_types"]
    semantic = type_centroids[node_types].copy()  # [n, 768]

    # 2. Spectral position influence (shared projection)
    spectral = data["spectral_vecs"]  # [n, 16]
    semantic += spectral @ spectral_proj  # [n, 768]

    # 3. Tree structure influence (shared projection)
    tree = data["tree_features"]  # [n, 4]
    semantic += tree @ tree_proj

    # 4. Per-node noise (varies per repo — not perfectly predictable)
    rng = np.random.RandomState(repo_seed)
    semantic += rng.randn(n, 768).astype(np.float32) * 0.15

    # 5. L2 normalize to unit norm (like real CodeLM embeddings)
    norms = np.linalg.norm(semantic, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    semantic = semantic / norms

    # Replace in NPZ
    data["semantic"] = semantic.astype(np.float32)
    np.savez_compressed(npz_path, **data)

    # Verify
    coverage = np.count_nonzero(np.linalg.norm(semantic, axis=1) > 1e-6)
    print(f"  {repo_dir.name}: {n} nodes, {coverage}/{n} non-zero embeddings")


def main():
    repos = sorted(
        d.name for d in EXAMPLES_DIR.iterdir()
        if d.is_dir() and (d / "features.npz").exists()
    )

    # Generate SHARED components with a fixed seed
    # These are the same across all repos — allows generalization
    shared_rng = np.random.RandomState(42)
    type_centroids = shared_rng.randn(12, 768).astype(np.float32) * 0.3
    spectral_proj = shared_rng.randn(16, 768).astype(np.float32) * 0.1
    tree_proj = shared_rng.randn(4, 768).astype(np.float32) * 0.05

    print(f"Generating synthetic embeddings for {len(repos)} repos...")
    print("  Using SHARED type centroids + projections (consistent across repos)")
    for i, name in enumerate(repos):
        generate_for_repo(
            EXAMPLES_DIR / name,
            type_centroids=type_centroids,
            spectral_proj=spectral_proj,
            tree_proj=tree_proj,
            repo_seed=42 + i,
        )

    print("\nDone. Synthetic embeddings are structurally correlated:")
    print("  - Similar node types → similar embeddings")
    print("  - Similar spectral positions → similar embeddings")
    print("  - Noise added for partial predictability")
    print("\nThis allows reconstruction loss to validate that")
    print("the model learns structure→semantics mapping.")


if __name__ == "__main__":
    main()
