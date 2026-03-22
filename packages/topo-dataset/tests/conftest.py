"""Shared test fixtures for topo-dataset tests."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def synthetic_repo(tmp_path: Path):
    """Create a synthetic repo directory with features.npz and features.meta.json.

    Produces a small graph with 10 nodes, 3 edge types, and known shapes.
    """
    n = 10
    k = 16
    embed_dim = 768

    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()

    # Create NPZ
    npz_path = repo_dir / "features.npz"
    np.savez(
        npz_path,
        semantic=np.random.randn(n, embed_dim).astype(np.float32),
        spectral_vecs=np.random.randn(n, k).astype(np.float32),
        spectral_vals=np.random.randn(n, k).astype(np.float32),
        rwpe=np.random.rand(n, k).astype(np.float32),
        tree_features=np.array([
            [0, 0, 10, 0],   # root
            [1, 0, 4, 10],
            [1, 1, 3, 10],
            [2, 0, 1, 4],
            [2, 1, 1, 4],
            [2, 0, 1, 3],
            [2, 1, 1, 3],
            [0, 0, 1, 0],    # orphan
            [0, 0, 1, 0],
            [0, 0, 1, 0],
        ], dtype=np.int32),
        node_types=np.array([0, 1, 2, 0, 0, 3, 0, 0, 1, 0], dtype=np.int32),
        edge_index_calls=np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32),
        edge_index_imports=np.array([[0, 0, 1], [1, 2, 3]], dtype=np.int32),
        edge_index_inherits=np.array([[5, 6], [2, 2]], dtype=np.int32),
    )

    # Create metadata sidecar
    meta = {
        "n_nodes": n,
        "n_edges": {
            "calls": 4,
            "imports": 3,
            "inherits": 2,
        },
        "node_ids": [f"node_{i}" for i in range(n)],
        "n_components": 2,
        "fiedler_value": 0.42,
    }
    (repo_dir / "features.meta.json").write_text(json.dumps(meta, indent=2))

    return repo_dir


@pytest.fixture
def synthetic_repo_no_inherits(tmp_path: Path):
    """Repo with no inherits edges (valid — some repos have none)."""
    n = 10
    k = 16
    embed_dim = 768

    repo_dir = tmp_path / "no_inherits_repo"
    repo_dir.mkdir()

    npz_path = repo_dir / "features.npz"
    np.savez(
        npz_path,
        semantic=np.zeros((n, embed_dim), dtype=np.float32),
        spectral_vecs=np.random.randn(n, k).astype(np.float32),
        spectral_vals=np.random.randn(n, k).astype(np.float32),
        rwpe=np.random.rand(n, k).astype(np.float32),
        tree_features=np.ones((n, 4), dtype=np.int32),
        node_types=np.zeros(n, dtype=np.int32),
        edge_index_calls=np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32),
        edge_index_imports=np.array([[0, 1], [4, 5]], dtype=np.int32),
        edge_index_inherits=np.zeros((2, 0), dtype=np.int32),  # empty
    )

    meta = {
        "n_nodes": n,
        "n_edges": {"calls": 3, "imports": 2, "inherits": 0},
        "node_ids": [f"node_{i}" for i in range(n)],
        "n_components": 1,
        "fiedler_value": 1.5,
    }
    (repo_dir / "features.meta.json").write_text(json.dumps(meta, indent=2))

    return repo_dir


@pytest.fixture
def multi_repo_dir(synthetic_repo, synthetic_repo_no_inherits, tmp_path: Path):
    """Directory containing multiple repos for dataset tests.

    Returns (examples_dir, split_file_path).
    """
    import shutil

    examples = tmp_path / "examples"
    examples.mkdir()

    shutil.copytree(synthetic_repo, examples / "repo_a")
    shutil.copytree(synthetic_repo_no_inherits, examples / "repo_b")

    split_file = examples / "train.txt"
    split_file.write_text("repo_a\nrepo_b\n")

    return examples, split_file
