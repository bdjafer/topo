"""NPZ + metadata.json -> PyG HeteroData conversion.

Reads the NPZ files produced by `topo export-features` and converts them
into PyTorch Geometric HeteroData objects ready for R-GIN consumption.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import HeteroData


def load_graph(repo_dir: Path) -> HeteroData:
    """Load a preprocessed graph as a PyG HeteroData object.

    Expects:
        repo_dir/features.npz       — arrays from topo export-features
        repo_dir/features.meta.json — metadata sidecar

    Returns a HeteroData with:
        data["node"].x_semantic       — [n, 768] float32
        data["node"].x_spectral_vecs  — [n, k]   float32
        data["node"].x_spectral_vals  — [n, k]   float32
        data["node"].x_rwpe           — [n, k]   float32
        data["node"].x_tree           — [n, 4]   float32 (log1p-compressed)
        data["node"].x_type           — [n]      int64
        data["node", edge_type, "node"].edge_index — [2, m] int64
    """
    npz_path = repo_dir / "features.npz"
    meta_path = repo_dir / "features.meta.json"

    meta = json.loads(meta_path.read_text())
    n = meta["n_nodes"]

    data = HeteroData()

    with np.load(npz_path) as arrays:
        # Node features
        semantic = arrays["semantic"]
        assert semantic.shape[0] == n, f"semantic has {semantic.shape[0]} rows, expected {n}"
        data["node"].x_semantic = torch.from_numpy(semantic.copy()).float()  # [n, 768]

        # Spectral PEs: kept separate for SignNet input (paired internally per eigenvector)
        data["node"].x_spectral_vecs = torch.from_numpy(arrays["spectral_vecs"].copy()).float()  # [n, k]
        data["node"].x_spectral_vals = torch.from_numpy(arrays["spectral_vals"].copy()).float()  # [n, k]

        data["node"].x_rwpe = torch.from_numpy(arrays["rwpe"].copy()).float()  # [n, k]

        # Tree features: log1p-compressed for scale normalization
        tree_raw = torch.from_numpy(arrays["tree_features"].copy()).float()  # [n, 4]
        data["node"].x_tree = torch.log1p(tree_raw)

        data["node"].x_type = torch.from_numpy(arrays["node_types"].copy()).long()  # [n]

        # Edges (homogeneous node type, heterogeneous edge types)
        for edge_type in ["calls", "imports", "inherits"]:
            key = f"edge_index_{edge_type}"
            if key in arrays:
                edge_index = torch.from_numpy(arrays[key].copy()).long()  # [2, m]
                if edge_index.numel() > 0:
                    data["node", edge_type, "node"].edge_index = edge_index

    data["node"].num_nodes = n

    # Metadata
    data.repo = meta.get("repo", repo_dir.name)
    data.n_nodes = n
    data.node_ids = meta.get("node_ids", [])
    data.graph_meta = meta

    return data
