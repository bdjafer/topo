"""Export trained R-GIN model artifacts as a model bundle."""

import json
from pathlib import Path

import numpy as np
import torch

from topo_model.config import RGINConfig
from topo_model.rgin import RGIN


def export_bundle(
    checkpoint_path: Path,
    output_dir: Path,
    node_type_vocab: dict[str, int] | None = None,
) -> Path:
    """Export a trained model checkpoint as a model bundle.

    Bundle contents:
    - best_model.pt: PyTorch checkpoint
    - R.npy: 32×32 bilinear matrix
    - config.json: Model hyperparameters
    - metadata.json: Training info
    - node_type_vocab.json: Frozen vocabulary

    Args:
        checkpoint_path: path to best_model.pt
        output_dir: where to write the bundle
        node_type_vocab: node type vocabulary (if None, uses default)

    Returns:
        Path to output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Reconstruct config
    config_dict = ckpt.get("config", {})
    if isinstance(config_dict, dict) and config_dict:
        config = RGINConfig(**{k: v for k, v in config_dict.items() if k in RGINConfig.__dataclass_fields__})
    else:
        config = RGINConfig()

    # Build model and load weights
    model = RGIN(config)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # --- Extract R matrix ---
    R = model.R.detach().cpu().numpy()
    np.save(output_dir / "R.npy", R)

    # R asymmetry analysis
    r_norm = np.linalg.norm(R)
    r_asymmetry = float(np.linalg.norm(R - R.T) / r_norm) if r_norm > 1e-8 else 0.0

    # --- Save config ---
    config.save(output_dir / "config.json")

    # --- Save checkpoint ---
    torch.save(ckpt, output_dir / "best_model.pt")

    # --- Save node type vocabulary ---
    if node_type_vocab is None:
        # Must match Rust NODE_TYPE_VOCAB in topo-analyzer/src/types.rs
        # DO NOT reorder — indices are baked into trained model weights
        node_type_vocab = {
            "function": 0,   # behavioral unit, coupling endpoint
            "module": 1,     # container, import hub
            "class": 2,      # concrete type (struct, enum, class, dataclass)
            "interface": 3,  # abstract contract (trait, interface, protocol)
            "unknown": 4,    # UNKNOWN_TYPE_INDEX fallback
        }
    with open(output_dir / "node_type_vocab.json", "w") as f:
        json.dump(node_type_vocab, f, indent=2)

    # --- Save metadata ---
    metadata = {
        "epoch": ckpt.get("epoch", -1),
        "val_metric": ckpt.get("val_metric", -1),
        "R_asymmetry": r_asymmetry,
        "R_norm": float(r_norm),
        "n_params": sum(p.numel() for p in model.parameters()),
        "architecture": "R-GIN",
        "hidden_dim": config.hidden_dim,
        "n_layers": config.n_layers,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model bundle exported to {output_dir}/")
    print(f"  R asymmetry: {r_asymmetry:.4f} ({'directional' if r_asymmetry >= 0.1 else 'symmetric'})")
    print(f"  Params: {metadata['n_params']:,}")

    return output_dir
