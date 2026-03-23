"""R-GIN model configuration."""

from dataclasses import dataclass, field, asdict
import json
from pathlib import Path


@dataclass
class RGINConfig:
    # Architecture
    hidden_dim: int = 256
    n_layers: int = 2
    edge_types: list[str] = field(default_factory=lambda: ["calls", "imports", "inherits"])
    n_node_types: int = 5  # 4 canonical types + 1 unknown (matches Rust NODE_TYPE_VOCAB)
    dropout: float = 0.1

    # SignNet
    spectral_k: int = 16
    signnet_hidden: int = 64
    signnet_out_per_eig: int = 2

    # Input dimensions (fixed by Step 0 feature export)
    semantic_dim: int = 768
    rwpe_dim: int = 16
    tree_dim: int = 4
    type_embed_dim: int = 16

    # Derived input dimension: 128 + 32 + 16 + 16 + 16 = 208
    # After projection to hidden_dim

    # Output decomposition
    invariant_dim: int = 64
    per_relation_dim: int = 32
    # z_str = 64 + 3*32 = 160

    # Training
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 200
    batch_size: int = 32
    grad_clip: float = 1.0

    # Loss weights (targets after ramp)
    alpha_crosslayer: float = 0.5
    beta_graph: float = 0.2
    lambda_decorr: float = 0.01

    # Ramp schedule
    ramp_start: int = 10
    ramp_end: int = 30

    # Masking
    mask_ratio: float = 0.65
    mask_ratio_plateau: float = 0.70

    # Contrastive
    temperature: float = 0.07
    neg_ratio: int = 5

    # Early stopping
    patience: int = 30

    # LR schedule
    warmup_epochs: int = 10
    decay_start: int = 30
    lr_min: float = 1e-5

    # Checkpointing
    save_every: int = 10
    eval_every: int = 5

    @property
    def input_dim(self) -> int:
        """Total input dimension before projection to hidden_dim."""
        sem_proj = 128
        spectral = self.spectral_k * self.signnet_out_per_eig  # 32
        return sem_proj + spectral + self.rwpe_dim + self.type_embed_dim + self.type_embed_dim
        # 128 + 32 + 16 + 16 + 16 = 208

    @property
    def z_str_dim(self) -> int:
        """Structural embedding dimension."""
        return self.invariant_dim + len(self.edge_types) * self.per_relation_dim
        # 64 + 3*32 = 160

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "RGINConfig":
        with open(path) as f:
            return cls(**json.load(f))
