"""R-GIN: Relation-typed Graph Isomorphism Network for code structure learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from topo_model.config import RGINConfig
from topo_model.signnet import SignNet


class GraphNorm(nn.Module):
    """Graph-level normalization (Cai et al., 2021).

    Normalizes per graph, not per batch — critical when batch mixes
    100-node CLIs with 5000-node monorepos.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.alpha = nn.Parameter(torch.ones(1))  # learnable mean subtraction weight

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        if x.shape[0] == 0:
            return x
        # Per-graph mean
        # scatter_mean: compute mean per graph, then index back to each node
        num_graphs = batch.max().item() + 1
        count = torch.zeros(num_graphs, dtype=x.dtype, device=x.device)
        count.scatter_add_(0, batch, torch.ones(x.size(0), dtype=x.dtype, device=x.device))
        count = count.clamp(min=1)

        sum_x = torch.zeros(num_graphs, x.size(1), dtype=x.dtype, device=x.device)
        sum_x.scatter_add_(0, batch.unsqueeze(1).expand_as(x), x)
        mean = sum_x / count.unsqueeze(1)

        x = x - self.alpha * mean[batch]

        # Per-graph variance
        sum_sq = torch.zeros(num_graphs, x.size(1), dtype=x.dtype, device=x.device)
        sum_sq.scatter_add_(0, batch.unsqueeze(1).expand_as(x), x.pow(2))
        var = sum_sq / count.unsqueeze(1)
        std = (var + 1e-6).sqrt()

        return self.gamma * x / std[batch] + self.beta


class RGIN(nn.Module):
    """Relation-typed Graph Isomorphism Network for code structure learning.

    2-layer GNN with separate MLPs per edge type (calls, imports, inherits),
    plus SignNet for spectral PEs. Produces per-node structural embeddings
    and graph-level embeddings.
    """

    def __init__(self, config: RGINConfig):
        super().__init__()
        self.config = config
        H = config.hidden_dim  # 256

        # --- Input projections ---
        self.sem_project = nn.Sequential(
            nn.Linear(config.semantic_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        self.sign_net = SignNet(
            k=config.spectral_k,
            hidden=config.signnet_hidden,
            out_per_eig=config.signnet_out_per_eig,
        )
        self.tree_project = nn.Sequential(
            nn.Linear(config.tree_dim, 32),
            nn.ReLU(),
            nn.Linear(32, config.type_embed_dim),
        )
        self.type_embed = nn.Embedding(config.n_node_types, config.type_embed_dim)

        # Input → hidden: 208d → 256d
        input_dim = 128 + self.sign_net.out_dim + config.rwpe_dim + config.type_embed_dim + config.type_embed_dim
        self.input_project = nn.Linear(input_dim, H)

        # Learnable mask token — initialized to zeros
        # (CodeLM embeddings are ~unit-norm; randn(768) would be wildly OOD)
        self.mask_token = nn.Parameter(torch.zeros(config.semantic_dim))

        # --- GIN layers ---
        self.gin_layers = nn.ModuleList()
        self.agg_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.epsilons = nn.ParameterList()

        for _l in range(config.n_layers):
            per_rel = nn.ModuleDict()
            per_eps = nn.ParameterDict()
            for rel in config.edge_types:
                per_rel[rel] = nn.Sequential(
                    nn.Linear(H, H),
                    nn.ReLU(),
                    nn.Linear(H, H),
                )
                per_eps[rel] = nn.Parameter(torch.zeros(1))
            self.gin_layers.append(per_rel)
            self.epsilons.append(per_eps)
            self.agg_layers.append(nn.Linear(len(config.edge_types) * H, H))
            self.norms.append(GraphNorm(H))

        # --- Output projections ---
        self.proj_invariant = nn.Linear(H, config.invariant_dim)
        self.proj_calls = nn.Linear(H, config.per_relation_dim)
        self.proj_imports = nn.Linear(H, config.per_relation_dim)
        self.proj_inherits = nn.Linear(H, config.per_relation_dim)

        # Decode head (160d → 768d)
        self.decoder = nn.Sequential(
            nn.Linear(config.z_str_dim, 512),
            nn.ReLU(),
            nn.Linear(512, config.semantic_dim),
        )

        # Cross-layer bilinear matrix R (32×32)
        self.R = nn.Parameter(torch.randn(config.per_relation_dim, config.per_relation_dim) * 0.01)

        # Graph-level attention pooling
        self.pool_attn = nn.Linear(config.invariant_dim, 1)

    def forward(self, batch, mask: Tensor):
        """Forward pass through R-GIN.

        Args:
            batch: PyG HeteroData batch with node features and edge indices
            mask: [N] boolean tensor, True = masked nodes

        Returns:
            z_str: [N, 160] structural embedding
            z_invariant: [N, 64] invariant component
            z_calls: [N, 32] calls component
            z_imports: [N, 32] imports component
            z_inherits: [N, 32] inherits component
            g_emb: [B, 64] graph-level embedding
        """
        # --- Input processing ---
        x_sem = batch["node"].x_semantic.float()  # [N, 768]

        # Apply mask: replace masked nodes' semantic input with mask token
        x_sem_masked = x_sem.clone()
        x_sem_masked[mask] = self.mask_token

        x_sem_proj = self.sem_project(x_sem_masked)  # [N, 128]
        x_spectral = self.sign_net(
            batch["node"].x_spectral_vecs.float(),
            batch["node"].x_spectral_vals.float(),
        )  # [N, 32]
        x_rwpe = batch["node"].x_rwpe.float()  # [N, 16]
        x_tree = self.tree_project(batch["node"].x_tree.float())  # [N, 16]
        x_type = self.type_embed(batch["node"].x_type.long())  # [N, 16]

        x = torch.cat([x_sem_proj, x_spectral, x_rwpe, x_tree, x_type], dim=-1)  # [N, 208]
        h = self.input_project(x)  # [N, 256]

        # --- GIN message passing ---
        # Get batch assignment vector
        batch_idx = batch["node"].batch  # [N] graph assignment

        h_per_rel_final = {}
        for l in range(self.config.n_layers):
            h_rels = []
            for rel in self.config.edge_types:
                edge_key = ("node", rel, "node")
                has_edges = (
                    edge_key in batch.edge_types
                    and batch[edge_key].edge_index.shape[1] > 0
                )

                eps = self.epsilons[l][rel]
                if has_edges:
                    edge_index = batch[edge_key].edge_index
                    # GIN: (1 + eps) * h(v) + sum(h(neighbors))
                    # scatter source features to target nodes
                    neighbor_sum = torch.zeros_like(h)
                    neighbor_sum.scatter_add_(
                        0,
                        edge_index[1].unsqueeze(1).expand(-1, h.shape[1]),
                        h[edge_index[0]],
                    )
                    h_rel = self.gin_layers[l][rel]((1 + eps) * h + neighbor_sum)
                else:
                    # No edges for this type: self-loop only
                    h_rel = self.gin_layers[l][rel]((1 + eps) * h)

                h_rels.append(h_rel)

                # Save final layer per-rel outputs for output decomposition
                if l == self.config.n_layers - 1:
                    h_per_rel_final[rel] = h_rel

            # Concat + project
            h_cat = torch.cat(h_rels, dim=-1)  # [N, 3*H]
            h_new = self.agg_layers[l](h_cat)  # [N, H]
            h_new = self.norms[l](h_new, batch_idx)  # GraphNorm
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.config.dropout, training=self.training)
            h_new = h_new + h  # residual
            h = h_new

        # --- Output decomposition ---
        z_invariant = self.proj_invariant(h)  # [N, 64]
        z_calls = self.proj_calls(h_per_rel_final["calls"])  # [N, 32]
        z_imports = self.proj_imports(h_per_rel_final["imports"])  # [N, 32]
        z_inherits = self.proj_inherits(h_per_rel_final["inherits"])  # [N, 32]
        z_str = torch.cat([z_invariant, z_calls, z_imports, z_inherits], dim=-1)  # [N, 160]

        # --- Graph-level embedding ---
        attn_scores = self.pool_attn(z_invariant).squeeze(-1)  # [N]
        # Softmax per graph
        num_graphs = batch_idx.max().item() + 1
        attn_max = torch.full((num_graphs,), float("-inf"), device=h.device)
        attn_max.scatter_reduce_(0, batch_idx, attn_scores, reduce="amax", include_self=True)
        attn_shifted = attn_scores - attn_max[batch_idx]
        attn_exp = attn_shifted.exp()
        attn_sum = torch.zeros(batch_idx.max() + 1, device=h.device)
        attn_sum.scatter_add_(0, batch_idx, attn_exp)
        attn_weights = attn_exp / attn_sum[batch_idx].clamp(min=1e-8)

        # Weighted sum per graph
        weighted = attn_weights.unsqueeze(-1) * z_invariant  # [N, 64]
        g_emb = torch.zeros(num_graphs, self.config.invariant_dim, device=h.device)
        g_emb.scatter_add_(0, batch_idx.unsqueeze(1).expand_as(weighted), weighted)

        return z_str, z_invariant, z_calls, z_imports, z_inherits, g_emb

    def decode(self, z_str: Tensor) -> Tensor:
        """Decode structural embedding to predict semantic features."""
        return self.decoder(z_str)  # [*, 160] → [*, 768]
