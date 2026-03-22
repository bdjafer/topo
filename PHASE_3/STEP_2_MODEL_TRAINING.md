# Step 2: Model Implementation & Training

This step implements the R-GIN architecture, all training objectives, and the training loop in Python/PyTorch. The model is trained on the dataset prepared in Step 1. The output is a trained model checkpoint, ONNX export, and trained artifacts (R matrix, depth probe weights).

**Note:** Several alternative architectures were considered and dropped during design. See PHASE_3.md §"What Was Dropped and Why" for rationale on: WL-Hash Contrastive, Interferometer/Gated Fusion, Learned Margins, Learnable Graph Refinement, Hyperbolic Geometry, Curriculum Contrastive, and Structural Edit Prediction. Do not re-propose these without reading the drop rationale.

---

## 1. Architecture: R-GIN

### Summary

R-GIN = Relation-typed Graph Isomorphism Network. 2-layer GNN with separate MLPs per edge type (calls, imports, inherits), plus SignNet for spectral PEs. Architecture rationale: see PHASE_3.md §Architecture.

### Input Features

For each node v, the model receives 5 feature groups (computed and exported in Step 0):

| Feature | Raw dim | Processed dim | Processing |
|---------|---------|---------------|------------|
| Semantic (CodeLM) | 768 | 128 | Learned MLP: 768→256→128 (ReLU) |
| Spectral PE | 16×2 = 32 | 32 | SignNet (see §2) |
| RWPE | 16 | 16 | Pass-through (already invariant) |
| Defines-tree | 4 | 16 | Learned MLP: 4→32→16 (ReLU) |
| Node type | 1 (categorical) | 16 | Learned embedding table |

**Total input: 208d** (128 + 32 + 16 + 16 + 16). Projected to hidden dim 256 via linear layer at entry.

### Model Specification

```
Hidden dim:      256
Layers:          2
Edge types:      3 (calls, imports, inherits)
Activation:      ReLU
Dropout:         0.1 (between layers)
Normalization:   GraphNorm per layer (not BatchNorm — batch mixes graphs of vastly different sizes)
Residual:        h^(l+1) += h^(l) between layers
Grad clip:       max_norm=1.0
```

### Per-Layer Processing

For each layer l and each edge type r ∈ {calls, imports, inherits}:

```
h_r^(l+1)(v) = MLP_r^(l)( (1 + ε_r^(l)) · h^(l)(v) + Σ_{u ∈ N_r(v)} h^(l)(u) )
```

where:
- `MLP_r^(l)` is a 2-layer MLP (256→256, ReLU, 256→256) specific to relation r and layer l
- `ε_r^(l)` is a learnable scalar per relation and layer
- `N_r(v)` is the set of neighbors of v in the edge type r subgraph

The per-relation outputs are **concatenated then projected** (not summed):

```
h^(l+1)(v) = W_agg^(l) · [h_calls^(l+1)(v) ‖ h_imports^(l+1)(v) ‖ h_inherits^(l+1)(v)] + b_agg^(l)
```

W_agg ∈ R^{256×768} projects 3×256 = 768d back to 256d. Then:

```
h^(l+1)(v) = GraphNorm(h^(l+1)(v))
h^(l+1)(v) = ReLU(h^(l+1)(v))
h^(l+1)(v) = Dropout(h^(l+1)(v), p=0.1)
h^(l+1)(v) += h^(l)(v)    # residual connection
```

### Output Decomposition

After the final GIN layer, decompose into 4 components:

```
z_calls(v)    = W_calls · h_calls^(final)(v)       # 256→32, from calls-only GIN output
z_imports(v)  = W_imports · h_imports^(final)(v)    # 256→32, from imports-only GIN output
z_inherits(v) = W_inherits · h_inherits^(final)(v) # 256→32, from inherits-only GIN output
z_invariant(v) = W_inv · h^(final)(v)              # 256→64, from post-aggregation hidden

z_str(v) = z_invariant(v) ⊕ z_calls(v) ⊕ z_imports(v) ⊕ z_inherits(v)
           64d              32d           32d             32d        = 160d
```

**Critical:** Per-layer components (z_calls, z_imports, z_inherits) are projected from the per-relation GIN outputs **BEFORE** the concat+project aggregation. This ensures each component is derived purely from its relation type.

**z_invariant source:** z_invariant is projected from the post-concat+project hidden state `h`. PHASE_3.md says "summed" (line 87) but also "concatenated then projected" (line 64). We follow line 64.

**Per-relation purity caveat:** The spec claims per-layer components are "derived purely from its relation type." This is only exactly true for a 1-layer model. In a 2-layer model, the second layer's per-relation GIN takes the aggregated `h` as input, which already contains cross-relation information from layer 1. The decomposition captures a per-relation *emphasis*, not a pure separation. This is acceptable — the HSIC regularizer encourages separation, and the emphasis is strong enough for downstream per-layer analysis.

**Zero-edge-type batches:** If a graph has zero inherits edges, exclude z_inherits from the HSIC loss for that graph. Compute HSIC only over (z_inv, z_calls) + (z_inv, z_imports).

### Graph-Level Embedding

```
g_embedding = AttentionPool(z_invariant)   # 64d
```

Attention-weighted mean pooling: `g = Σ_v α_v · z_invariant(v)` where `α_v = softmax(w^T z_invariant(v))` and w is a learned 64d vector.

---

## 2. SignNet for Spectral PE

### Problem

Eigenvectors have sign ambiguity: if v is an eigenvector, so is -v. Across different graphs (or different runs of the eigensolver), the sign choice is arbitrary. Training on raw eigenvectors would learn sign-dependent features that don't transfer.

### Solution: SignNet (Lim et al., ICML 2022)

For each eigenvector i (i = 1..16):

```
φ_i(v) = ρ(ψ_i([u_i(v), λ_i]) + ψ_i([-u_i(v), λ_i]))
```

where:
- `[u_i(v), λ_i]` is the 2d input: eigenvector component + eigenvalue
- `ψ_i` is a per-eigenvector MLP: 2→64→2 (ReLU activation, one hidden layer)
- The sign-invariant trick: process both `+u_i(v)` and `-u_i(v)`, add results
- `ρ` aggregates across eigenvectors: concatenate all 16 φ_i outputs (16×2 = 32d)

**Why eigenvalue pairing:** λ₂=0.001 (near-disconnect) and λ₂=0.5 (gradual gradient) are qualitatively different even if eigenvector coordinates look similar. The eigenvalue provides structural scale.

**Zero-padded eigenvectors:** For graphs with fewer than 16 non-trivial eigenvectors, the zero-padded entries have u_i(v) = 0 and λ_i = 0. SignNet processes them but the MLP learns to output near-zero for these inputs — no structural information, no influence.

### Implementation

```python
class SignNet(nn.Module):
    def __init__(self, k: int = 16, hidden: int = 64, out_per_eig: int = 2):
        super().__init__()
        # Per-eigenvector MLPs: input is [eigvec_component, eigenvalue] = 2d
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2, hidden),
                nn.ReLU(),
                nn.Linear(hidden, out_per_eig),
            ) for _ in range(k)
        ])
        self.k = k
        self.out_dim = k * out_per_eig  # 16 * 2 = 32

    def forward(self, spectral_vecs: Tensor, spectral_vals: Tensor) -> Tensor:
        """
        spectral_vecs: [n, k]  — eigenvector components
        spectral_vals: [n, k]  — eigenvalues (per-node, component-specific)
        Returns: [n, 32]
        """
        outputs = []
        for i in range(self.k):
            x_pos = torch.stack([spectral_vecs[:, i], spectral_vals[:, i]], dim=-1)  # [n, 2]
            x_neg = torch.stack([-spectral_vecs[:, i], spectral_vals[:, i]], dim=-1)  # [n, 2]
            out = self.mlps[i](x_pos) + self.mlps[i](x_neg)  # sign-invariant
            outputs.append(out)
        return torch.cat(outputs, dim=-1)  # [n, k * out_per_eig]
```

---

## 3. Training Objectives

Three losses plus one regularizer. Budget determined by PHASE_3.md — 6+ losses cause gradient competition; 3+1 is the maximum that trains reliably.

### Loss 1: Masked Semantic Feature Prediction (Primary)

**Objective:** Mask a node's CodeLM embedding, predict it from structural context.

```
L_reconstruct = (1/|M|) · Σ_{v ∈ M} (1 - cos(f_predict(v), x_sem(v)))
```

where:
- M = masked nodes (60-70% of nodes per graph, sampled uniformly per batch)
- x_sem(v) = frozen CodeLM embedding (768d, raw — NOT the 128d projection)
- f_predict(v) = MLP_decode(z_str(v)): 160→512 (ReLU) → 768

**Masking implementation:** Replace the semantic input of masked nodes with a learnable `[MASK]` token (a single 768d vector, learned end-to-end). The semantic MLP processes this token like any other input. This is strictly better than zero-masking: the model can learn to distinguish "masked" from "zero embedding."

**Masking ratio schedule:** Start at 60%, increase to 70% if validation reconstruction loss plateaus for 10 epochs. Higher masking forces reliance on graph structure rather than neighbor feature leakage.

### Loss 2: Asymmetric Cross-Layer Edge Prediction (Secondary)

**Objective:** Using only import-layer structural position, predict which call edges exist.

```
L_crosslayer = BCE(y_uv, σ(z_imports(u)^T · R · z_imports(v)))
```

where:
- z_imports(u), z_imports(v) are 32d import-layer embeddings
- R is a learnable 32×32 bilinear matrix
- y_uv = 1 if call edge (u,v) exists, else 0
- Negative sampling: 5:1 ratio (5 random non-edges per positive edge)

**Both sides use z_imports.** The question is "can import position predict call edges?" — not cross-space alignment. R learns which pairs of import-positions tend to co-occur with call edges.

**Negative sampling:** For each positive call edge (u,v), sample 5 negative pairs (u, v') where v' is a random node NOT connected to u by a call edge. Sample v' uniformly from the graph — more sophisticated strategies (degree-weighted, hard negatives) add complexity without proven benefit at this scale.

### Loss 3: Graph-Level Contrastive (Tertiary)

**Objective:** Subgraph samples from the same repo should have similar graph-level embeddings; samples from different repos should differ.

```
L_graph = -log( exp(sim(g_i, g_i') / τ) / Σ_j exp(sim(g_i, g_j) / τ) )
```

where:
- g_i, g_i' = graph embeddings of two BFS subgraph samples from the same repo
- g_j = graph embeddings from other repos in the batch (negative examples)
- sim = cosine similarity
- τ = 0.07 (temperature)

**Subgraph sampling:** For each graph in the batch, sample two overlapping subgraphs via BFS from different random start nodes, each containing 60-80% of nodes. Done at training time (see Step 1 §Data Augmentation).

### Regularizer: HSIC Decorrelation

```
L_decorrelation = HSIC(z_invariant, z_calls) + HSIC(z_invariant, z_imports) + HSIC(z_invariant, z_inherits)
```

**HSIC (Hilbert-Schmidt Independence Criterion)** is computed per-graph, not per-batch:

```python
def hsic(X: Tensor, Y: Tensor) -> Tensor:
    """Biased HSIC estimator with median bandwidth heuristic.
    X: [n, d1], Y: [n, d2]. Returns scalar."""
    n = X.shape[0]
    if n < 5:
        return torch.tensor(0.0, device=X.device)

    # RBF kernel matrices
    K_X = rbf_kernel(X)  # [n, n]
    K_Y = rbf_kernel(Y)  # [n, n]

    # Centering matrix H = I - (1/n) 11^T
    # HSIC = (1/n²) tr(K_X H K_Y H)
    H = torch.eye(n, device=X.device) - 1.0 / n
    HK_X = H @ K_X
    HK_Y = H @ K_Y
    return ((HK_X * HK_Y.T).sum() / (n * n)).clamp(min=0)  # biased estimator can go negative

def rbf_kernel(X: Tensor) -> Tensor:
    """RBF kernel with median bandwidth heuristic."""
    dists = torch.cdist(X, X, p=2)  # [n, n]
    # Use upper triangle only (exclude diagonal zeros — Gretton 2012)
    upper = dists[torch.triu(torch.ones(dists.shape, dtype=torch.bool), diagonal=1)]
    sigma = upper.median().clamp(min=1e-5)
    return torch.exp(-dists.pow(2) / (2 * sigma.pow(2)))
```

**Computed per-graph:** For each graph in the batch, compute HSIC over that graph's nodes (n ≈ 500-1000). Average across graphs. Per-batch HSIC would require kernel matrices over 16K-32K nodes — infeasible.

**Non-negativity:** The biased HSIC estimator can produce small negative values for finite samples (unlike the unbiased estimator which is strictly non-negative in expectation). Clamp to max(0, HSIC) before adding to the loss to prevent negative regularization.

### Combined Loss

```
L_total = L_reconstruct + α · L_crosslayer + β · L_graph + λ · L_decorrelation
```

| Weight | Target value | Ramp schedule |
|--------|-------------|---------------|
| L_reconstruct | 1.0 | Full weight from epoch 1 |
| α (L_crosslayer) | 0.5 | 0 → 0.5 linear over epochs 10-30 |
| β (L_graph) | 0.2 | 0 → 0.2 linear over epochs 10-30 |
| λ (L_decorrelation) | 0.01 | 0 → 0.01 linear over epochs 10-30 |

---

## 4. Training Loop

### Design Invariants

**Masking scope:** The mask token replaces ONLY the semantic input (768d CodeLM embedding). Spectral PEs, RWPE, tree features, and node type are NEVER masked. This is load-bearing — the model must have positional/structural features for masked nodes so it can predict their semantics from structural position. Masking all features would remove the signal the model needs to reconstruct from.

**Feature normalization:** Only tree features need normalization: apply `log1p` in the data loader (Step 1) to compress raw integers (0-5000) to ~0-8.5. All other features are naturally bounded.

**Random seeds:** Set torch/numpy/CUDA seeds at training start. Record seed in `config.json`.

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | AdamW | Standard for GNN pre-training |
| Learning rate | 1e-3 | Standard for GIN |
| LR schedule | Linear warmup (0→1e-3, epochs 1-10) + constant at 1e-3 (epochs 10-30) + cosine decay (1e-3→1e-5, epochs 30-200) |
| Weight decay | 1e-4 | Standard regularization |
| Batch size | 32 graphs | Fits in GPU memory for median graph sizes |
| Epochs | 200 (max) | Early stopping may trigger earlier |
| Masking ratio | 0.65 (start), 0.70 (if plateau) | Higher = more structural reliance |
| Negative sampling ratio | 5:1 | Standard for link prediction |
| Temperature τ | 0.07 | Standard for contrastive learning |
| Grad clip | max_norm=1.0 | Prevents spikes when auxiliary losses activate |
| Dropout | 0.1 | Between GIN layers |
| SignNet hidden | 64 | Per-eigenvector MLP |
| Early stopping patience | 30 epochs | On validation masked reconstruction cosine similarity |

### Training Schedule

```
Epochs 1-10:   LR warmup (0 → 1e-3 linearly).
               Only L_reconstruct at full weight.
               Auxiliary weights = 0.
               LR held constant at warmup target after reaching it.

Epochs 10-30:  LR constant at 1e-3 (no decay yet).
               Auxiliary weights ramp linearly to target:
               L_crosslayer: 0 → 0.5
               L_graph: 0 → 0.2
               L_decorrelation: 0 → 0.01

Epochs 30-200: Full training. All losses at target weights.
               Cosine LR decay from 1e-3 to 1e-5.
               (Note: PHASE_3.md says 300 epochs; we use 200 as a
               tighter budget. Adjust if early stopping never triggers.)

Early stopping: If validation masked reconstruction cosine similarity
                has not improved for 30 epochs, stop.
```

### Per-Epoch Pseudocode

```python
def train_epoch(model, loader, optimizer, epoch, config):
    model.train()
    total_loss = 0

    # Compute loss weight ramps
    alpha = ramp(epoch, start=10, end=30, target=0.5)
    beta = ramp(epoch, start=10, end=30, target=0.2)
    lam = ramp(epoch, start=10, end=30, target=0.01)
    mask_ratio = 0.65  # increase to 0.70 if plateau

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # 1. Generate mask (per-graph)
        mask = generate_mask(batch, ratio=mask_ratio)

        # 2. Forward pass
        z_str, z_inv, z_calls, z_imports, z_inherits, g_emb = model(batch, mask)

        # 3. Loss 1: Masked reconstruction
        predictions = model.decode(z_str[mask])
        targets = batch["node"].x_semantic[mask]  # raw 768d
        loss_recon = 1 - F.cosine_similarity(predictions, targets, dim=-1).mean()

        # 4. Loss 2: Cross-layer edge prediction
        pos_edges = batch["node", "calls", "node"].edge_index
        neg_edges = sample_negatives(batch, pos_edges, ratio=5)
        loss_cross = cross_layer_loss(z_imports, pos_edges, neg_edges, model.R)

        # 5. Loss 3: Graph contrastive
        loss_graph = graph_contrastive_loss(batch, g_emb, tau=0.07)

        # 6. Regularizer: HSIC decorrelation
        loss_decorr = per_graph_hsic(batch, z_inv, z_calls, z_imports, z_inherits)

        # 7. Combined
        loss = loss_recon + alpha * loss_cross + beta * loss_graph + lam * loss_decorr

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)
```

### Mask Generation

```python
def generate_mask(batch, ratio: float = 0.65) -> Tensor:
    """Generate a boolean mask (True = masked) per node, respecting per-graph boundaries."""
    masks = []
    ptr = batch.ptr  # graph boundary pointers from PyG batching
    for i in range(len(ptr) - 1):
        start, end = ptr[i].item(), ptr[i+1].item()
        n = end - start
        n_mask = int(n * ratio)
        perm = torch.randperm(n, device=batch.x_semantic.device)
        graph_mask = torch.zeros(n, dtype=torch.bool, device=batch.x_semantic.device)
        graph_mask[perm[:n_mask]] = True
        masks.append(graph_mask)
    return torch.cat(masks)
```

### Negative Sampling for Cross-Layer Loss

```python
def sample_negatives(batch, pos_edges: Tensor, ratio: int = 5) -> Tensor:
    """Sample negative edges for cross-layer prediction.
    For each positive call edge (u, v), sample `ratio` random non-call-neighbor nodes
    FROM THE SAME GRAPH (respecting batch boundaries)."""
    n_neg = pos_edges.shape[1] * ratio
    device = pos_edges.device

    # For each positive source, sample targets within the same graph
    src = pos_edges[0].repeat(ratio)
    graph_ids = batch.batch[src]  # which graph each source belongs to

    # Sample random targets per graph
    tgt = torch.zeros(n_neg, dtype=torch.long, device=device)
    for g_id in graph_ids.unique():
        mask = graph_ids == g_id
        g_start, g_end = batch.ptr[g_id], batch.ptr[g_id + 1]
        tgt[mask] = torch.randint(g_start, g_end, (mask.sum(),), device=device)

    return torch.stack([src, tgt])
```

### Checkpointing

```python
# Save checkpoint every 10 epochs + best model
if val_metric > best_val_metric:
    best_val_metric = val_metric
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_metric": val_metric,
        "config": config,
    }, "checkpoints/best_model.pt")
```

---

## 5. Model Implementation

### Full Module Structure

```python
# packages/topo-model/src/topo_model/rgin.py

class RGIN(nn.Module):
    """Relation-typed Graph Isomorphism Network for code structure learning."""

    def __init__(self, config: RGINConfig):
        super().__init__()
        self.config = config
        H = config.hidden_dim  # 256

        # Input projections
        self.sem_project = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 128))
        self.sign_net = SignNet(k=16, hidden=64, out_per_eig=2)  # 32d output
        self.tree_project = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 16))
        self.type_embed = nn.Embedding(12, 16)  # 12 node types

        # Input → hidden projection: 208d → 256d
        self.input_project = nn.Linear(128 + 32 + 16 + 16 + 16, H)

        # Learnable mask token — initialized to zeros (CodeLM embeddings are ~unit-norm;
        # randn(768) has magnitude ~27 which is wildly out-of-distribution)
        self.mask_token = nn.Parameter(torch.zeros(768))

        # GIN layers (2 layers × 3 edge types)
        self.gin_layers = nn.ModuleList()
        self.agg_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.epsilons = nn.ParameterList()

        for l in range(config.n_layers):
            per_rel = nn.ModuleDict()
            per_eps = nn.ParameterDict()
            for rel in ["calls", "imports", "inherits"]:
                per_rel[rel] = nn.Sequential(
                    nn.Linear(H, H), nn.ReLU(), nn.Linear(H, H)
                )
                per_eps[rel] = nn.Parameter(torch.zeros(1))
            self.gin_layers.append(per_rel)
            self.epsilons.append(per_eps)
            self.agg_layers.append(nn.Linear(3 * H, H))
            self.norms.append(GraphNorm(H))

        # Output projections
        self.proj_invariant = nn.Linear(H, 64)
        self.proj_calls = nn.Linear(H, 32)
        self.proj_imports = nn.Linear(H, 32)
        self.proj_inherits = nn.Linear(H, 32)

        # Decode head (160d → 768d)
        self.decoder = nn.Sequential(nn.Linear(160, 512), nn.ReLU(), nn.Linear(512, 768))

        # Cross-layer bilinear matrix R (32×32)
        self.R = nn.Parameter(torch.randn(32, 32) * 0.01)

        # Graph-level attention pooling
        self.pool_attn = nn.Linear(64, 1)

    def forward(self, batch, mask: Tensor):
        # --- Input processing ---
        x_sem = batch["node"].x_semantic.float()  # [N, 768]

        # Apply mask: replace masked nodes' semantic input with mask token
        x_sem_masked = x_sem.clone()
        x_sem_masked[mask] = self.mask_token

        x_sem_proj = self.sem_project(x_sem_masked)          # [N, 128]
        x_spectral = self.sign_net(
            batch["node"].x_spectral_vecs,
            batch["node"].x_spectral_vals
        )                                                      # [N, 32]
        x_rwpe = batch["node"].x_rwpe.float()                 # [N, 16]
        x_tree = self.tree_project(batch["node"].x_tree.float())  # [N, 16]
        x_type = self.type_embed(batch["node"].x_type.long())     # [N, 16]

        x = torch.cat([x_sem_proj, x_spectral, x_rwpe, x_tree, x_type], dim=-1)  # [N, 208]
        h = self.input_project(x)  # [N, 256]

        # --- GIN message passing ---
        h_per_rel_final = {}
        for l in range(self.config.n_layers):
            h_rels = []
            for rel in ["calls", "imports", "inherits"]:
                edge_key = ("node", rel, "node")
                if edge_key in batch.edge_types and batch[edge_key].edge_index.shape[1] > 0:
                    edge_index = batch[edge_key].edge_index
                    # GIN aggregation: (1 + eps) * h(v) + sum(h(neighbors))
                    eps = self.epsilons[l][rel]
                    # Neighbor sum via scatter
                    neighbor_sum = torch.zeros_like(h)
                    neighbor_sum.scatter_add_(0, edge_index[1].unsqueeze(1).expand(-1, h.shape[1]), h[edge_index[0]])
                    h_rel = self.gin_layers[l][rel]((1 + eps) * h + neighbor_sum)
                else:
                    # No edges for this type: self-loop only
                    eps = self.epsilons[l][rel]
                    h_rel = self.gin_layers[l][rel]((1 + eps) * h)

                h_rels.append(h_rel)

                # Save final layer per-rel outputs for output decomposition
                if l == self.config.n_layers - 1:
                    h_per_rel_final[rel] = h_rel

            # Concat + project
            h_cat = torch.cat(h_rels, dim=-1)  # [N, 768]
            h_new = self.agg_layers[l](h_cat)   # [N, 256]
            h_new = self.norms[l](h_new, batch.batch)  # GraphNorm per-graph
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.config.dropout, training=self.training)
            h_new = h_new + h  # residual
            h = h_new

        # --- Output decomposition ---
        z_invariant = self.proj_invariant(h)                        # [N, 64]
        z_calls = self.proj_calls(h_per_rel_final["calls"])         # [N, 32]
        z_imports = self.proj_imports(h_per_rel_final["imports"])    # [N, 32]
        z_inherits = self.proj_inherits(h_per_rel_final["inherits"]) # [N, 32]
        z_str = torch.cat([z_invariant, z_calls, z_imports, z_inherits], dim=-1)  # [N, 160]

        # --- Graph-level embedding ---
        attn_scores = self.pool_attn(z_invariant).squeeze(-1)  # [N]
        # Softmax per graph using batch assignment
        attn_weights = scatter_softmax(attn_scores, batch.batch, dim=0)  # [N]
        g_emb = scatter_sum(attn_weights.unsqueeze(-1) * z_invariant, batch.batch, dim=0)  # [B, 64]

        return z_str, z_invariant, z_calls, z_imports, z_inherits, g_emb

    def decode(self, z_str: Tensor) -> Tensor:
        """Decode structural embedding to predict semantic features."""
        return self.decoder(z_str)  # [*, 160] → [*, 768]

    def reconstruction_error(self, z_str: Tensor, x_sem: Tensor) -> Tensor:
        """Compute per-node reconstruction error (cosine distance)."""
        predicted = self.decode(z_str)
        return 1 - F.cosine_similarity(predicted, x_sem, dim=-1)
```

### GraphNorm

```python
class GraphNorm(nn.Module):
    """Graph-level normalization (Cai et al., 2021).
    Normalizes per graph, not per batch — critical when batch mixes
    100-node CLIs with 5000-node monorepos."""
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.alpha = nn.Parameter(torch.ones(1))  # learnable mean subtraction weight

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        # Per-graph mean and std
        mean = scatter_mean(x, batch, dim=0)[batch]  # [N, d]
        x = x - self.alpha * mean
        var = scatter_mean(x.pow(2), batch, dim=0)[batch]  # [N, d]
        std = (var + 1e-6).sqrt()
        return self.gamma * x / std + self.beta
```

---

## 6. GIN Message Passing Direction

**Critical design choice:** The GIN formula specifies `Σ_{u ∈ N_r(v)} h(u)` — sum over neighbors of v in edge type r. For **directed** code graphs, "neighbor" must be precisely defined.

In topo's edge format: an edge (u, v) means "u depends on v" (u calls v, u imports v). The R-GIN's `edge_index` follows PyG convention: `edge_index[0]` = source, `edge_index[1]` = target.

The scatter in the forward pass: `scatter_add_(0, edge_index[1], h[edge_index[0]])` aggregates **source** node features at each **target** node. For a call edge (caller→callee), the **callee** receives information from callers. For an import edge (importer→imported), the **imported module** receives information from importers.

**This is the correct direction for structural role learning.** A node's role is largely defined by who depends on it (its "consumers"). A utility function's role is defined by having many callers. An entry point's role is defined by having no callers but many callees. The R-GIN learns these patterns from the incoming-message perspective.

**Bidirectional alternative:** To also capture outgoing dependencies (what a node depends on), add reversed edges as a separate processing step per layer:

```python
# For each edge type, also process reversed edges
neighbor_sum_forward = scatter_add(h[edge_index[0]], edge_index[1], dim=0, dim_size=n)
neighbor_sum_reverse = scatter_add(h[edge_index[1]], edge_index[0], dim=0, dim_size=n)
neighbor_sum = neighbor_sum_forward + neighbor_sum_reverse
```

**Recommendation:** Start with forward-only (simpler, matches GIN spec). If perturbation sensitivity (Step 3, Tier 3) is weak, add bidirectional as the first ablation. The cost is 2x message passing but no new parameters.

---

## 7. Post-Training Artifacts

### R Matrix Extraction

After training, extract the 32×32 bilinear matrix R from `model.R`:

```python
R = model.R.detach().cpu().numpy()  # [32, 32]

# Check asymmetry
asymmetry = np.linalg.norm(R - R.T) / np.linalg.norm(R)
print(f"R asymmetry ratio: {asymmetry:.4f}")
# If < 0.1: R learned proximity but not directionality.
# Health score falls back to binary violation counting.
# If >= 0.1: direction_surprise signal is usable.
```

### Semantic Depth Probe

Fit a linear probe mapping module semantic centroids to layer position:

```python
def fit_depth_probe(model, train_dataset) -> tuple[np.ndarray, float]:
    """Post-training: linear probe from CodeLM centroids to layer position."""
    centroids = []  # [n_modules, 768]
    depths = []     # [n_modules]

    for data in train_dataset:
        # Skip repos with cycles (need clean DAG structure)
        if data.metadata.get("cycle_freedom", 0) < 0.95:
            continue

        # Compute per-module CodeLM centroid
        modules = data.metadata["modules"]
        for mod in modules:
            member_embeddings = data.x_semantic[mod["member_indices"]]  # [k, 768]
            centroid = member_embeddings.mean(dim=0).numpy()
            centroids.append(centroid)
            depths.append(mod["normalized_layer_position"])

    X = np.stack(centroids)  # [M, 768]
    y = np.array(depths)     # [M]

    # OLS: w = (X^T X)^{-1} X^T y
    w, _, _, _ = np.linalg.lstsq(
        np.column_stack([X, np.ones(len(X))]),
        y, rcond=None
    )
    depth_probe_w = w[:-1]  # [768]
    depth_probe_b = w[-1]    # scalar

    return depth_probe_w, depth_probe_b
```

### Model Bundle

The trained model is packaged as a bundle containing:

```
model_bundle/
  rgin.onnx                # ONNX-exported model (or rgin_weights.npz for native Rust)
  R.npy                    # 32×32 bilinear matrix
  depth_probe_w.npy        # 768d weight vector
  depth_probe_b.npy        # scalar bias
  config.json              # Model hyperparameters (hidden_dim, n_layers, etc.)
  metadata.json            # Training info (epochs, best val metric, R asymmetry, etc.)
  node_type_vocab.json     # Frozen vocabulary (must match Step 0's NODE_TYPE_VOCAB)
```

---

## 8. ONNX Export

### Why ONNX

The model trains in PyTorch but must run in Rust for inference. ONNX (Open Neural Network Exchange) is the bridge. The `ort` crate (ONNX Runtime for Rust) is already a dependency of topo via `fastembed-rs`.

### Export Procedure

```python
import torch.onnx

# Dummy inputs for tracing
n = 100
dummy_batch = create_dummy_batch(n=n)
dummy_mask = torch.zeros(n, dtype=torch.bool)

torch.onnx.export(
    model,
    (dummy_batch, dummy_mask),
    "rgin.onnx",
    input_names=["x_semantic", "x_spectral_vecs", "x_spectral_vals",
                 "x_rwpe", "x_tree", "x_type",
                 "edge_index_calls", "edge_index_imports", "edge_index_inherits",
                 "mask"],
    output_names=["z_str", "z_invariant", "z_calls", "z_imports", "z_inherits", "g_emb"],
    dynamic_axes={
        "x_semantic": {0: "num_nodes"},
        "x_spectral_vecs": {0: "num_nodes"},
        # ... all node-level tensors are dynamic on dim 0
        "edge_index_calls": {1: "num_call_edges"},
        # ... all edge tensors are dynamic on dim 1
    },
    opset_version=17,
)
```

### ONNX Export Risks

**High risk:** PyG's batching, scatter operations, and graph-level pooling use custom kernels that may not export cleanly to ONNX.

**Mitigations:**
1. Write an `OnnxExportWrapper` that replaces all PyG scatter ops with pure PyTorch equivalents (`torch.zeros().index_add_()` instead of `scatter_add_`).
2. Replace `scatter_softmax` with manual softmax: compute per-graph max, subtract, exp, per-graph sum, divide.
3. Replace GraphNorm with a simpler per-graph normalization using `batch` index.
4. Test the exported ONNX model against PyTorch outputs for 5 test graphs. Require max absolute error < 1e-5.
5. If any of steps 1-4 fail, immediately switch to the native Rust fallback (don't spend more than 2 days on ONNX).

**Fallback: Native Rust inference (~1500 lines).** If ONNX export proves too fragile, hand-implement the R-GIN forward pass in Rust. The model is small (1.8M params) and architecturally simple (linear layers, ReLU, sum aggregation). Weight loading from NPZ. This also enables WASM portability (ONNX Runtime does not compile to WASM).

The ONNX vs native Rust decision is made in Step 3 based on export test results.

---

## 9. Parameter Count

```
GIN layers:       2 layers × 3 relations × 2-layer MLP (256→256→256)   ≈  790K
Concat+project:   2 layers × (768→256 linear)                          ≈  394K
SignNet:           16 eigenvectors × (2→64→2 MLP)                       ≈    4K
Projection heads:  256→64 + 3×(256→32)                                 ≈   41K
Decode head:       160→512→768                                          ≈  477K
Input projections: 768→256→128 (2-layer) + 4→32→16 + 12×16 embed       ≈  230K
Misc:              mask token, R matrix, pool attention, norms          ≈   10K
─────────────────────────────────────────────────────────────────────────────
Total:                                                                  ≈ 1.95M
```

Data-to-parameter ratio: ~350:1 (500K-1M training nodes / 1.95M params). Comfortably outside overfitting regime with self-supervised masking generating O(n) examples per epoch with different masks.

---

## 10. Intrinsic Evaluation (During Training)

Computed every 5 epochs on the validation set:

| Metric | Target | What it measures |
|--------|--------|------------------|
| Masked reconstruction cosine similarity | > 0.6 | Structure predicts semantics |
| Cross-layer AUC (imports → calls) | > 0.75 | Import structure predicts call structure |
| Graph-level contrastive accuracy | > 0.8 | Same-repo subgraphs more similar than cross-repo |
| R asymmetry ratio | > 0.1 | R learned directionality |
| HSIC(z_inv, z_calls) | Decreasing trend | Decorrelation working |

### Validation Loop

```python
def validate(model, val_loader) -> dict:
    model.eval()
    recon_sims = []
    crosslayer_preds, crosslayer_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            mask = generate_mask(batch, ratio=0.65)
            z_str, z_inv, z_calls, z_imports, z_inherits, g_emb = model(batch, mask)

            # Reconstruction similarity (on masked nodes only)
            pred = model.decode(z_str[mask])
            target = batch["node"].x_semantic[mask]
            sim = F.cosine_similarity(pred, target, dim=-1).mean()
            recon_sims.append(sim.item())

            # Cross-layer AUC (skip if no call edges in this batch)
            call_key = ("node", "calls", "node")
            if call_key in batch.edge_types and batch[call_key].edge_index.shape[1] > 0:
                pos_edges = batch[call_key].edge_index
                neg_edges = sample_negatives(batch, pos_edges, ratio=5)
                pos_scores = cross_layer_score(z_imports, pos_edges, model.R)
                neg_scores = cross_layer_score(z_imports, neg_edges, model.R)
                crosslayer_preds.extend(pos_scores.tolist() + neg_scores.tolist())
                crosslayer_labels.extend([1]*len(pos_scores) + [0]*len(neg_scores))

    return {
        "recon_cosine_sim": np.mean(recon_sims),
        "crosslayer_auc": roc_auc_score(crosslayer_labels, crosslayer_preds) if crosslayer_labels else None,
    }
```

---

## 11. Package Structure

```
packages/topo-model/
  pyproject.toml
  src/topo_model/
    __init__.py
    rgin.py            # R-GIN model class
    signnet.py         # SignNet module
    losses.py          # All loss functions + HSIC
    train.py           # Training loop
    validate.py        # Validation + intrinsic metrics
    export.py          # ONNX export + artifact extraction
    config.py          # RGINConfig dataclass
    depth_probe.py     # Post-training semantic depth probe
  scripts/
    train.sh           # Training launch script
    export_bundle.py   # Package model bundle
  tests/
    test_rgin.py       # Forward pass shape tests
    test_signnet.py    # Sign invariance property test
    test_losses.py     # Loss computation correctness
    test_hsic.py       # HSIC is zero for independent inputs
    test_export.py     # ONNX round-trip accuracy
```

---

## 12. Hardware Requirements

Single GPU (RTX 3090+ or equivalent). ~4-8 hours training for 1000 graphs × 200 epochs. GPU memory < 1 GB per batch.

---

## 13. Definition of Done

- [ ] R-GIN forward pass produces correct output shapes for variable-size batches.
- [ ] SignNet is sign-invariant: `SignNet(v, λ) == SignNet(-v, λ)` for all inputs.
- [ ] All 3 losses + HSIC compute correctly on test inputs.
- [ ] Training loop runs for 5 epochs on a small subset (10 graphs) without NaN/inf.
- [ ] Loss weights ramp correctly (check at epochs 1, 15, 30, 50).
- [ ] Validation metrics are computed and logged.
- [ ] Early stopping triggers correctly.
- [ ] Best model checkpoint is saved and loadable.
- [ ] R matrix asymmetry is reported.
- [ ] Depth probe fits and produces sensible predictions.
- [ ] ONNX export produces valid .onnx file (or native Rust fallback decision documented).
- [ ] Model bundle contains all required artifacts.
- [ ] Intrinsic metrics meet targets on validation set:
  - Reconstruction cosine sim > 0.6
  - Cross-layer AUC > 0.75
  - Graph contrastive accuracy > 0.8
