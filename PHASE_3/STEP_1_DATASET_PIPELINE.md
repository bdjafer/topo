# Step 1: Dataset Pipeline

This step builds the infrastructure to curate, parse, embed, and preprocess 500-2000 GitHub repositories into R-GIN training data. The output is a directory of NPZ feature files (one per repo) ready for PyTorch Geometric data loading.

This is the calendar-time bottleneck of Phase 3. PHASE_3.md calls preprocessing "the most underestimated step." Parsing 1000 repos with topo's parsers + computing CodeLM embeddings is I/O and compute heavy. The pipeline must be **resumable**, **parallelizable**, and **auditable**.

**Hard prerequisite:** Step 0 (Codebase Prep) must be complete before this step can execute. The `topo export-features` command, `rwpe.rs`, `tree.rs`, and `spectral_pe_export()` do not exist yet — they are created in Step 0. Without them, the per-repo pipeline (§3) cannot produce NPZ files.

**Embedding mode mandate:** ALL repos must be processed with the SAME embedding mode (either Mode A: CLI-integrated via `topo export-features`, or Mode B: Python fastembed). Mixing modes produces subtly different embeddings due to ONNX runtime version differences, corrupting the training distribution. **Mode A is the default.** Mode B is the fallback only if the CLI semantic feature is unavailable for the entire dataset run.

---

## Current state (as of 2026-03-22)

Steps 1-2 (curation and checkout/parse) are **partially implemented**:

- **`examples/registry.toml`** is the single source of truth for all repos. Currently 50 repos (22 Python, 28 Rust) with pinned commits. This replaces the previously planned `repos.jsonl`.
- **`benchmark/scripts/harvest_corpus.py`** clones and parses repos, writing `graph.json` to `examples/<name>/`. Resumable (skips cached). Uses the pre-built `topo` binary.
- **`examples/scripts/fetch_and_analyze.sh`** does the full pipeline (clone → parse → analyze) for curated examples.
- **12 repos are parsed** with cached `graph.json` files. 2 (ripgrep, tantivy) have full analysis artifacts committed.

**Not yet implemented:** embedding pipeline, NPZ export orchestration, quality filters, train/val/test splits, PyG data loading (`topo-dataset` package).

---

## 1. Repository Curation

### Source: `examples/registry.toml`

The registry is the single manifest for all repos. Each entry is an `[[example]]` block:

```toml
[[example]]
name = "flask"
repo = "https://github.com/pallets/flask"
commit = "2f8c0e3"   # pinned SHA for reproducibility
ref = "3.1.0"         # git tag (for cloning)
language = "python"
entrypoint = "src/flask"
description = "Python micro web framework."

[example.tags]
size = "small"
quality = "clean"
pattern = ["layered", "sansio", "plugin-system"]

[example.cli_overrides]
exclude = "testing"   # dirs to skip during parsing
```

To scale to 500-2000 repos, add entries to registry.toml. Repos without tags are valid — tags are for curation metadata, not pipeline requirements.

### Scaling via automated curation

To reach 500+ repos, build `packages/topo-dataset/scripts/curate_repos.py` that queries the GitHub API and appends qualifying repos to `registry.toml`.

### Filter Criteria

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Stars | ≥ 100 | Filters homework, dead projects, trivial forks |
| Contributors OR active years | ≥ 50 contributors OR ≥ 3 years active (commits spanning 3+ calendar years) | Filters personal/toy projects |
| CI/CD | GitHub Actions, Travis, CircleCI, or similar present | Proxy for code quality and maintenance |
| Primary language | Python OR Rust | Matches topo's parsers |
| Fork | Excluded (only source repos) | Deduplication |
| License | OSI-approved | Legal clarity for training data |
| Graph size | 50 ≤ nodes ≤ 50,000 after parsing | Too small = no structure; too large = memory/time issues |
| Archived | Excluded | Dead codebases have stale structure |

### Stratification

To ensure architectural diversity, target these proportions:

| Style | Target % | Identification heuristic |
|-------|----------|--------------------------|
| Web application (Django, Flask, FastAPI, Actix, Rocket) | 25% | Framework dependency in requirements.txt / Cargo.toml |
| Library / framework | 25% | PyPI/crates.io published, no web framework dep |
| CLI tool | 15% | Entry point binary, `clap`/`argparse`/`click` dependency |
| Data pipeline / ML | 15% | `pandas`/`numpy`/`sklearn`/`torch` dependency |
| Systems / infrastructure | 10% | `tokio`/`async-std`, low-level crates, no web framework |
| Monorepo / multi-package | 10% | Multiple `Cargo.toml` or `setup.py` files |

**Stratification is approximate.** A repo may fit multiple categories. Use the primary framework dependency as the tiebreaker. The goal is coverage, not precise percentages.

### Parser Language Coverage

**Rust repos:** Parsed by `topo-parser-rust` (native, uses rust-analyzer HIR). Extracts all 4 edge types (calls, imports, inherits, defines). High fidelity.

**Python repos:** Parsed by the Python parser subprocess bridge (`topo-parser` dispatches to `python -m topo_parser_python`). The Python parser must be installed and available. It extracts the same edge types but relies on AST analysis (no semantic engine equivalent to rust-analyzer). Edge extraction fidelity is lower for dynamic dispatch, metaclasses, and decorators.

**Implication for dataset:** Expect ~5-10% higher parse failure rate for Python repos vs Rust. The quality filter (Step 1 §4) catches degenerate parse results.

**Python parser path issue:** The Python parser may emit absolute file paths in `node.file` fields. The embedding pipeline's source text extraction expects relative paths (relative to repo root). The preprocessing script must normalize paths to relative before embedding. Check: `if os.path.isabs(node.file): node.file = os.path.relpath(node.file, repo_path)`.

**Python parser dependency:** The Python parser requires `pycg` for inter-procedural call graph analysis. Without it, only AST-level call extraction is available (weaker signal). Ensure `pycg` is installed in the preprocessing environment: `pip install pycg`.

**Memory warning for Rust parsing:** The Rust parser loads the full workspace into rust-analyzer, consuming 4-8GB RAM for large projects (tokio, serde). With `--workers 4`, peak memory can hit 16-32GB. Set `--workers 2` for Rust repos or process them sequentially. The preprocessing script should detect language and adjust parallelism accordingly.

---

## 2. Repository Checkout & Parse

### Already implemented

`benchmark/scripts/harvest_corpus.py` handles clone + parse:

```bash
# Parse specific repos
make harvest REPOS="flask click requests"

# Parse all 50 registered repos
make harvest-all

# Force re-parse (ignore cache)
python3 benchmark/scripts/harvest_corpus.py flask --force
```

Clones go to `/tmp/topo-corpus/` (ephemeral). Parsed graphs go to `examples/<name>/graph.json` (cached, re-used by all downstream consumers).

### Pinning

Commit SHAs are pinned in `registry.toml`. The harvest script checks out the exact commit before parsing. Repos with `commit = "HEAD"` use the latest default branch — these should be pinned to a specific SHA before a dataset release.

### Workspace Layout

```
examples/
  registry.toml              # Single manifest (replaces repos.jsonl)
  flask/
    graph.json               # Parsed graph (from harvest)
    analysis.json            # Full analysis (from fetch_and_analyze.sh, optional)
    features.npz             # Exported features (from preprocess.py, Step 1)
    features.meta.json       # Feature metadata sidecar
    embeddings.json          # Semantic embeddings (optional)
  ripgrep/
    graph.json
    analysis.json
    ...
  splits/                    # Train/val/test splits (from split.py, Step 1)
    train.txt
    val.txt
    test.txt
  quality_report.json        # Preprocessing quality metrics
```

All per-repo artifacts live alongside the repo's `graph.json` in `examples/<name>/`. This keeps the cache layer unified — downstream consumers (`benchmark/`, `packages/topo-dataset/`) read from `examples/` and never write to it.

---

## 3. Batch Preprocessing Pipeline

### Overview

For each repo, the pipeline:
1. **Parse** → `graph.json` *(already implemented via harvest_corpus.py)*
2. **Embed** → 768d CodeLM vectors per node
3. **Analyze** → Spectral decomposition, modules, roles
4. **Export** → NPZ + metadata (using `topo export-features` from Step 0)

### Orchestrator

```python
# packages/topo-dataset/scripts/preprocess.py

"""
Batch preprocessing: embed and export features for all parsed repos.

Usage:
    python preprocess.py --workers 4
    python preprocess.py --resume            # skip already-processed
    python preprocess.py --repos flask,click  # specific repos only
"""
```

The orchestrator reads `examples/registry.toml` for the repo list and checks `examples/<name>/graph.json` exists (must be parsed first via `make harvest`).

### Per-Repo Pipeline

```python
def process_repo(name: str, examples_dir: Path) -> dict:
    """Process one repo: embed + export. Returns quality metrics or error."""
    repo_dir = examples_dir / name
    graph_path = repo_dir / "graph.json"
    output_npz = repo_dir / "features.npz"
    output_meta = repo_dir / "features.meta.json"

    # Skip if already processed (resume mode)
    if output_npz.exists() and output_meta.exists():
        return {"repo": name, "status": "skipped"}

    if not graph_path.exists():
        return {"repo": name, "status": "not_parsed"}

    try:
        with open(graph_path) as f:
            graph = json.load(f)

        n_nodes = len(graph["nodes"])
        if n_nodes < 50:
            return {"repo": name, "status": "too_small", "n_nodes": n_nodes}
        if n_nodes > 50000:
            return {"repo": name, "status": "too_large", "n_nodes": n_nodes}

        # Export features (calls topo export-features, which handles
        # spectral PE, RWPE, tree features, and optionally embeddings)
        run_topo_export(graph_path, output_npz)

        # Validate output
        validate_npz(output_npz, n_nodes)

        return {"repo": name, "status": "ok", "n_nodes": n_nodes}

    except Exception as e:
        return {"repo": name, "status": "error", "error": str(e)}
```

### Parallelization

Use Python `multiprocessing.Pool` or `concurrent.futures.ProcessPoolExecutor` with `--workers N`.

**Note:** The CodeLM embedding step (fastembed/ONNX) is CPU-bound and uses internal parallelism (ONNX Runtime threads). Set `ORT_NUM_THREADS=2` per worker to avoid oversubscription. With 4 workers × 2 ORT threads = 8 CPU cores utilized.

**Note:** The Rust `topo` binary is called as a subprocess. Each worker gets its own process. No shared state, no locking issues.

### Embedding Computation

Two modes:

**Mode A: CLI-integrated (preferred)**
The `topo export-features <path>` command handles parsing + embedding + export in one shot. This uses the `--features semantic` build of topo-cli which includes fastembed.

**Mode B: Separate embedding step (fallback)**
If the CLI semantic feature is not available (e.g., WASM build), compute embeddings separately:

```python
from fastembed import TextEmbedding

model = TextEmbedding(model_name="jinaai/jina-embeddings-v2-base-code")

def compute_embeddings(graph: dict, repo_path: Path) -> dict[str, list[float]]:
    """Compute CodeLM embeddings for all nodes."""
    texts = []
    node_ids = []
    for node in graph["nodes"]:
        source_text = extract_source_text(node, repo_path)
        texts.append(source_text)
        node_ids.append(node["id"])

    # Batch embed
    embeddings = model.embed(texts)
    return dict(zip(node_ids, [emb.tolist() for emb in embeddings]))
```

**Source text extraction:** For each node, assemble a context window following PHASE_2.md §Embedding Input:
```
# module: <enclosing_module_path>
# file: <file_path>

<function signature + body up to ~6K tokens>
```

This requires the node's file path, line number, and line_end (or byte span) from the parsed graph. The parser must provide span information. If spans are missing, fall back to the node ID string as input (weaker signal but non-fatal).

### Resumability

The pipeline must be **idempotent and resumable**:

- Each repo produces independent files in `examples/<name>/` (features.npz, features.meta.json).
- If files already exist, skip (`--resume` flag).
- If processing fails, log the error and continue to the next repo.
- A `examples/quality_report.json` summarizes all outcomes (ok, skipped, too_small, too_large, error).

**Error handling:** Log and skip failures. Record status per repo in `quality_report.json`.

---

## 4. Quality Filters

After preprocessing, filter the dataset to remove degenerate graphs that would hurt training.

### Per-Graph Quality Checks

| Check | Threshold | Rationale |
|-------|-----------|-----------|
| Node count | ≥ 50 | Too small for meaningful module structure |
| Edge count (coupling only) | ≥ n/2 | Extremely sparse graphs have no coupling signal |
| Largest component ratio | ≥ 0.5 | Heavily fragmented graphs have weak spectral signal |
| Spectral k actual | ≥ 2 | Must have at least 2 non-trivial eigenvectors |
| Semantic embedding coverage | ≥ 80% of nodes embedded | Missing embeddings corrupt masked reconstruction |
| Non-degenerate spectral | silhouette > 0.0 or not degenerate by `is_degenerate()` | Package-fallback graphs contribute no spectral learning signal |

### Per-Graph Quality Metrics (recorded, not filtered)

| Metric | Purpose |
|--------|---------|
| modularity_q | Downstream quality indicator |
| fiedler_value | Connectivity strength |
| semantic_smoothness | Phase 2 Rayleigh quotient (if available) |
| semantic_structural_ami | Phase 2 AMI (if available) |
| edge_type_distribution | {calls: %, imports: %, inherits: %} — detect missing edge types |
| node_type_distribution | {function: %, class: %, ...} — detect parser anomalies |

### Post-Filter Dataset Size

Expect ~10-20% loss from quality filters (parse failures, too small, degenerate). Curate 600-2500 repos to end up with 500-2000 usable training graphs.

---

## 5. Train/Val/Test Split

### Split Ratios

| Set | % | Purpose |
|-----|---|---------|
| Train | 80% | Model training |
| Val | 10% | Early stopping, hyperparameter tuning |
| Test | 10% | Final evaluation, never touched during training |

### Stratification

Split is **stratified by architectural style** (from `[example.tags]` in registry.toml) to ensure each set has representative coverage. Use scikit-learn's `StratifiedShuffleSplit` or manual proportional sampling.

### Constraints

- **No data leakage:** If a repo has multiple versions (e.g., flask v2 and flask v3), all versions go to the same split.
- **Test set stability:** Once assigned, the test set is frozen. New repos added later go to train or val only.
- **Repo-level, not graph-level:** The split is at the repository level. All subgraphs from the same repo are in the same split.

### Output

```
examples/splits/train.txt    # One repo name per line: "flask"
examples/splits/val.txt
examples/splits/test.txt
```

---

## 6. PyTorch Geometric Data Loading

### NPZ → PyG HeteroData Conversion

```python
# packages/topo-dataset/src/topo_dataset/loader.py

import numpy as np
import torch
from torch_geometric.data import HeteroData

def load_graph(repo_dir: Path) -> HeteroData:
    """Load a preprocessed graph as a PyG HeteroData object."""
    npz_path = repo_dir / "features.npz"
    meta_path = repo_dir / "features.meta.json"

    arrays = np.load(npz_path)
    meta = json.loads(meta_path.read_text())

    data = HeteroData()

    # Node features
    data["node"].x_semantic = torch.from_numpy(arrays["semantic"])           # [n, 768]
    # Spectral PEs stored as two flat arrays; keep separate for SignNet input
    data["node"].x_spectral_vecs = torch.from_numpy(arrays["spectral_vecs"]).float() # [n, 16]
    data["node"].x_spectral_vals = torch.from_numpy(arrays["spectral_vals"]).float() # [n, 16]
    # Note: SignNet (Step 2) receives these as two separate tensors and pairs them
    # internally per eigenvector: [u_i(v), λ_i]. NOT reshaped to [n, 16, 2] here.
    data["node"].x_rwpe = torch.from_numpy(arrays["rwpe"])                   # [n, 16]
    data["node"].x_tree = torch.from_numpy(arrays["tree_features"]).float()
    data["node"].x_tree = torch.log1p(data["node"].x_tree)                   # [n, 4] log-compressed
    data["node"].x_type = torch.from_numpy(arrays["node_types"])             # [n]

    # Edges (homogeneous node type, heterogeneous edge types)
    for edge_type in ["calls", "imports", "inherits"]:
        key = f"edge_index_{edge_type}"
        if key in arrays:
            edge_index = torch.from_numpy(arrays[key]).long()  # [2, m]
            data["node", edge_type, "node"].edge_index = edge_index

    # Metadata
    data.repo = meta.get("repo", repo_dir.name)
    data.n_nodes = meta["n_nodes"]
    data.node_ids = meta.get("node_ids", [])
    data.metadata = meta

    return data
```

### Dataset Class

```python
# packages/topo-dataset/src/topo_dataset/dataset.py

from torch_geometric.data import Dataset

class TopoDataset(Dataset):
    def __init__(self, examples_dir: Path, split_file: Path, transform=None):
        self.repo_names = split_file.read_text().strip().split("\n")
        self.examples_dir = examples_dir
        super().__init__(root=str(examples_dir), transform=transform)

    def len(self) -> int:
        return len(self.repo_names)

    def get(self, idx: int) -> HeteroData:
        name = self.repo_names[idx]
        return load_graph(self.examples_dir / name)
```

### Batching

PyG's `DataLoader` handles variable-size graph batching via index offsetting. Each mini-batch is a single large disconnected graph with batch assignment vectors.

```python
from torch_geometric.loader import DataLoader

train_dataset = TopoDataset(examples_dir, examples_dir / "splits" / "train.txt")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

**Batch size considerations:**
- Median graph: ~500-1000 nodes. Batch of 32 = ~16K-32K nodes.
- HSIC decorrelation is computed **per-graph** (kernel matrices per graph, not per batch). See PHASE_3.md §Decorrelation.
- Graph contrastive loss (Loss 3) needs subgraph pairs. Sampling happens inside the training loop, not during data loading.

---

## 7. Edge Type Handling

### Missing Edge Types

Some repos may lack certain edge types entirely:
- **No inherits edges:** Python repos with no classes, or Rust repos with no trait impls.
- **No imports edges:** Unlikely but possible in very small single-file projects.

For repos missing an edge type, the corresponding R-GIN branch sees **empty neighborhoods** — every node's aggregation over that layer is zero. This is mathematically valid (GIN reduces to `h_r^(l+1)(v) = MLP_r((1+ε) · h(v))`, a self-loop-only update). But if many training graphs are missing a layer, the per-layer MLP doesn't learn useful patterns.

**Mitigation:**
- Record edge type distribution per graph in metadata.
- Ensure training set has ≥ 80% of graphs with all 3 edge types.
- For the ~20% missing inherits: the model learns that "no inherits neighbors" is a valid structural signal (most functions don't participate in inheritance).

### Edge Direction

The parsed graph has directed edges. The R-GIN processes them as directed (message passing follows edge direction). The RWPE uses symmetrized edges (§Step 0). These are different by design — R-GIN captures directed coupling patterns, RWPE captures undirected local topology.

---

## 8. Scale Estimates

Expect ~10-30 GB disk total, ~3-5 hours preprocessing for 1000 repos (dominated by CodeLM embedding at ~50s per 1K-node repo).

---

## 9. Reproducibility

Pin repo commit SHAs in `examples/registry.toml`, topo binary version, and CodeLM model version. **Test set is frozen** — new repos go to train/val only. Incremental updates via `preprocess.py --resume`.

---

## 10. Package Structure

```
packages/topo-dataset/
  pyproject.toml           # Python package config (uv/pip)
  scripts/
    curate_repos.py        # GitHub search → append to registry.toml
    preprocess.py          # Batch embed + export (reads examples/*/graph.json)
    split.py               # Train/val/test split generation
    validate.py            # Post-processing quality checks
  src/topo_dataset/
    __init__.py
    loader.py              # NPZ → PyG HeteroData (reads examples/<name>/)
    dataset.py             # PyG Dataset class
    transforms.py          # Data augmentation (subgraph sampling, etc.)
    utils.py               # Shared utilities
  tests/
    test_loader.py         # Round-trip load test
    test_dataset.py        # Dataset iteration test
    test_transforms.py     # Augmentation correctness
```

### Dependencies

```toml
[project]
dependencies = [
    "torch>=2.0",
    "torch-geometric>=2.4",
    "numpy>=1.24",
    "fastembed>=0.2",      # For Mode B embedding (fallback only)
    "pycg>=0.0.7",         # Required by Python parser for inter-procedural call graph
    "scikit-learn>=1.3",   # For stratified splits, NMI computation
]
```

**Pre-download model:** Before batch preprocessing, download the CodeLM model once:
```bash
python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='jinaai/jina-embeddings-v2-base-code')"
```
This prevents race conditions when multiple workers attempt concurrent downloads.

---

## 11. Data Augmentation (for Training)

### Subgraph Sampling (Loss 3: Graph Contrastive)

For each repo in a batch, sample two overlapping subgraphs (PHASE_3.md §Loss 3):

```python
def sample_subgraph(data: HeteroData, ratio: float = 0.7) -> HeteroData:
    """Sample a connected subgraph containing `ratio` of nodes via BFS from a random start."""
    n = data.num_nodes
    target_size = int(n * ratio)
    start = torch.randint(0, n, (1,)).item()

    # BFS expansion until target_size reached
    visited = {start}
    frontier = [start]
    while len(visited) < target_size and frontier:
        next_frontier = []
        for node in frontier:
            for edge_type in ["calls", "imports", "inherits"]:
                edge_key = ("node", edge_type, "node")
                if edge_key in data.edge_types:
                    edges = data[edge_key].edge_index
                    neighbors = edges[1][edges[0] == node].tolist()
                    neighbors += edges[0][edges[1] == node].tolist()  # undirected BFS
                    for nb in neighbors:
                        if nb not in visited and len(visited) < target_size:
                            visited.add(nb)
                            next_frontier.append(nb)
        frontier = next_frontier

    # Induce subgraph
    return data.subgraph(torch.tensor(sorted(visited)))
```

### Node Feature Masking (Loss 1: Masked Reconstruction)

Masking is applied during training, not during data loading. The mask ratio (60-70%) is a training hyperparameter. See Step 2 for details.

---

## 12. Definition of Done

- [x] Registry manifest with 50+ repos, pinned commits. *(examples/registry.toml — 50 repos)*
- [x] Clone + parse pipeline, resumable. *(harvest_corpus.py)*
- [ ] `curate_repos.py` scales registry to 600-2500 repos via GitHub API.
- [ ] `preprocess.py` processes repos end-to-end: embed → export → NPZ.
- [ ] Pipeline is parallelized (`--workers N`).
- [ ] Quality filters remove degenerate graphs.
- [ ] `quality_report.json` records per-repo status and metrics.
- [ ] `split.py` produces stratified train/val/test splits.
- [ ] `TopoDataset` loads NPZ files and returns valid PyG `HeteroData`.
- [ ] `DataLoader` batches correctly (test with batch_size=4, verify index offsets).
- [ ] ≥ 500 usable training graphs after quality filtering.
- [ ] ≥ 80% of training graphs have all 3 edge types (calls, imports, inherits).
- [ ] All scripts documented with `--help`.
- [ ] Reproducibility: pinned versions, checksums, frozen test set.
