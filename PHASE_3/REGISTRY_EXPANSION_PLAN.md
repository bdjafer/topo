# Registry Expansion Pipeline — Design Plan

Expand `examples/registry.toml` from 50 repos to 600+ usable training graphs (targeting 1000–1500 candidates to survive filtering). This is the dataset foundation for R-GIN model training (Step 2).

**Design for scale:** The pipeline is designed to be incremental and additive — no reprocessing of existing data when expanding. The architecture supports scaling from 500 → 10K → 100K+ repos over time by adding new batches, languages, and sources without invalidating prior work.

---

## 0. Why 500+ Repos

### The Primary Constraint: Distributional Coverage

The R-GIN model (1.95M params) needs 500K–1M training nodes for a healthy data-to-parameter ratio (~350:1). But **total node count is not the real bottleneck — distributional coverage is.**

The model must learn structural patterns across:
- Different architectural styles (layered, flat, monolith, workspace, pipeline)
- Different quality levels (clean, mixed, messy — all three are training signal)
- Different domains (web, CLI, data/ML, systems, libraries)
- Different sizes (50–50K nodes)
- Both supported languages (Python, Rust)
- Different edge type distributions (call-heavy, import-heavy, inherits-heavy)

500 copies of Flask-like web frameworks would satisfy the node count but fail the coverage requirement. The pipeline optimizes for **diversity first, volume second.**

### Loss Rate Budget

The existing 50 repos were hand-curated. Automated discovery from GitHub will have significantly higher failure rates:

| Loss source | Estimated rate |
|-------------|---------------|
| Parse failures (especially Python dynamic patterns) | 10–15% |
| Node count out of range (< 50 or > 50K) | 10–15% |
| Degenerate graphs (fragmented, no spectral signal) | 5–10% |
| Entrypoint detection failure | 5–10% |
| Missing edge types / sparse graphs | 5% |
| **Total expected loss** | **30–50%** |

**Target: discover 1000–1500 candidate repos to yield 500–800 usable training graphs.** At the pessimistic end (50% loss), 1000 candidates yields 500. At the optimistic end (30% loss), 1000 yields 700. The plan targets the conservative end.

### Quality Diversity Is a Feature

The model must learn structural patterns across the full quality spectrum. Messy codebases with architectural debt, god modules, and accidental coupling are **training signal** — the model needs to represent these structures to detect them downstream (issues pipeline).

The curation criteria must be loose enough to admit messy repos while tight enough to exclude trivially broken ones. The quality tag (`clean` / `mixed` / `messy`) is tracked metadata, not a filter.

---

## 1. Current State

### Registry Inventory (50 repos)

| Category | Python | Rust | Total |
|----------|--------|------|-------|
| Small | 7 | 6 | 13 |
| Medium | 6 | 10 | 16 |
| Large | 9 | 12 | 21 |
| **Total** | **22** | **28** | **50** |

### Gaps

| Gap | Details |
|-----|---------|
| **Volume** | 50 repos → need 10–15x more |
| **Domain: Data/ML** | Zero repos. No pandas, numpy, sklearn, torch, polars, etc. |
| **Domain: DevOps/Infra** | Minimal. No ansible, terraform, pulumi, etc. |
| **Domain: Testing/Tooling** | Only pytest. No tox, nox, coverage tools |
| **Quality diversity** | 2 tagged "messy" (celery, sqlalchemy). Need 15–20% messy |
| **Pinning** | 38/50 repos use `commit = "HEAD"` (not reproducible) |
| **Tags** | 30+ entries lack size/quality/pattern tags |
| **Language balance** | 44% Python / 56% Rust — adequate but skewed |

### Existing Infrastructure

| Component | Status | Location |
|-----------|--------|----------|
| Registry manifest | Working | `examples/registry.toml` |
| Clone + parse pipeline | Working | `benchmark/scripts/harvest_corpus.py` |
| GitHub metadata fetcher | Working (basic) | `examples/scripts/collect_metadata.py` |
| Preprocessing (NPZ export) | Partially working (pending Step 0 `topo export-features` CLI) | `packages/topo-dataset/scripts/preprocess.py` |
| Validation | Working | `packages/topo-dataset/scripts/validate.py` |
| Train/val/test splits | Working | `packages/topo-dataset/scripts/split.py` |
| **Curation script** | **NOT IMPLEMENTED** | `packages/topo-dataset/scripts/curate_repos.py` |

---

## 2. Pipeline Architecture

```
                    ┌─────────────────────────────────────────┐
                    │     Stage 1: DISCOVERY                  │
                    │  GitHub Search API (paginated queries)   │
                    │  Split by language × star-range buckets  │
                    └────────────────┬────────────────────────┘
                                     │ ~3000–5000 raw candidates
                                     ▼
                    ┌─────────────────────────────────────────┐
                    │     Stage 2: PRE-FILTER                 │
                    │  Per-repo API: stars, license, fork,     │
                    │  archived, contributors, created_at      │
                    └────────────────┬────────────────────────┘
                                     │ ~1500–2500 pass pre-filter
                                     ▼
                    ┌─────────────────────────────────────────┐
                    │     Stage 3: ENTRYPOINT DETECTION       │
                    │  Read pyproject.toml / Cargo.toml via    │
                    │  GitHub Contents API. Auto-detect pkg.   │
                    └────────────────┬────────────────────────┘
                                     │ ~1200–2000 with resolved entrypoints
                                     ▼
                    ┌─────────────────────────────────────────┐
                    │     Stage 4: REGISTRY INGESTION         │
                    │  Pin commit SHA, assign domain tag,      │
                    │  generate [[example]] block, append to   │
                    │  registry.toml                           │
                    └────────────────┬────────────────────────┘
                                     │ ~800–1500 in registry
                                     ▼
                    ┌─────────────────────────────────────────┐
                    │     Stage 5: PARSE + POST-FILTER        │
                    │  Clone, parse with topo, check graph     │
                    │  quality (nodes, edges, components)      │
                    └────────────────┬────────────────────────┘
                                     │ ~500–800 usable graphs
                                     ▼
                    ┌─────────────────────────────────────────┐
                    │     Stage 6: TAG + BALANCE              │
                    │  Auto-tag (size, domain). Flag quality.  │
                    │  Check distributional coverage.          │
                    │  Iterate if gaps remain.                 │
                    └─────────────────────────────────────────┘
```

### Incremental Expansion Strategy

Do NOT go from 50 to 1500 in one shot. Expand in batches to validate assumptions:

| Batch | Registry target | Expected usable (after 30-50% loss) | Purpose |
|-------|----------------|--------------------------------------|---------|
| **Batch 0** | Pin + tag existing 50 | ~35–40 (lower loss: hand-curated) | Fix unpinned SHAs, backfill tags, measure baseline loss |
| **Batch 1** | +200 → 250 total | ~125–175 | Validate pipeline, measure actual loss rate |
| **Batch 2** | +250 → 500 total | ~250–350 | Fill domain gaps (Data/ML, Systems) |
| **Batch 3** | +300 → 800 total | ~400–560 | Saturate coverage, add quality diversity |
| **Batch 4** | +200–400 → 1000–1200 | **500–840** | Reach minimum 500 usable target |

**After each batch:** parse all new repos, run quality filters, check distributional coverage, adjust strategy. Batch 4 is sized based on the actual loss rate measured in Batches 1–3. If the measured loss rate is ≤30%, Batch 4 may not be needed. If >40%, increase Batch 4 size.

**Exit criterion:** ≥500 usable graphs with distributional coverage meeting all targets (±5%). If this is not achievable after Batch 4, either relax the star threshold to ≥50 for underrepresented domains, or reduce the model's hidden dimension from 256 to 128 (cutting parameter count to ~600K, which needs fewer training nodes).

---

## 3. Stage 1: Discovery

### GitHub Search API Strategy

The Search API returns max 1000 results per query. To exceed 1000 repos per language, split queries by **star-count ranges**:

```
# Python queries (estimated pool sizes)
language:python stars:100..200       → ~3000+ repos (take first 1000)
language:python stars:200..500       → ~2000+ repos (take first 1000)
language:python stars:500..2000      → ~1500+ repos (take first 1000)
language:python stars:2000..10000    → ~800  repos
language:python stars:>10000         → ~200  repos

# Rust queries (smaller ecosystem)
language:rust stars:100..200         → ~800  repos
language:rust stars:200..500         → ~500  repos
language:rust stars:500..2000        → ~400  repos
language:rust stars:>2000            → ~150  repos

# TypeScript queries (large ecosystem, future parser)
language:typescript stars:100..200   → ~5000+ repos (take first 1000)
language:typescript stars:200..500   → ~3000+ repos (take first 1000)
language:typescript stars:500..2000  → ~2000+ repos (take first 1000)
language:typescript stars:2000..10000 → ~1000 repos
language:typescript stars:>10000     → ~300  repos
```

**TypeScript repos:** Included in the registry now even though topo does not yet have a TypeScript parser. Rationale: (1) the registry is a curated list of repos, not a list of parseable repos — curation is the expensive part, (2) when a TS parser ships, the preprocessed dataset can grow instantly without re-curating, (3) TypeScript has the largest pool of high-quality open-source repos of the three languages, providing far more architectural diversity. TS repos are tagged `language = "typescript"` and excluded from parse/preprocess stages until the parser exists.

Additional search qualifiers per query:
```
fork:false                  # Source repos only
archived:false              # Active repos only
sort:stars                  # Most popular first (quality proxy)
```

### Search API Implementation

```python
# curate_repos.py — Stage 1 pseudocode

STAR_BUCKETS = [
    (100, 200),
    (200, 500),
    (500, 2000),
    (2000, 10000),
    (10000, None),  # 10000+
]

def discover_candidates(language: str, max_per_bucket: int = 300) -> list[dict]:
    """Query GitHub Search API across star-range buckets."""
    candidates = []
    for low, high in STAR_BUCKETS:
        star_query = f"stars:{low}..{high}" if high else f"stars:>={low}"
        query = f"language:{language} {star_query} fork:false archived:false"

        results = gh_search_repos(query, per_page=100, max_results=max_per_bucket)
        candidates.extend(results)

        # Respect rate limit: 30 search requests/minute
        time.sleep(2)  # conservative spacing

    return candidates
```

### Rate Limit Strategy

**Recommended: GraphQL for enrichment, REST Search for discovery.** GraphQL collapses 3–5 REST calls per repo into a single batched query for ~30 repos at once — a 90x reduction.

| Phase | API | Rate limit | Time for 1500 repos |
|-------|-----|-----------|---------------------|
| Discovery | REST Search API | 30 req/min | ~50 queries, ~2 min |
| Enrichment | **GraphQL API** | 5000 points/hr | ~50 queries (30/batch), **~5 min** |
| Contributor count | REST API | 5000 req/hr | ~1500 calls, ~20 min |
| **Total** | | | **~30 min** |

A single GraphQL query fetches stars, license, default branch + HEAD SHA, CI/CD directory check, and `pyproject.toml`/`Cargo.toml` content for ~30 repos at once:

```graphql
query {
  repo1: repository(owner: "pallets", name: "flask") {
    stargazerCount
    primaryLanguage { name }
    licenseInfo { spdxId }
    isArchived
    isFork
    pushedAt
    createdAt
    defaultBranchRef { name, target { oid } }
    workflows: object(expression: "HEAD:.github/workflows") {
      ... on Tree { entries { name } }
    }
    pyproject: object(expression: "HEAD:pyproject.toml") {
      ... on Blob { text }
    }
    cargo: object(expression: "HEAD:Cargo.toml") {
      ... on Blob { text }
    }
  }
  repo2: repository(owner: "psf", name: "requests") { ...same fields... }
  # ... up to ~30 repos per query
}
```

**Contributor count gap:** GraphQL doesn't directly expose contributor count. Use one REST call per repo with `per_page=20&page=1` — if 20 results returned, the repo passes the ≥20 threshold (no need to paginate). This adds ~20 min but can run in parallel with other stages.

**Authentication:** Use `gh api graphql` CLI (same auth as REST). All API calls via `gh api` subprocess.

**Caching:** Write intermediate results to `examples/.cache/discovery_<language>_<timestamp>.json` so the pipeline is resumable if interrupted.

### What About HuggingFace?

**Verdict: not useful for this pipeline.** The Stack and similar datasets are collections of individual source files, not curated repo lists with metadata. Extracting repo-level information (stars, contributors, architectural style) requires reverse-engineering file-to-repo mappings and then querying GitHub anyway. The GitHub Search API is strictly faster and provides richer metadata.

HuggingFace could be useful for a different purpose: pre-computed code embeddings. But our pipeline uses a specific CodeLM model (jina-embeddings-v2-base-code) and needs embeddings aligned with topo's source text extraction format. Pre-computed embeddings from another model would not be compatible.

---

## 4. Stage 2: Pre-Filtering

### Filter Criteria

Applied via GitHub API metadata (no cloning required):

| Criterion | Threshold | API source | Rationale |
|-----------|-----------|------------|-----------|
| Stars | ≥ 100 | Search query | Filters trivial/homework projects |
| Fork | Source only | Search query | Deduplication |
| Archived | Excluded | Search query | Dead codebases have stale structure |
| License | OSI-approved | `repos/{owner}/{repo}` → `license.spdx_id` | Legal clarity |
| Activity | Pushed within last 3 years | `repos/{owner}/{repo}` → `pushed_at` | Exclude abandoned repos |
| Size | > 50 KB repo size | `repos/{owner}/{repo}` → `size` | Filter empty/stub repos (note: this is git repo size, a rough proxy only) |
| CI/CD | Has `.github/workflows/` OR `.travis.yml` OR `circle.yml` | Contents API tree check | Proxy for code quality and maintenance (per STEP_1 spec) |

### Relaxed Contributor Filter

The STEP_1 spec says "≥ 50 contributors OR ≥ 3 years active." The critic correctly identified that this excludes Data/ML and Systems repos that tend to be smaller communities. **Revised filter:**

```
≥ 20 contributors OR ≥ 2 years active (pushed_at - created_at ≥ 2 years)
```

**Rationale:** The star threshold (≥ 100) already filters toy projects. Lowering the contributor bar from 50 to 20 admits niche-but-quality libraries (polars, ratatui, seaborn) while still excluding single-developer experiments. The 2-year activity window catches repos that are maintained but small-community.

**For Data/ML repos specifically:** Further relax to `≥ 10 contributors OR ≥ 50 stars` (stars serve as quality proxy for this domain where solo-maintained ML libraries are common and high-quality).

### Pre-Filter Implementation

```python
def pre_filter(repo: dict) -> tuple[bool, str]:
    """Check if a repo passes pre-filter. Returns (pass, reason)."""
    now = datetime.now(timezone.utc)

    # License check
    license_id = repo.get("license", {}).get("spdx_id", "")
    if license_id not in OSI_LICENSES and license_id != "NOASSERTION":
        return False, f"license:{license_id}"

    # Activity check
    pushed = parse_date(repo["pushed_at"])
    created = parse_date(repo["created_at"])
    age_years = (now - created).days / 365
    recent = (now - pushed).days < 365 * 3

    if not recent:
        return False, "inactive"

    # Contributor check (relaxed from STEP_1 spec: ≥20 OR ≥2y, not ≥50 OR ≥3y)
    # Note: age_years uses repo creation date as proxy for commit span.
    # True commit span would require git log analysis (too expensive pre-clone).
    contributors = get_contributor_count(repo["full_name"])
    if contributors < 20 and age_years < 2:
        return False, f"contributors:{contributors},age:{age_years:.1f}y"

    # Size check (rough proxy — GitHub 'size' is git repo size in KB, includes history)
    if repo["size"] < 50:  # KB
        return False, f"size:{repo['size']}KB"

    # CI/CD check (per STEP_1 spec: proxy for code quality)
    if not has_ci_cd(repo["full_name"]):
        return False, "no_ci_cd"

    return True, "ok"


def has_ci_cd(owner_repo: str) -> bool:
    """Check for CI/CD configuration via GitHub Contents API."""
    # Check .github/workflows/ (most common)
    if gh_path_exists(owner_repo, ".github/workflows"):
        return True
    # Check .travis.yml
    if gh_path_exists(owner_repo, ".travis.yml"):
        return True
    # Check .circleci/config.yml
    if gh_path_exists(owner_repo, ".circleci/config.yml"):
        return True
    return False
```

**Note on CI/CD cost:** This adds 1–3 API calls per candidate. To reduce API usage, check `.github/workflows` first (covers ~90% of active repos) and short-circuit on match.

---

## 5. Stage 3: Entrypoint Detection

This is the highest-automation-risk stage. Every existing registry entry has a manually specified `entrypoint`. At 800+ repos, manual specification is infeasible.

### Rust Repos: Trivial

Rust repos always use `entrypoint = "."` because the Rust parser reads `Cargo.toml` at the workspace root and discovers all crates automatically. **No detection needed.**

### TypeScript Repos: Convention-Based

TypeScript repos typically have `src/` as the source root. Detection heuristic:
1. Read `tsconfig.json` → `compilerOptions.rootDir` or `include` paths
2. Read `package.json` → `main` or `types` field to infer package root
3. Fallback: check if `src/` directory exists → use `"src"`
4. Final fallback: `"."` (whole repo)

Since TS repos are registry-only (not parseable yet), entrypoint detection failures are non-blocking — tag as `entrypoint = "."` and refine later when the parser ships.

### Python Repos: Heuristic Detection

Read the repo's `pyproject.toml` or `setup.py` via GitHub Contents API and extract the package location.

```python
def detect_python_entrypoint(owner_repo: str) -> str | None:
    """Auto-detect Python package entrypoint from project config."""

    # Try pyproject.toml first
    content = gh_get_file(owner_repo, "pyproject.toml")
    if content:
        parsed = tomllib.loads(content)

        # Extract project name first (needed by multiple methods)
        project_name = parsed.get("project", {}).get("name", "").replace("-", "_")

        # Method 1: [tool.setuptools.packages.find] where = ["src"]
        # NOTE: `where` gives the parent directory (e.g., "src"), NOT the package
        # directory (e.g., "src/flask"). Must combine with project name.
        find = parsed.get("tool", {}).get("setuptools", {}).get("packages", {}).get("find", {})
        where = find.get("where", [])
        if where and project_name:
            candidate = f"{where[0]}/{project_name}"
            if gh_path_exists(owner_repo, candidate):
                return candidate
            # Fallback: maybe the where dir IS the package (less common)
            if gh_path_exists(owner_repo, where[0]):
                return where[0]

        # Method 2: [tool.setuptools.package-dir] "" = "src"
        pkg_dir = parsed.get("tool", {}).get("setuptools", {}).get("package-dir", {})
        if "" in pkg_dir and project_name:
            candidate = f"{pkg_dir['']}/{project_name}"
            if gh_path_exists(owner_repo, candidate):
                return candidate

        # Method 3: [project] name → check src/<name> or <name>
        if project_name:
            if gh_path_exists(owner_repo, f"src/{project_name}"):
                return f"src/{project_name}"
            if gh_path_exists(owner_repo, project_name):
                return project_name

    # Try setup.py (legacy)
    content = gh_get_file(owner_repo, "setup.py")
    if content:
        # Regex for package_dir={"": "src"} or packages=find_packages("src")
        # NOTE: These return the parent dir (e.g., "src"), not the package dir.
        # Must combine with project_name, same as pyproject.toml Method 1.
        match = re.search(r'package_dir\s*=\s*\{\s*["\'][\s]*["\']\s*:\s*["\'](\w+)["\']', content)
        parent_dir = match.group(1) if match else None

        if not parent_dir:
            match = re.search(r'find_packages\(\s*["\'](\w+)["\']', content)
            parent_dir = match.group(1) if match else None

        if parent_dir and project_name:
            candidate = f"{parent_dir}/{project_name}"
            if gh_path_exists(owner_repo, candidate):
                return candidate

    # Fallback: look for top-level directory matching repo name
    project_name = owner_repo.split("/")[-1].replace("-", "_")
    if gh_path_exists(owner_repo, f"src/{project_name}"):
        return f"src/{project_name}"
    if gh_path_exists(owner_repo, project_name):
        return project_name

    return None  # Entrypoint could not be detected — skip this repo
```

### Failure Handling

Repos where entrypoint detection fails are logged and skipped. Expected failure rate: 5–10%. These repos can be manually reviewed later if needed to fill specific domain gaps.

---

## 6. Stage 4: Registry Ingestion

### Commit Pinning

Every repo must have a pinned commit SHA for reproducibility. Resolve at discovery time:

```python
def resolve_commit_sha(owner_repo: str, branch: str = None) -> str | None:
    """Resolve default branch HEAD to a concrete SHA."""
    if branch is None:
        repo_info = gh_api(f"repos/{owner_repo}")
        branch = repo_info["default_branch"]

    commit_info = gh_api(f"repos/{owner_repo}/commits/{branch}")
    return commit_info["sha"][:12]  # 12-char SHA for collision safety at scale
    # Note: existing entries use 7-char SHAs (hand-verified). 12 chars is safer
    # for automated scale (7-char collisions are possible across 800+ repos)
```

### Registry Entry Generation

```python
def generate_registry_entry(repo: dict, entrypoint: str, sha: str, domain: str) -> str:
    """Generate a TOML [[example]] block."""
    name = repo["name"]
    language = repo["language"].lower()
    size = classify_size(repo)
    # Escape TOML special characters in description (quotes, backslashes, newlines)
    description = (repo.get("description") or "")[:100]
    description = description.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")

    entry = f'''
[[example]]
name = "{name}"
repo = "{repo['html_url']}"
commit = "{sha}"
language = "{language}"
entrypoint = "{entrypoint}"
description = "{description}"

[example.tags]
size = "{size}"
domain = "{domain}"
'''
    return entry.strip()
```

### Size Classification

```python
def classify_size(repo: dict) -> str:
    """Classify repo size from GitHub API 'size' field (KB)."""
    size_kb = repo.get("size", 0)
    if size_kb < 2000:       # < 2 MB
        return "small"
    elif size_kb < 20000:    # < 20 MB
        return "medium"
    else:
        return "large"
```

**Note:** This is a pre-parse heuristic. The definitive size is the node count after parsing. The tag is updated in Stage 6 after parsing.

### Deduplication

Before appending, check:
1. **Name collision:** repo name already in registry → skip
2. **URL collision:** same GitHub URL → skip
3. **Org clustering:** max 8 repos per GitHub organization to prevent org-dominated clusters (e.g., don't add 20 pallets/* repos). Cap of 8 (not 5) because some orgs like tokio-rs maintain many structurally distinct projects. Applied after structural deduplication — if 8 repos from the same org are all structurally distinct, they all stay.

---

## 7. Stage 5: Parse + Post-Filter

This stage uses the existing `harvest_corpus.py` pipeline. After cloning and parsing, apply quality filters:

### Post-Parse Quality Checks

| Check | Threshold | Action on fail |
|-------|-----------|----------------|
| Node count | 50 ≤ n ≤ 50,000 | Remove from registry |
| Edge count | edges ≥ n/2 | Remove |
| Largest connected component | ≥ 50% of nodes | Remove |
| Non-trivial eigenvectors | ≥ 2 | Remove |
| Parse errors | Error during parse | Log, remove |
| Edge type coverage | At least 2 of 3 edge types present | Flag (don't remove) |

### Quality Tagging (Post-Parse)

After parsing, compute quality indicators and tag:

```python
def tag_quality(graph: dict) -> str:
    """Heuristic quality classification from graph structure."""
    n_nodes = len(graph["nodes"])
    n_edges = len(graph["edges"])

    # Edge density
    density = n_edges / max(n_nodes, 1)

    # Compute from graph metadata if available
    modularity = graph.get("metrics", {}).get("modularity_q", 0)

    # Heuristics:
    # - Very high density (> 5 edges/node) suggests tangled coupling
    # - Very low modularity (< 0.2) suggests monolithic structure
    # - Presence of cycles in the module dependency graph
    if density > 5 or modularity < 0.2:
        return "messy"
    elif density > 3 or modularity < 0.4:
        return "mixed"
    else:
        return "clean"
```

**This is an approximation.** Quality is ultimately a human judgment. The heuristic gets us 80% of the way; manual review refines it.

### Node Count–Based Size Re-Tagging

Update the `size` tag after parsing with actual node counts:

| Size | Node count range |
|------|-----------------|
| small | 50–300 |
| medium | 300–2000 |
| large | 2000–50000 |

---

## 8. Stage 6: Distributional Coverage Check

After each batch, measure coverage across required dimensions and identify gaps.

### Target Distribution

**Note:** This table intentionally deviates from STEP_1 §Stratification (which defines 6 categories). Changes: (1) Split "Monorepo" into a cross-cutting tag rather than a domain category (a monorepo can be web, CLI, or library — it's a structural pattern, not a domain). (2) Added "DevOps" and "Other" to capture repos that don't fit the original 6. (3) Adjusted percentages to reflect achievable targets from GitHub API discovery.

| Dimension | Category | Target % | Min count (for 600) |
|-----------|----------|----------|-------------------|
| **Domain** | Web (Django/Flask/FastAPI/Actix/Rocket) | 20–25% | 120 |
| | Library / framework | 20–25% | 120 |
| | CLI tool | 10–15% | 60 |
| | Data pipeline / ML | 10–15% | 60 |
| | Systems / infrastructure | 8–12% | 48 |
| | DevOps / tooling | 5–8% | 30 |
| | Other (editors, compilers, games, etc.) | 10–15% | 60 |
| **Language** | Python | 35–45% | 210 |
| | Rust | 35–45% | 210 |
| | TypeScript | 15–25% (registry only — not parseable yet) | 90 |
| **Size** | Small (50–300 nodes) | 25–35% | 150 |
| | Medium (300–2000 nodes) | 40–50% | 240 |
| | Large (2000–50K nodes) | 20–30% | 120 |
| **Quality** | Clean | 40–50% | 240 |
| | Mixed | 30–40% | 180 |
| | Messy | 15–25% | 90 |

### Domain Detection (Automatable)

Detect domain from dependency manifests:

```python
DOMAIN_SIGNALS = {
    "web": {
        "python": ["django", "flask", "fastapi", "starlette", "aiohttp", "tornado", "sanic", "pyramid", "bottle"],
        "rust": ["actix-web", "axum", "rocket", "warp", "hyper", "tower-http", "poem"],
        "typescript": ["express", "nestjs", "next", "nuxt", "fastify", "hono", "koa", "hapi", "remix"],
    },
    "cli": {
        # Note: argparse is stdlib — won't appear in dependency manifests. Omitted.
        "python": ["click", "typer", "fire", "rich", "prompt-toolkit", "textual"],
        "rust": ["clap", "structopt", "argh", "dialoguer", "indicatif", "ratatui"],
        "typescript": ["commander", "yargs", "inquirer", "chalk", "ora", "oclif"],
    },
    "data_ml": {
        "python": ["pandas", "numpy", "scipy", "sklearn", "torch", "tensorflow", "polars",
                    "matplotlib", "seaborn", "xgboost", "transformers", "datasets"],
        "rust": ["ndarray", "polars", "candle", "burn", "linfa", "smartcore"],
        "typescript": ["tensorflow", "onnxruntime-node", "ml5", "danfojs"],
    },
    "systems": {
        # Note: asyncio/multiprocessing are stdlib — won't appear in dependency manifests.
        # Use installable packages only.
        "python": ["celery", "redis", "kafka-python", "grpcio", "aioredis", "zmq",
                    "kombu", "dramatiq", "huey", "rq"],
        "rust": ["tokio", "async-std", "mio", "crossbeam", "parking_lot", "rayon"],
        "typescript": ["ioredis", "kafkajs", "bullmq", "amqplib", "grpc-js"],
    },
    "devops": {
        "python": ["ansible", "fabric", "invoke", "boto3", "docker", "kubernetes"],
        "rust": ["bollard", "kube", "k8s-openapi"],
        "typescript": ["aws-cdk-lib", "pulumi", "projen", "cdktf"],
    },
    "library": {
        # Fallback: published to PyPI/crates.io, no domain-specific deps
    },
}

def detect_domain(deps: list[str], language: str, has_pyproject: bool = False,
                   has_cargo_lib: bool = False) -> str:
    """Classify repo domain from its dependencies."""
    for domain, lang_signals in DOMAIN_SIGNALS.items():
        if domain == "library":
            continue  # library is the default, not signal-matched
        signals = lang_signals.get(language, [])
        if any(dep in signals for dep in deps):
            return domain

    # If no domain-specific signals matched, classify as "library" if the repo
    # appears to be a published package (has pyproject.toml [project] or Cargo.toml [lib]),
    # otherwise "other".
    if has_pyproject or has_cargo_lib:
        return "library"
    return "other"
```

### Dependency Extraction

```python
def extract_dependencies(owner_repo: str, language: str) -> list[str]:
    """Extract dependency names from project manifest via GitHub Contents API."""
    if language == "python":
        # Read pyproject.toml → [project.dependencies]
        content = gh_get_file(owner_repo, "pyproject.toml")
        if content:
            parsed = tomllib.loads(content)
            deps = parsed.get("project", {}).get("dependencies", [])
            # Extract package names (strip version specifiers)
            return [re.split(r"[><=!~\[]", d)[0].strip().lower() for d in deps]

        # Fallback: requirements.txt
        content = gh_get_file(owner_repo, "requirements.txt")
        if content:
            return [re.split(r"[><=!~\[]", line)[0].strip().lower()
                    for line in content.splitlines()
                    if line.strip() and not line.startswith("#")]

    elif language == "rust":
        # Read Cargo.toml → [dependencies] or [workspace.dependencies]
        content = gh_get_file(owner_repo, "Cargo.toml")
        if content:
            parsed = tomllib.loads(content)
            deps = list(parsed.get("dependencies", {}).keys())
            # Workspace repos often have deps under [workspace.dependencies]
            deps += list(parsed.get("workspace", {}).get("dependencies", {}).keys())
            return deps

    elif language == "typescript":
        # Read package.json → dependencies + devDependencies
        content = gh_get_file(owner_repo, "package.json")
        if content:
            pkg = json.loads(content)
            deps = list(pkg.get("dependencies", {}).keys())
            deps += list(pkg.get("devDependencies", {}).keys())
            return [d.split("/")[-1] for d in deps]  # strip @scope/ prefix

    return []
```

### Gap-Filling Queries

If a batch reveals an underrepresented category, run targeted search queries:

```python
# Example: Data/ML gap detected after Batch 1
TARGETED_QUERIES = {
    "data_ml_python": [
        "language:python topic:machine-learning stars:>100 fork:false",
        "language:python topic:data-science stars:>100 fork:false",
        "language:python topic:deep-learning stars:>100 fork:false",
        "language:python topic:pytorch stars:>100 fork:false",
        "language:python pandas in:description stars:>100 fork:false",
    ],
    "systems_rust": [
        "language:rust topic:async stars:>50 fork:false",
        "language:rust topic:networking stars:>50 fork:false",
        "language:rust tokio in:description stars:>50 fork:false",
    ],
}
```

---

## 9. Structural Deduplication

Many popular repos have nearly identical graph structures (e.g., small Rust CLI tools: bat, fd, hyperfine, zoxide all follow the same single-crate + clap pattern). Adding 50 more of these doesn't improve distributional coverage.

### Post-Parse Deduplication

After parsing, compute a structural fingerprint for each graph:

```python
def structural_fingerprint(graph: dict) -> tuple:
    """Coarse structural fingerprint for deduplication."""
    n = len(graph["nodes"])
    e = len(graph["edges"])

    # Node type distribution (normalized)
    type_dist = Counter(node["kind"] for node in graph["nodes"])
    total = sum(type_dist.values())
    type_vec = tuple(type_dist.get(k, 0) / total for k in sorted(NODE_TYPES))

    # Edge type distribution (normalized)
    edge_dist = Counter(edge["kind"] for edge in graph["edges"])
    total_e = sum(edge_dist.values())
    edge_vec = tuple(edge_dist.get(k, 0) / max(total_e, 1) for k in ["calls", "imports", "inherits"])

    # Size bucket
    size_bucket = "S" if n < 300 else "M" if n < 2000 else "L"

    # Density bucket
    density = e / max(n, 1)
    density_bucket = "sparse" if density < 1.5 else "medium" if density < 4 else "dense"

    return (size_bucket, density_bucket, type_vec, edge_vec)
```

**Soft cap:** Max 10 repos per structural fingerprint cluster. If a cluster exceeds 10, prefer repos from underrepresented domains/orgs.

---

## 10. Script Design: `curate_repos.py`

### CLI Interface

```
usage: curate_repos.py [-h] [--target N] [--batch-size N]
                       [--languages {python,rust,typescript,all}]
                       [--fill-domain DOMAIN]
                       [--dry-run] [--resume]
                       [--cache-dir DIR]

Expand registry.toml via GitHub API discovery.

options:
  --target N          Target number of repos in registry (default: 800)
  --batch-size N      Repos to add per batch (default: 200)
  --languages         Languages to search (default: both)
  --fill-domain       Target a specific underrepresented domain
  --dry-run           Print candidates without modifying registry
  --resume            Resume from cached discovery results
  --cache-dir         Cache directory (default: examples/.cache/)
```

### Main Flow

```python
def main():
    args = parse_args()
    registry = load_registry()
    existing_names = {e["name"] for e in registry}
    existing_urls = {e["repo"] for e in registry}

    # Stage 1: Discover candidates
    candidates = []
    for language in args.languages:
        raw = discover_candidates(language, cache_dir=args.cache_dir)
        candidates.extend(raw)

    print(f"Discovered {len(candidates)} raw candidates")

    # Stage 2: Pre-filter
    filtered = []
    for repo in tqdm(candidates, desc="Pre-filtering"):
        if repo["html_url"] in existing_urls:
            continue
        if normalize_name(repo["name"]) in existing_names:
            continue
        passed, reason = pre_filter(repo)
        if passed:
            filtered.append(repo)
        else:
            log_skip(repo, reason)

    print(f"{len(filtered)} pass pre-filter")

    # Stage 3: Entrypoint detection
    with_entrypoints = []
    for repo in tqdm(filtered, desc="Detecting entrypoints"):
        language = repo["language"].lower()
        if language == "rust":
            entrypoint = "."
        else:
            entrypoint = detect_python_entrypoint(repo["full_name"])
        if entrypoint:
            repo["_entrypoint"] = entrypoint
            with_entrypoints.append(repo)
        else:
            log_skip(repo, "no_entrypoint")

    print(f"{len(with_entrypoints)} with resolved entrypoints")

    # Stage 4: Pin commits + detect domain
    entries = []
    for repo in tqdm(with_entrypoints, desc="Pinning commits"):
        sha = resolve_commit_sha(repo["full_name"])
        deps = extract_dependencies(repo["full_name"], repo["language"].lower())
        domain = detect_domain(deps, repo["language"].lower())
        entries.append({
            "repo": repo,
            "sha": sha,
            "entrypoint": repo["_entrypoint"],
            "domain": domain,
            "deps": deps,
        })

    # Dedup by org (max 8 per org — see Section 6 deduplication rationale)
    entries = apply_org_cap(entries, max_per_org=8)

    # Take batch_size
    entries = entries[:args.batch_size]

    if args.dry_run:
        for e in entries:
            print(f"  {e['repo']['full_name']} [{e['domain']}] @ {e['sha']}")
        return

    # Stage 4: Append to registry.toml
    append_to_registry(entries, REGISTRY)
    print(f"Added {len(entries)} repos to registry.toml")
```

### Resume / Caching

All API results are cached to `examples/.cache/`:
- `discovery_python_<date>.json` — raw search results
- `discovery_rust_<date>.json`
- `prefilter_results_<date>.json` — pre-filter pass/fail per repo
- `entrypoints_<date>.json` — detected entrypoints

The `--resume` flag loads from cache instead of re-querying the API.

---

## 11. Script Design: `pin_existing.py`

Fix the 38 unpinned repos in the current registry before expanding.

```
usage: pin_existing.py [-h] [--dry-run]

Pin all HEAD-committed repos in registry.toml to concrete SHAs.
```

```python
def main():
    registry_path = PROJECT_ROOT / "examples" / "registry.toml"

    # Use tomlkit for round-trip-safe TOML editing (preserves comments and formatting)
    # pip install tomlkit
    import tomlkit
    doc = tomlkit.parse(registry_path.read_text())

    for entry in doc.get("example", []):
        if entry.get("commit") == "HEAD":
            name = entry["name"]
            sha = resolve_commit_sha(extract_owner_repo(entry["repo"]))
            if sha:
                entry["commit"] = sha
                print(f"  Pinned {name} → {sha}")

    # Verify round-trip: parse the output and check it matches
    output = tomlkit.dumps(doc)
    verify = tomlkit.parse(output)
    assert len(verify["example"]) == len(doc["example"]), "Round-trip lost entries!"

    registry_path.write_text(output)
```

**Dependency:** `tomlkit` (supports round-trip TOML editing with comment/formatting preservation). Add to `topo-dataset` dev dependencies. This is strictly safer than regex-based replacement, which risks matching "HEAD" in descriptions or comments.

---

## 12. Script Design: `coverage_report.py`

Measure distributional coverage after each batch.

```
usage: coverage_report.py [-h] [--parsed-only]

Report distributional coverage of the current registry.
```

Output:
```
Registry Coverage Report
========================
Total repos: 800  (parsed: 650, usable: 580)

Language Distribution:
  python:     320 (40.0%) ✓ target: 35-45%
  rust:       300 (37.5%) ✓ target: 35-45%
  typescript: 180 (22.5%) ✓ target: 15-25% (registry-only, not parsed yet)

Domain Distribution:
  web:         185 (23.1%) ✓ target: 20-25%
  library:     190 (23.8%) ✓ target: 20-25%
  cli:          95 (11.9%) ✓ target: 10-15%
  data_ml:      80 (10.0%) ✓ target: 10-15%
  systems:      70  (8.8%) ✓ target: 8-12%
  devops:       40  (5.0%) ✓ target: 5-8%
  other:       140 (17.5%) ~ target: 10-15% (slightly over)

Size Distribution (parsed repos only):
  small:  180 (27.7%) ✓ target: 25-35%
  medium: 290 (44.6%) ✓ target: 40-50%
  large:  180 (27.7%) ✓ target: 20-30%

Quality Distribution (parsed repos only):
  clean:  250 (38.5%) ~ target: 40-50%
  mixed:  230 (35.4%) ✓ target: 30-40%
  messy:  170 (26.2%) ✓ target: 15-25%

Edge Type Coverage:
  3/3 edge types: 520 (80.0%) ✓ target: ≥80%
  2/3 edge types: 110 (16.9%)
  1/3 edge types:  20  (3.1%)

GAPS:
  ⚠ "clean" quality is 2% below minimum — consider adding well-maintained libraries
```

---

## 13. Tag Taxonomy

### Automatable Tags (assigned by pipeline)

| Tag | Field | Values | Detection method |
|-----|-------|--------|-----------------|
| `size` | `[example.tags].size` | small, medium, large | Node count after parsing |
| `domain` | `[example.tags].domain` | web, library, cli, data_ml, systems, devops, other | Dependency manifest analysis |
| `language` | `[example].language` | python, rust, typescript | GitHub API language field |

### Heuristic Tags (best-effort automation)

| Tag | Field | Values | Detection method |
|-----|-------|--------|-----------------|
| `quality` | `[example.tags].quality` | clean, mixed, messy | Post-parse edge density + modularity heuristic |

### Cross-Cutting Tags (best-effort automation)

| Tag | Field | Values | Detection method |
|-----|-------|--------|-----------------|
| `monorepo` | `[example.tags].monorepo` | true/false | Check for multiple `Cargo.toml` or `setup.py`/`pyproject.toml` files via GitHub tree API |

```python
def detect_monorepo(owner_repo: str, language: str) -> bool:
    """Check if repo is a monorepo / multi-package workspace."""
    tree = gh_api(f"repos/{owner_repo}/git/trees/HEAD?recursive=1")
    if not tree:
        return False
    paths = [item["path"] for item in tree.get("tree", []) if item["type"] == "blob"]
    if language == "rust":
        cargo_count = sum(1 for p in paths if p.endswith("Cargo.toml"))
        return cargo_count > 2  # root + at least 2 sub-crates
    else:
        pyproject_count = sum(1 for p in paths if p.endswith("pyproject.toml") or p.endswith("setup.py"))
        return pyproject_count > 1
```

### Manual Tags (not automated — assigned during review)

| Tag | Field | Values | Detection method |
|-----|-------|--------|-----------------|
| `pattern` | `[example.tags].pattern` | layered, flat-library, monolith, workspace, pipeline, plugin-system, decorator-api, visitor-pattern, sansio, facade, etc. | Requires structural analysis (circular dependency with training) |

**Decision:** Stratification for train/val/test splits uses `domain` (automatable) as the primary key, not `pattern` (manual). Pattern tags are nice-to-have metadata, not pipeline requirements. The `monorepo` cross-cutting tag is tracked but not used for stratification — it's a structural property that cuts across domains.

---

## 14. Integration with Existing Pipeline

### Workflow After Registry Expansion

```bash
# 0. Pin existing repos
python packages/topo-dataset/scripts/pin_existing.py

# 1. Expand registry (batch 1)
python packages/topo-dataset/scripts/curate_repos.py --batch-size 200

# 2. Parse all new repos
make harvest-all

# 3. Validate quality
python packages/topo-dataset/scripts/validate.py

# 4. Check coverage
python packages/topo-dataset/scripts/coverage_report.py

# 5. If gaps remain, target-fill
python packages/topo-dataset/scripts/curate_repos.py --fill-domain data_ml --batch-size 50

# 6. Repeat steps 2-5 until target met

# 7. Generate train/val/test splits
python packages/topo-dataset/scripts/split.py --seed 42

# 8. Preprocess (embed + export NPZ) — when Step 0 is complete
python packages/topo-dataset/scripts/preprocess.py --workers 4
```

### Makefile Targets (new)

```makefile
curate:                ## Expand registry by one batch (200 repos)
	python packages/topo-dataset/scripts/curate_repos.py --batch-size 200

curate-fill:           ## Fill a specific domain gap
	python packages/topo-dataset/scripts/curate_repos.py --fill-domain $(DOMAIN) --batch-size 50

pin-registry:          ## Pin all HEAD commits to concrete SHAs
	python packages/topo-dataset/scripts/pin_existing.py

coverage-report:       ## Show distributional coverage of current registry
	python packages/topo-dataset/scripts/coverage_report.py

curate-dry-run:        ## Preview what curate would add without modifying registry
	python packages/topo-dataset/scripts/curate_repos.py --batch-size 200 --dry-run
```

---

## 15. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| GitHub API rate limiting halts pipeline | Medium | Low | Cache all results; resume from cache; 2s sleep between requests |
| Python entrypoint detection fails for >20% of repos | Medium | Medium | Manual review queue for high-value repos; lower batch ambition |
| Parse failures exceed 30% | Medium | High | Increase candidate pool; relax star threshold to ≥50 for underrepresented domains |
| Data/ML domain has insufficient Rust repos | High | Low | Accept Python-heavy ML category; compensate with Rust systems repos |
| Structural deduplication removes too many repos | Low | Medium | Relax fingerprint clustering; increase cluster cap from 10 to 20 |
| `registry.toml` becomes unwieldy (>500 entries) | High | Low | Move to TOML array-of-tables format; add inline comments per section |
| TOML round-trip editing corrupts existing entries | Medium | High | Use `tomlkit` for round-trip-safe editing of existing entries; append-only for new entries via f-string; always test on copy before writing |
| Unpinned commits drift between discovery and parse | Low | Medium | Pin SHA at discovery time; verify SHA at clone time |

---

## 16. Definition of Done

### Phase 1: Foundation (Batch 0)
- [ ] All 50 existing repos pinned to concrete SHAs
- [ ] All 50 existing repos tagged with domain (auto) and size (from parse)
- [ ] `pin_existing.py` script working
- [ ] `coverage_report.py` script working and showing current gaps

### Phase 2: Pipeline (Batch 1)
- [ ] `curate_repos.py` discovers, filters, and appends 200 repos in one run
- [ ] Entrypoint auto-detection works for ≥80% of Python candidates
- [ ] All new entries have pinned SHAs, domain tags, size tags
- [ ] `make harvest-all` successfully parses ≥70% of new repos
- [ ] Coverage report shows improvement in all gap categories
- [ ] API caching enables full resume after interruption
- [ ] **BLOCKING:** `split.py` updated to stratify on `domain` tag (not `pattern`) — without this, the expanded dataset cannot be split correctly

### Phase 3: Scale (Batches 2–4)
- [ ] Registry reaches 800+ entries
- [ ] ≥600 repos parsed successfully (usable graph.json)
- [ ] ≥500 pass post-parse quality filters
- [ ] Distributional coverage meets all targets (±5%)
- [ ] Language balance: Python 35–45%, Rust 35–45%, TypeScript 15–25%
- [ ] Quality diversity: ≥15% messy, ≥30% mixed
- [ ] ≥80% of usable graphs have all 3 edge types
- [ ] Structural deduplication applied (no cluster >10)

### Phase 4: Integration
- [ ] `split.py` produces stratified splits on expanded dataset
- [ ] Train/val/test splits have representative domain coverage
- [ ] Test set frozen (flag in split.py)
- [ ] All scripts have `--help` documentation
- [ ] End-to-end: `curate → harvest → validate → coverage → split` runs without manual intervention

---

## 17. Scalability Architecture (10K → 100K+)

The pipeline must support scaling from 500 to 100K+ repos without reprocessing existing data. This is a first-class design constraint, not an afterthought.

### Invariant: Additive-Only Growth

Every stage of the pipeline is **additive** — new repos are appended, never inserted or merged:

| Artifact | Growth model | Why it works |
|----------|-------------|-------------|
| `registry.toml` | Append new `[[example]]` blocks | TOML arrays are order-independent |
| `examples/<name>/graph.json` | One directory per repo | New repos get new directories; old ones untouched |
| `examples/<name>/features.npz` | One file per repo | `--resume` skips existing |
| `examples/splits/` | Re-generated from scratch | Cheap operation (seconds, not hours) |
| `examples/quality_report.json` | Merged (new + existing) | Union of per-repo entries |

**Key principle:** The expensive operations (clone, parse, embed) are per-repo and cached. Adding 1000 new repos never reprocesses the existing 500. Only the cheap operations (split, coverage report) are re-run globally.

### Registry Format at Scale

At 10K+ entries, a single `registry.toml` file becomes unwieldy (~200K lines). Split into per-language files:

```
examples/
  registry/
    python.toml      # All Python entries
    rust.toml         # All Rust entries
    typescript.toml   # All TypeScript entries
    index.toml        # Metadata: total counts, last-updated, schema version
```

The curation script auto-detects which format is in use (single file vs directory) and writes accordingly. Migration from single-file to directory format is a one-time script.

### Selective Parsing

At scale, you don't parse everything. The pipeline supports selective processing:

```bash
# Parse only repos not yet parsed
python preprocess.py --resume

# Parse only repos matching a filter
python preprocess.py --language python --domain web

# Parse only repos added after a date
python preprocess.py --added-after 2026-04-01

# Parse a specific batch
python preprocess.py --batch 5
```

Each `[[example]]` entry gets a `batch = N` field set at curation time, enabling batch-level operations.

### Scaling Stages

| Scale | Strategy | Time estimate |
|-------|----------|---------------|
| **500–2K** | Single `registry.toml`, local parsing, single machine | ~4–8 hours |
| **2K–10K** | Split registry, parallel parsing (`--workers 8`), single machine | ~1–3 days |
| **10K–100K** | Split registry, distributed parsing (VAST.AI or similar), sharded NPZ storage | ~1–2 weeks |
| **100K+** | Registry as database (SQLite or PostgreSQL), streaming processing, object storage for artifacts | Ongoing |

### Incremental Training

The R-GIN training pipeline (Step 2) also supports growth:
- `TopoDataset` reads from `splits/train.txt` — adding repos to the split file is enough
- The PyG `DataLoader` handles variable-count datasets
- Checkpoints from smaller datasets can warm-start training on larger datasets (transfer the weights, reset the optimizer)
- New repos can be added to train/val sets only (test set is frozen)

### Future Language Support

When new parsers ship (TypeScript, Go, Java, etc.):
1. Repos are already curated and in the registry (tagged `language = "typescript"`)
2. Run `make harvest-all --language typescript` to parse them
3. Run `make preprocess-all --language typescript` to generate features
4. Update splits to include the new language
5. Fine-tune or retrain the model on the expanded dataset

No re-curation needed. The expensive human/API work (finding good repos, detecting entrypoints, assigning domains) is done once and reused across parser generations.

---

## 18. File Inventory

### New Scripts

| Script | Location | Purpose |
|--------|----------|---------|
| `curate_repos.py` | `packages/topo-dataset/scripts/` | GitHub API discovery + registry expansion |
| `pin_existing.py` | `packages/topo-dataset/scripts/` | Pin HEAD commits to SHAs |
| `coverage_report.py` | `packages/topo-dataset/scripts/` | Distributional coverage analysis |

### Modified Files

| File | Change |
|------|--------|
| `examples/registry.toml` | Expand from 50 to 800+ entries |
| `Makefile` | Add curate/pin/coverage targets |
| `packages/topo-dataset/scripts/split.py` | Use `domain` tag for stratification (not `pattern`) |

### New Artifacts

| Artifact | Location | Purpose |
|----------|----------|---------|
| Discovery cache | `examples/.cache/` | Resumable API results (gitignored) |
| Coverage report | `examples/coverage_report.json` | Machine-readable coverage metrics |

---

## Appendix A: Candidate Repo Sources by Domain

### Data/ML (Python-heavy, target: 60–90 repos)

High-value candidates (manual seed list):
- scikit-learn, pandas, numpy, polars, xgboost, lightgbm
- pytorch (core), torchvision, torchaudio, transformers, datasets
- matplotlib, seaborn, plotly, altair, bokeh
- scipy, statsmodels, networkx, igraph
- dask, ray, prefect, dagster, airflow
- mlflow, wandb, optuna, hydra
- spacy, nltk, gensim, sentence-transformers

### Systems/Infrastructure (Rust-heavy, target: 50–70 repos)

High-value candidates:
- tokio, async-std, smol, mio, crossbeam
- tonic (gRPC), tower, hyper, h2
- sled, tikv, raft-rs
- rustls, ring, webpki
- nix, libc, memmap2
- tracing, log, env_logger
- parking_lot, dashmap, flume

### DevOps/Tooling (target: 30–50 repos)

High-value candidates:
- ansible (Python), fabric, invoke
- terraform providers (not Go — skip), pulumi
- docker-py, kubernetes-client
- pre-commit, tox, nox, coverage
- mypy, ruff, pyright (Python tooling from Rust)
- cargo-edit, cargo-deny, cargo-expand

### Messy/Legacy Repos (for quality diversity, target: 90–150 repos)

Discovery strategy: search for repos with high star counts but known complexity:
```
language:python stars:>500 topics:legacy OR topics:refactoring
language:python stars:>200 pushed:<2024-01-01  # stale but popular
language:rust stars:>100 topics:experimental
```

Also include: repos with many open issues relative to stars (complexity signal), repos with long-lived PRs (architectural friction signal).

### TypeScript (large pool, future parser — target: 90–200 registry entries)

High-value candidates (curate now, parse later):
- next.js, nuxt, remix, astro, svelte, solid
- express, fastify, nestjs, hono, trpc
- prisma, drizzle-orm, typeorm, sequelize
- zod, valibot, io-ts, typebox
- commander, inquirer, oclif
- jest, vitest, playwright, cypress
- eslint, prettier, biome
- turborepo, nx, lerna
- react, angular, vue (core — these are huge and architecturally distinct)
- grafana, n8n, supabase, cal.com (full applications)

---

## Appendix B: GitHub API Call Budget

| Stage | API calls per candidate | For 1500 candidates | Notes |
|-------|------------------------|---------------------|-------|
| Discovery (Search) | ~50 paginated requests total | 50 | Split by star-range buckets |
| Pre-filter (repo metadata) | 1 per candidate | 1500 | |
| Contributor count | 1 per candidate | 1500 | Use `per_page=20&page=1` — if 20 returned, passes threshold. No need to paginate further. |
| CI/CD check | 1–3 per candidate | 2000 | Check `.github/workflows` first (short-circuit ~90%) |
| Entrypoint detection | 1–3 per candidate (file reads) | 3000 | Python only; Rust always "." |
| Commit pinning | 1 per candidate | 1500 | |
| Dependency extraction | 1 per candidate | 1500 | |
| Monorepo detection | 1 per candidate (tree API) | 1500 | Can be batched with entrypoint detection |
| **Total** | | **~12,500** | |

At 5000 REST API calls/hour (authenticated via `gh`): **~2.5 hours total.**

With 2s sleep between search requests (30/min limit): search phase takes ~3 minutes.

**Optimization:** Use GitHub GraphQL API to batch repo metadata + default branch + commit SHA + CI/CD directory check in a single query per repo, reducing total calls by ~50%. GraphQL allows querying up to 100 repos per request.

```graphql
# Example GraphQL query for batch metadata
query($cursor: String) {
  search(query: "language:python stars:>100 fork:false", type: REPOSITORY, first: 100, after: $cursor) {
    edges {
      node {
        ... on Repository {
          nameWithOwner
          stargazerCount
          licenseInfo { spdxId }
          pushedAt
          createdAt
          isArchived
          defaultBranchRef { target { oid } }
          object(expression: "HEAD:.github/workflows") { ... on Tree { entries { name } } }
        }
      }
    }
    pageInfo { hasNextPage endCursor }
  }
}
```
