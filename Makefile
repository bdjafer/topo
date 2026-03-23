# topo — build & analysis targets

CACHE := .topo/make
GRAPH := $(CACHE)/graph.json
EMBEDDINGS := $(CACHE)/embeddings.json
TOPO := ./target/release/topo

# ── Analysis ──────────────────────────────────────────────────────

.PHONY: analyze domain

## Ensure the binary is built (only rebuilds if sources changed)
$(TOPO): $(shell find packages -name '*.rs' -not -path '*/target/*' 2>/dev/null)
	cargo build -p topo-cli --release

## Parse (only re-parses if binary is newer — i.e. code changed)
$(GRAPH): $(TOPO)
	@mkdir -p $(CACHE)
	$(TOPO) parse . -o $(GRAPH)

## Embed (only re-embeds if graph changed)
$(EMBEDDINGS): $(GRAPH)
	python3 scripts/generate_embeddings.py $(GRAPH) --timeout 180 -o $(EMBEDDINGS)

## Run structural analysis (issues + health)
analyze: $(EMBEDDINGS)
	$(TOPO) analyze . --embeddings $(EMBEDDINGS)

## Show domain / architecture for this repo
domain: $(EMBEDDINGS)
	$(TOPO) domain . --embeddings $(EMBEDDINGS)

# ── WASM ──────────────────────────────────────────────────────────

.PHONY: wasm-build wasm-analyze wasm-test wasm-dev

## Build the WASM binary (requires wasm-pack + Rust toolchain)
wasm-build:
	cd packages/topo-analyzer && \
		wasm-pack build --target web --features wasm --no-default-features --out-dir pkg

## Parse this codebase and analyze it via WASM
wasm-analyze:
	cargo run -p topo-cli -- parse packages/ -o /tmp/topo-graph.json
	node packages/topo-web/bin/topo-wasm.mjs /tmp/topo-graph.json

## Run WASM end-to-end tests (includes parsing + analysis + assertions)
wasm-test:
	node packages/topo-web/bin/test-e2e.mjs

## Start the browser UI dev server
wasm-dev:
	cd packages/topo-web && pnpm dev

# ── Examples & Benchmark ──────────────────────────────────────────

.PHONY: harvest harvest-all benchmark examples-list

## List all registered example repos
examples-list:
	@./examples/scripts/fetch_and_analyze.sh --list

## Harvest (parse) specific repos: make harvest REPOS="flask click requests"
harvest: $(TOPO)
	python3 benchmark/scripts/harvest_corpus.py $(REPOS)

## Harvest all 50 repos from registry
harvest-all: $(TOPO)
	python3 benchmark/scripts/harvest_corpus.py

## Run mutation benchmark on all available repos
benchmark: $(TOPO)
	python3 benchmark/scripts/evaluate_mutations.py

# ── Dataset Pipeline ─────────────────────────────────────────────

.PHONY: preprocess preprocess-all validate-dataset split-dataset

## Preprocess specific repos: make preprocess REPOS="flask,click"
preprocess: $(TOPO)
	python3 packages/topo-dataset/scripts/preprocess.py --repos $(REPOS)

## Preprocess all parsed repos
preprocess-all: $(TOPO)
	python3 packages/topo-dataset/scripts/preprocess.py

## Validate preprocessed feature files
validate-dataset:
	python3 packages/topo-dataset/scripts/validate.py

## Generate train/val/test splits
split-dataset:
	python3 packages/topo-dataset/scripts/split.py

# ── Registry Curation ────────────────────────────────────────────

.PHONY: curate curate-dry-run curate-fill pin-registry coverage-report

## Expand registry by one batch (200 repos)
curate:
	python3 packages/topo-dataset/scripts/curate_repos.py --batch-size 200

## Preview what curate would add without modifying registry
curate-dry-run:
	python3 packages/topo-dataset/scripts/curate_repos.py --batch-size 200 --dry-run

## Fill a specific domain gap: make curate-fill DOMAIN=data_ml
curate-fill:
	python3 packages/topo-dataset/scripts/curate_repos.py --fill-domain $(DOMAIN) --batch-size 50

## Pin all HEAD commits to concrete SHAs
pin-registry:
	python3 packages/topo-dataset/scripts/pin_existing.py

## Show distributional coverage of current registry
coverage-report:
	python3 packages/topo-dataset/scripts/coverage_report.py

# ── Python ────────────────────────────────────────────────────────

.PHONY: test

## Run all Python tests
test:
	uv run pytest -x -q

# ── Rust ──────────────────────────────────────────────────────────

.PHONY: rust-test rust-build

## Run Rust unit tests (all workspace crates)
rust-test:
	cargo test --workspace

## Build the Rust crates (native, not WASM)
rust-build:
	cargo build --workspace --release
