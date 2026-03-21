# topo — build & analysis targets

# ── WASM ───────────────────────────────────────────────────────────

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

# ── Python ─────────────────────────────────────────────────────────

.PHONY: test

## Run all Python tests
test:
	uv run pytest -x -q

# ── Rust ───────────────────────────────────────────────────────────

.PHONY: rust-test rust-build rust-analyze analyze

## Run Rust unit tests (all workspace crates)
rust-test:
	cargo test --workspace

## Build the Rust crates (native, not WASM)
rust-build:
	cargo build --workspace --release

## Run the full analysis on this repo (all Rust packages)
analyze:
	cargo run -p topo-cli --release -- .
