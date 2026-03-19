# topo — build & analysis targets

# ── WASM ───────────────────────────────────────────────────────────

.PHONY: wasm-build wasm-analyze wasm-test wasm-dev

## Build the WASM binary (requires wasm-pack + Rust toolchain)
wasm-build:
	cd packages/topo-core && \
		wasm-pack build --target web --features wasm --no-default-features --out-dir pkg

## Parse this codebase and analyze it via WASM
wasm-analyze:
	uv run topo parse packages/ -o /tmp/topo-graph.json
	node packages/topo-web/bin/topo-wasm.mjs /tmp/topo-graph.json

## Run WASM end-to-end tests (includes parsing + analysis + assertions)
wasm-test:
	node packages/topo-web/bin/test-e2e.mjs

## Start the browser UI dev server
wasm-dev:
	cd packages/topo-web && pnpm dev

# ── Python ─────────────────────────────────────────────────────────

.PHONY: test lint

## Run all Python tests
test:
	uv run pytest -x -q

## Run the full analysis on this codebase via Python
analyze:
	uv run topo packages/

# ── Rust ───────────────────────────────────────────────────────────

.PHONY: rust-test rust-build

## Run Rust unit tests
rust-test:
	cd packages/topo-core && cargo test

## Build the Rust crate (native, not WASM)
rust-build:
	cd packages/topo-core && cargo build --release
