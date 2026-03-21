---
name: test
description: Run all tests for the topo workspace (Rust + Python). Use after implementation to verify nothing is broken.
allowed-tools: Bash, Read
context: fork
agent: build-test
---

Run the full test suite for the topo workspace.

## Steps

1. Build and run Rust tests:
```bash
cd /Users/bryandjafer/Documents/personal/topo && cargo test --workspace 2>&1
```

2. Run Python tests:
```bash
cd /Users/bryandjafer/Documents/personal/topo && uv run pytest tests/ -x 2>&1
```

## Reporting

Report:
- BUILD: OK/FAILED
- RUST TESTS: N passed, M failed
- PYTHON TESTS: N passed, M failed
- If any failures: root cause + file:line + fix suggestion
