# Topo — Structural Intelligence for Codebases

## Project Structure

Monorepo with three packages under `packages/`:

- **topo-parser**: Source code → typed multilayer graph. Parses Python codebases into nodes (functions, classes, modules) and edges (calls, imports, inheritance, co-location).
- **topo-analyzer**: Graph → structural intelligence. Spectral decomposition, module detection, role classification, anomaly detection.
- **topo-cli**: Developer-facing interface. CLI commands that run the pipeline and produce human/LLM-readable output.

## Development

```bash
uv sync                    # Install all packages in dev mode
uv run pytest              # Run all tests
uv run topo <path>         # Run the CLI
```

## Architecture Invariants

- Parser knows nothing about analysis. Analyzer knows nothing about the CLI.
- The shared contract between packages is the graph data model in `topo-parser` (`topo_parser.graph`).
- Each package has its own `pyproject.toml` and can be developed/tested independently.

## Key Design Decisions

- Python-first: both the tool implementation and the first parse target.
- Spectral analysis is the core bet — must be validated empirically against known architectures.
- Output must be useful both to humans and as LLM context.
