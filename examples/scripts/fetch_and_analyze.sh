#!/usr/bin/env bash
#
# Fetch, parse, and analyze example codebases from registry.toml.
#
# Usage:
#   ./examples/scripts/fetch_and_analyze.sh            # All examples
#   ./examples/scripts/fetch_and_analyze.sh ripgrep     # Single example
#   ./examples/scripts/fetch_and_analyze.sh --list      # List registered examples
#
# Requires: cargo (built topo-cli), python3.11+, gh (for metadata)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXAMPLES_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$EXAMPLES_DIR")"
REGISTRY="$EXAMPLES_DIR/registry.toml"
CLONE_DIR="/tmp/topo-examples"
TOPO="cargo run -p topo-cli --"

# ── Helpers ──────────────────────────────────────────────────

log()  { echo "▸ $*"; }
err()  { echo "✗ $*" >&2; }
ok()   { echo "✓ $*"; }

# Parse registry.toml using Python's tomllib (stdlib since 3.11)
parse_registry() {
    python3 -c "
import tomllib, json, sys
with open('$REGISTRY', 'rb') as f:
    reg = tomllib.load(f)
examples = reg.get('example', [])
if not examples:
    print('[]')
    sys.exit(0)
json.dump(examples, sys.stdout)
"
}

get_example_json() {
    local name="$1"
    python3 -c "
import tomllib, json, sys
with open('$REGISTRY', 'rb') as f:
    reg = tomllib.load(f)
for ex in reg.get('example', []):
    if ex['name'] == '$name':
        json.dump(ex, sys.stdout)
        sys.exit(0)
print('null')
"
}

list_examples() {
    python3 -c "
import tomllib
with open('$REGISTRY', 'rb') as f:
    reg = tomllib.load(f)
for ex in reg.get('example', []):
    tags = ex.get('tags', {})
    print(f\"  {ex['name']:<16} {ex['language']:<8} {tags.get('size','?'):<8} {tags.get('quality','?'):<8} {ex.get('ref','?')}\")
"
}

# Process a single example
process_example() {
    local name repo commit ref language entrypoint
    name=$(echo "$1" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['name'])")
    repo=$(echo "$1" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['repo'])")
    commit=$(echo "$1" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['commit'])")
    ref=$(echo "$1" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('ref','main'))")
    language=$(echo "$1" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['language'])")
    entrypoint=$(echo "$1" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['entrypoint'])")

    local exclude
    exclude=$(echo "$1" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('cli_overrides',{}).get('exclude',''))")

    local out_dir="$EXAMPLES_DIR/$name"
    local clone_path="$CLONE_DIR/$name"

    log "Processing $name ($language, ref=$ref)"
    mkdir -p "$out_dir"

    # Clone
    if [ -d "$clone_path" ]; then
        log "  Using cached clone at $clone_path"
    else
        log "  Cloning $repo..."
        git clone --depth 50 --branch "$ref" "$repo" "$clone_path" 2>/dev/null || \
            git clone "$repo" "$clone_path" 2>/dev/null
    fi

    # Checkout exact commit if specified
    if [ -n "$commit" ]; then
        (cd "$clone_path" && git checkout "$commit" 2>/dev/null) || true
    fi

    local src_path="$clone_path"
    if [ "$entrypoint" != "." ]; then
        src_path="$clone_path/$entrypoint"
    fi

    # Build CLI args
    local extra_args=""
    if [ -n "$exclude" ]; then
        extra_args="--exclude $exclude"
    fi

    # Parse
    log "  Parsing..."
    (cd "$PROJECT_ROOT" && $TOPO parse "$src_path" --language "$language" $extra_args -o "$out_dir/graph.json") || {
        err "  Parse failed for $name"
        return 1
    }

    # Analyze (JSON)
    log "  Analyzing (JSON)..."
    (cd "$PROJECT_ROOT" && $TOPO analyze --input "$out_dir/graph.json" --format json > "$out_dir/analysis.json") || {
        err "  Analysis failed for $name"
        return 1
    }

    # Analyze (text)
    log "  Analyzing (text)..."
    (cd "$PROJECT_ROOT" && $TOPO analyze --input "$out_dir/graph.json" > "$out_dir/analysis.txt") || {
        err "  Text analysis failed for $name"
        return 1
    }

    # Collect metadata
    log "  Collecting metadata..."
    python3 "$SCRIPT_DIR/collect_metadata.py" "$name" "$out_dir" "$repo" || {
        err "  Metadata collection failed for $name (non-fatal)"
    }

    ok "Done: $name"
}

# ── Main ─────────────────────────────────────────────────────

if [ "${1:-}" = "--list" ]; then
    echo "Registered examples:"
    list_examples
    exit 0
fi

mkdir -p "$CLONE_DIR"

if [ -n "${1:-}" ] && [ "$1" != "--list" ]; then
    # Single example
    example_json=$(get_example_json "$1")
    if [ "$example_json" = "null" ]; then
        err "Example '$1' not found in registry"
        exit 1
    fi
    process_example "$example_json"
else
    # All examples
    parse_registry | python3 -c "
import json, sys
examples = json.load(sys.stdin)
for ex in examples:
    print(json.dumps(ex))
" | while IFS= read -r line; do
        process_example "$line" || true
    done
fi

log "Cleaning up clones..."
# Optionally: rm -rf "$CLONE_DIR"
log "Done. Clones preserved in $CLONE_DIR for re-runs."
