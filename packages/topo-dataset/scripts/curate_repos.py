#!/usr/bin/env python3
"""
Expand registry.toml via GitHub API discovery.

6-stage pipeline: Discovery → Pre-filter → Entrypoint detection →
Registry ingestion (pin + tag) → Parse + post-filter → Tag + balance.

This script covers stages 1–4 (API-based). Stages 5–6 are handled by
harvest_corpus.py, validate.py, and coverage_report.py.

Usage:
    python curate_repos.py                                  # Default: +200 repos
    python curate_repos.py --batch-size 50 --dry-run        # Preview 50 candidates
    python curate_repos.py --languages python               # Python only
    python curate_repos.py --fill-domain data_ml            # Target-fill a domain
    python curate_repos.py --resume                         # Resume from cache
"""

import argparse
import json
import re
import subprocess
import sys
import time
import tomllib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import tomlkit

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples"
REGISTRY = EXAMPLES_DIR / "registry.toml"
DEFAULT_CACHE_DIR = EXAMPLES_DIR / ".cache"

# ── OSI-approved licenses (common subset) ─────────────────────────

OSI_LICENSES = {
    "MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC",
    "MPL-2.0", "LGPL-2.1", "LGPL-3.0", "GPL-2.0", "GPL-3.0",
    "AGPL-3.0", "Unlicense", "0BSD", "Artistic-2.0", "Zlib",
    "BSL-1.0", "PostgreSQL", "EUPL-1.2", "WTFPL",
    # SPDX variants
    "LGPL-2.1-only", "LGPL-3.0-only", "GPL-2.0-only", "GPL-3.0-only",
    "AGPL-3.0-only", "LGPL-2.1-or-later", "LGPL-3.0-or-later",
    "GPL-2.0-or-later", "GPL-3.0-or-later", "AGPL-3.0-or-later",
}

# ── Star-range buckets for search API (max 1000 results per query) ─

STAR_BUCKETS = [
    (100, 200),
    (200, 500),
    (500, 2000),
    (2000, 10000),
    (10000, None),  # 10000+
]

# ── Domain detection signals ──────────────────────────────────────

DOMAIN_SIGNALS: dict[str, dict[str, list[str]]] = {
    "web": {
        "python": ["django", "flask", "fastapi", "starlette", "aiohttp", "tornado", "sanic", "pyramid", "bottle"],
        "rust": ["actix-web", "axum", "rocket", "warp", "hyper", "tower-http", "poem"],
        "typescript": ["express", "nestjs", "next", "nuxt", "fastify", "hono", "koa", "hapi", "remix"],
    },
    "cli": {
        "python": ["click", "typer", "fire", "rich", "prompt-toolkit", "textual"],
        "rust": ["clap", "structopt", "argh", "dialoguer", "indicatif", "ratatui"],
        "typescript": ["commander", "yargs", "inquirer", "chalk", "ora", "oclif"],
    },
    "data_ml": {
        "python": [
            "pandas", "numpy", "scipy", "sklearn", "torch", "tensorflow", "polars",
            "matplotlib", "seaborn", "xgboost", "transformers", "datasets",
        ],
        "rust": ["ndarray", "polars", "candle", "burn", "linfa", "smartcore"],
        "typescript": ["tensorflow", "onnxruntime-node", "ml5", "danfojs"],
    },
    "systems": {
        "python": [
            "celery", "redis", "kafka-python", "grpcio", "aioredis", "zmq",
            "kombu", "dramatiq", "huey", "rq",
        ],
        "rust": ["tokio", "async-std", "mio", "crossbeam", "parking_lot", "rayon"],
        "typescript": ["ioredis", "kafkajs", "bullmq", "amqplib", "grpc-js"],
    },
    "devops": {
        "python": ["ansible", "fabric", "invoke", "boto3", "docker", "kubernetes"],
        "rust": ["bollard", "kube", "k8s-openapi"],
        "typescript": ["aws-cdk-lib", "pulumi", "projen", "cdktf"],
    },
}

# Targeted search queries for filling domain gaps.
# Keyed by (domain, language). Only relevant language/domain combos are listed.
TARGETED_QUERIES: dict[str, dict[str, list[str]]] = {
    "data_ml": {
        "python": [
            "topic:machine-learning stars:>100 fork:false",
            "topic:data-science stars:>100 fork:false",
            "topic:deep-learning stars:>100 fork:false",
            "topic:pytorch stars:>100 fork:false",
        ],
        "rust": [
            "topic:machine-learning stars:>50 fork:false",
        ],
    },
    "systems": {
        "python": [
            "topic:distributed stars:>100 fork:false",
        ],
        "rust": [
            "topic:async stars:>50 fork:false",
            "topic:networking stars:>50 fork:false",
        ],
    },
    "devops": {
        "python": [
            "topic:devops stars:>100 fork:false",
            "topic:infrastructure stars:>100 fork:false",
        ],
    },
    "cli": {
        "python": [
            "topic:cli stars:>100 fork:false",
            "topic:command-line stars:>100 fork:false",
        ],
        "rust": [
            "topic:cli stars:>50 fork:false",
            "topic:command-line stars:>50 fork:false",
        ],
    },
}


# ── GitHub API helpers ────────────────────────────────────────────

def gh_api(endpoint: str, method: str = "GET") -> dict | list | None:
    """Call GitHub API via gh CLI. Returns None on failure."""
    try:
        cmd = ["gh", "api", endpoint]
        if method != "GET":
            cmd.extend(["-X", method])
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        pass
    return None


def gh_search_repos(query: str, per_page: int = 100, max_results: int = 300) -> list[dict]:
    """Search GitHub repos via REST Search API. Returns list of repo dicts."""
    results = []
    page = 1
    encoded_query = quote(query, safe="")
    while len(results) < max_results:
        endpoint = f"search/repositories?q={encoded_query}&sort=stars&order=desc&per_page={per_page}&page={page}"
        data = gh_api(endpoint)
        if not data or "items" not in data:
            break
        items = data["items"]
        if not items:
            break
        results.extend(items)
        if len(items) < per_page:
            break
        page += 1
        # Respect rate limit: 30 search requests/minute
        time.sleep(2)
    return results[:max_results]


def gh_get_file(owner_repo: str, path: str) -> str | None:
    """Get raw file content from a repo via GitHub Contents API."""
    try:
        result = subprocess.run(
            ["gh", "api", f"repos/{owner_repo}/contents/{path}",
             "-H", "Accept: application/vnd.github.raw+json"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            return result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def gh_path_exists(owner_repo: str, path: str) -> bool:
    """Check if a path exists in a repo via GitHub Contents API."""
    try:
        result = subprocess.run(
            ["gh", "api", f"repos/{owner_repo}/contents/{path}", "--silent"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def get_contributor_count(owner_repo: str) -> int:
    """Get approximate contributor count. Returns count or 0 on failure."""
    data = gh_api(f"repos/{owner_repo}/contributors?per_page=20&page=1&anon=false")
    if isinstance(data, list):
        return len(data)
    return 0


def extract_owner_repo(url: str) -> str:
    """Extract 'owner/repo' from GitHub URL."""
    url = url.rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    parts = url.split("/")
    return f"{parts[-2]}/{parts[-1]}"


def parse_date(date_str: str) -> datetime:
    """Parse ISO date string to datetime."""
    return datetime.fromisoformat(date_str.replace("Z", "+00:00"))


# ── Stage 1: Discovery ───────────────────────────────────────────

def discover_candidates(
    language: str,
    max_per_bucket: int = 300,
    cache_dir: Path | None = None,
    fill_domain: str | None = None,
    resume: bool = False,
) -> list[dict]:
    """Query GitHub Search API across star-range buckets."""

    # Check cache first (only when --resume)
    if cache_dir and resume:
        suffix = f"_{fill_domain}" if fill_domain else ""
        cache_file = cache_dir / f"discovery_{language}{suffix}_{datetime.now().strftime('%Y%m%d')}.json"
        if cache_file.exists():
            print(f"  Loading cached discovery results: {cache_file.name}")
            return json.loads(cache_file.read_text())

    candidates = []

    if fill_domain and fill_domain in TARGETED_QUERIES:
        # Targeted domain-filling queries (language-specific)
        lang_queries = TARGETED_QUERIES[fill_domain].get(language, [])
        if not lang_queries:
            print(f"  No targeted queries for {fill_domain}/{language}, skipping")
        for query_suffix in lang_queries:
            query = f"language:{language} {query_suffix}"
            print(f"  Searching: {query}")
            results = gh_search_repos(query, per_page=100, max_results=max_per_bucket)
            candidates.extend(results)
            time.sleep(2)
    else:
        # Standard star-range bucketed search
        for low, high in STAR_BUCKETS:
            star_query = f"stars:{low}..{high}" if high else f"stars:>={low}"
            query = f"language:{language} {star_query} fork:false archived:false"
            print(f"  Searching: {query}")
            results = gh_search_repos(query, per_page=100, max_results=max_per_bucket)
            candidates.extend(results)
            time.sleep(2)

    # Deduplicate by full_name
    seen = set()
    deduped = []
    for r in candidates:
        key = r["full_name"]
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    # Cache results
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_{fill_domain}" if fill_domain else ""
        cache_file = cache_dir / f"discovery_{language}{suffix}_{datetime.now().strftime('%Y%m%d')}.json"
        cache_file.write_text(json.dumps(deduped, indent=2, default=str))
        print(f"  Cached {len(deduped)} candidates → {cache_file.name}")

    return deduped


# ── Stage 2: Pre-filter ──────────────────────────────────────────

def has_ci_cd(owner_repo: str) -> bool:
    """Check for CI/CD configuration via GitHub Contents API."""
    # Check .github/workflows/ first (most common, covers ~90%)
    if gh_path_exists(owner_repo, ".github/workflows"):
        return True
    if gh_path_exists(owner_repo, ".travis.yml"):
        return True
    if gh_path_exists(owner_repo, ".circleci/config.yml"):
        return True
    return False


def pre_filter(repo: dict) -> tuple[bool, str]:
    """Check if a repo passes pre-filter. Returns (pass, reason)."""
    now = datetime.now(timezone.utc)

    # License check: reject repos with no license (legally unusable) or non-OSI
    license_info = repo.get("license") or {}
    license_id = license_info.get("spdx_id", "")
    if not license_id or (license_id not in OSI_LICENSES and license_id != "NOASSERTION"):
        return False, f"license:{license_id or 'none'}"

    # Activity check
    pushed_at = repo.get("pushed_at")
    created_at = repo.get("created_at")
    if not pushed_at or not created_at:
        return False, "missing_dates"

    pushed = parse_date(pushed_at)
    created = parse_date(created_at)
    age_years = (now - created).days / 365
    recent = (now - pushed).days < 365 * 3

    if not recent:
        return False, "inactive"

    # Size check (rough proxy — GitHub 'size' is git repo size in KB)
    if repo.get("size", 0) < 50:
        return False, f"size:{repo.get('size', 0)}KB"

    # Contributor check (relaxed: ≥20 OR ≥2y active)
    # For Data/ML repos (stars ≥ 50 as proxy): further relax to ≥10 contributors
    contributors = get_contributor_count(repo["full_name"])
    stars = repo.get("stargazers_count", 0)
    min_contributors = 10 if stars >= 50 else 20
    if contributors < min_contributors and age_years < 2:
        return False, f"contributors:{contributors},age:{age_years:.1f}y"

    # CI/CD check
    if not has_ci_cd(repo["full_name"]):
        return False, "no_ci_cd"

    return True, "ok"


# ── Stage 3: Entrypoint Detection ────────────────────────────────

def detect_python_entrypoint(owner_repo: str) -> str | None:
    """Auto-detect Python package entrypoint from project config."""

    # Try pyproject.toml first
    content = gh_get_file(owner_repo, "pyproject.toml")
    if content:
        try:
            parsed = tomllib.loads(content)
        except Exception:
            parsed = {}

        # Extract project name (needed by multiple methods)
        project_name = parsed.get("project", {}).get("name", "").replace("-", "_")

        # Method 1: [tool.setuptools.packages.find] where = ["src"]
        packages = parsed.get("tool", {}).get("setuptools", {}).get("packages", {})
        find = packages.get("find", {}) if isinstance(packages, dict) else {}
        where = find.get("where", []) if isinstance(find, dict) else []
        if where and project_name:
            candidate = f"{where[0]}/{project_name}"
            if gh_path_exists(owner_repo, candidate):
                return candidate
            # Fallback: maybe the where dir IS the package
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
        match = re.search(r'package_dir\s*=\s*\{\s*["\'][\s]*["\']\s*:\s*["\'](\w+)["\']', content)
        parent_dir = match.group(1) if match else None

        if not parent_dir:
            match = re.search(r'find_packages\(\s*["\'](\w+)["\']', content)
            parent_dir = match.group(1) if match else None

        # Extract project name from setup.py
        name_match = re.search(r'name\s*=\s*["\']([^"\']+)["\']', content)
        setup_name = name_match.group(1).replace("-", "_") if name_match else None

        if parent_dir and setup_name:
            candidate = f"{parent_dir}/{setup_name}"
            if gh_path_exists(owner_repo, candidate):
                return candidate

    # Fallback: look for top-level directory matching repo name
    repo_name = owner_repo.split("/")[-1].replace("-", "_")
    if gh_path_exists(owner_repo, f"src/{repo_name}"):
        return f"src/{repo_name}"
    if gh_path_exists(owner_repo, repo_name):
        return repo_name

    # Final fallback: look for any top-level directory with __init__.py
    # This catches repos where the package name doesn't match the repo name
    tree = gh_api(f"repos/{owner_repo}/contents/")
    if isinstance(tree, list):
        dirs = [item["name"] for item in tree if item["type"] == "dir"]
        for d in dirs:
            if d.startswith(".") or d.startswith("_") or d in (
                "tests", "test", "docs", "doc", "examples", "scripts",
                "benchmarks", "bench", "tools", "ci", "build", "dist",
                "vendor", "third_party", "node_modules",
            ):
                continue
            if gh_path_exists(owner_repo, f"{d}/__init__.py"):
                return d

    # Also check src/ subdirectories
    src_tree = gh_api(f"repos/{owner_repo}/contents/src")
    if isinstance(src_tree, list):
        dirs = [item["name"] for item in src_tree if item["type"] == "dir"]
        for d in dirs:
            if not d.startswith(".") and not d.startswith("_"):
                if gh_path_exists(owner_repo, f"src/{d}/__init__.py"):
                    return f"src/{d}"

    return None


def detect_typescript_entrypoint(owner_repo: str) -> str:
    """Auto-detect TypeScript entrypoint. Returns best guess (never None)."""
    # Check tsconfig.json for rootDir
    content = gh_get_file(owner_repo, "tsconfig.json")
    if content:
        try:
            # tsconfig may have comments — try parsing anyway
            # Strip single-line comments for basic JSON parse
            cleaned = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
            cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
            tsconfig = json.loads(cleaned)
            root_dir = tsconfig.get("compilerOptions", {}).get("rootDir")
            if root_dir and gh_path_exists(owner_repo, root_dir):
                return root_dir
        except json.JSONDecodeError:
            pass

    # Check for src/ directory
    if gh_path_exists(owner_repo, "src"):
        return "src"

    return "."


# ── Stage 4: Registry Ingestion ──────────────────────────────────

def resolve_commit_sha(owner_repo: str) -> str | None:
    """Resolve default branch HEAD to a concrete 12-char SHA."""
    repo_info = gh_api(f"repos/{owner_repo}")
    if not repo_info:
        return None
    branch = repo_info.get("default_branch", "main")
    commit_info = gh_api(f"repos/{owner_repo}/commits/{branch}")
    if not commit_info:
        return None
    return commit_info["sha"][:12]


def extract_dependencies(owner_repo: str, language: str) -> list[str]:
    """Extract dependency names from project manifest via GitHub Contents API."""
    if language == "python":
        content = gh_get_file(owner_repo, "pyproject.toml")
        if content:
            try:
                parsed = tomllib.loads(content)
                deps = parsed.get("project", {}).get("dependencies", [])
                return [re.split(r"[><=!~\[]", d)[0].strip().lower() for d in deps]
            except Exception:
                pass
        # Fallback: requirements.txt
        content = gh_get_file(owner_repo, "requirements.txt")
        if content:
            return [
                re.split(r"[><=!~\[]", line)[0].strip().lower()
                for line in content.splitlines()
                if line.strip() and not line.startswith("#") and not line.startswith("-")
            ]

    elif language == "rust":
        content = gh_get_file(owner_repo, "Cargo.toml")
        if content:
            try:
                parsed = tomllib.loads(content)
                deps = list(parsed.get("dependencies", {}).keys())
                deps += list(parsed.get("workspace", {}).get("dependencies", {}).keys())
                return deps
            except Exception:
                pass

    elif language == "typescript":
        content = gh_get_file(owner_repo, "package.json")
        if content:
            try:
                pkg = json.loads(content)
                deps = list(pkg.get("dependencies", {}).keys())
                deps += list(pkg.get("devDependencies", {}).keys())
                # Strip @scope/ prefix
                return [d.split("/")[-1] for d in deps]
            except json.JSONDecodeError:
                pass

    return []


def detect_domain(
    deps: list[str],
    language: str,
    has_pyproject: bool = False,
    has_cargo_lib: bool = False,
) -> str:
    """Classify repo domain from its dependencies."""
    for domain, lang_signals in DOMAIN_SIGNALS.items():
        signals = lang_signals.get(language, [])
        if any(dep in signals for dep in deps):
            return domain

    # If no domain-specific signals matched, classify as "library" if published package
    if has_pyproject or has_cargo_lib:
        return "library"
    return "other"


def classify_size(repo: dict) -> str:
    """Classify repo size from GitHub API 'size' field (KB). Pre-parse heuristic."""
    size_kb = repo.get("size", 0)
    if size_kb < 2000:
        return "small"
    elif size_kb < 20000:
        return "medium"
    else:
        return "large"


def normalize_name(name: str) -> str:
    """Normalize a repo name for deduplication."""
    return name.lower().replace("-", "_").replace(".", "_")


def generate_registry_entry(
    repo: dict, entrypoint: str, sha: str, domain: str, batch: int = 0,
) -> str:
    """Generate a TOML [[example]] block as a string."""
    name = repo["name"]
    language = repo["language"].lower()
    size = classify_size(repo)
    # Escape TOML special characters in description
    description = (repo.get("description") or "")[:100]
    description = description.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")

    return f"""[[example]]
name = "{name}"
repo = "{repo['html_url']}"
commit = "{sha}"
language = "{language}"
entrypoint = "{entrypoint}"
description = "{description}"
batch = {batch}

[example.tags]
size = "{size}"
domain = "{domain}"
"""


def apply_org_cap(entries: list[dict], max_per_org: int = 8) -> list[dict]:
    """Cap repos per GitHub organization."""
    org_counts: Counter[str] = Counter()
    result = []
    for entry in entries:
        org = entry["repo"]["full_name"].split("/")[0]
        if org_counts[org] < max_per_org:
            result.append(entry)
            org_counts[org] += 1
    return result


# ── Registry I/O ─────────────────────────────────────────────────

def load_registry() -> list[dict]:
    """Load registry entries from registry.toml."""
    with open(REGISTRY, "rb") as f:
        reg = tomllib.load(f)
    return reg.get("example", [])


def append_to_registry(entries: list[dict], registry_path: Path, batch: int = 0) -> None:
    """Append new [[example]] blocks to registry.toml."""
    new_blocks = []
    for entry in entries:
        block = generate_registry_entry(
            entry["repo"],
            entry["entrypoint"],
            entry["sha"],
            entry["domain"],
            batch=batch,
        )
        new_blocks.append(block)

    # Append to file with a section header
    separator = f"\n# ──────────────────────────────────────────────────────────────\n"
    separator += f"# Auto-curated — {datetime.now(timezone.utc).strftime('%Y-%m-%d')}\n"
    separator += f"# ──────────────────────────────────────────────────────────────\n\n"

    with open(registry_path, "a") as f:
        f.write(separator)
        f.write("\n".join(new_blocks))


# ── Logging ──────────────────────────────────────────────────────

def log_skip(repo: dict, reason: str, log_file: Path | None = None) -> None:
    """Log a skipped repo."""
    entry = {
        "name": repo.get("full_name", repo.get("name", "?")),
        "reason": reason,
        "stars": repo.get("stargazers_count", 0),
    }
    if log_file:
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")


# ── Main Pipeline ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Expand registry.toml via GitHub API discovery."
    )
    parser.add_argument("--target", type=int, default=800,
                        help="Target number of repos in registry (default: 800)")
    parser.add_argument("--batch-size", type=int, default=200,
                        help="Repos to add per batch (default: 200)")
    parser.add_argument("--languages", nargs="+",
                        choices=["python", "rust", "typescript", "all"],
                        default=["all"],
                        help="Languages to search (default: all)")
    parser.add_argument("--fill-domain", type=str, default=None,
                        choices=list(DOMAIN_SIGNALS.keys()),
                        help="Target a specific underrepresented domain")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print candidates without modifying registry")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from cached discovery results")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                        help=f"Cache directory (default: {DEFAULT_CACHE_DIR})")
    parser.add_argument("--max-per-bucket", type=int, default=300,
                        help="Max candidates per star-range bucket (default: 300)")
    args = parser.parse_args()

    # Resolve language list
    if "all" in args.languages:
        languages = ["python", "rust", "typescript"]
    else:
        languages = args.languages

    # Load existing registry
    registry = load_registry()
    existing_names = {normalize_name(e["name"]) for e in registry}
    existing_urls = {e["repo"] for e in registry}
    # Compute next batch number from existing entries
    max_batch = max((e.get("batch", 0) for e in registry), default=0)
    next_batch = max_batch + 1
    print(f"Existing registry: {len(registry)} repos (next batch: {next_batch})")

    # Set up cache — always cache results; --resume loads from cache instead of re-querying
    cache_dir = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    skip_log = cache_dir / f"skipped_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

    # ── Stage 1: Discovery ──
    print("\n=== Stage 1: Discovery ===")
    candidates = []
    for language in languages:
        print(f"\n  [{language}]")
        raw = discover_candidates(
            language,
            max_per_bucket=args.max_per_bucket,
            cache_dir=cache_dir,
            fill_domain=args.fill_domain,
            resume=args.resume,
        )
        candidates.extend(raw)

    print(f"\nDiscovered {len(candidates)} raw candidates")

    # ── Stage 2: Pre-filter ──
    print("\n=== Stage 2: Pre-filter ===")
    filtered = []
    skipped_reasons: Counter[str] = Counter()

    for i, repo in enumerate(candidates):
        if i % 50 == 0 and i > 0:
            print(f"  Pre-filtering... {i}/{len(candidates)}")

        # Skip existing repos
        if repo.get("html_url", "") in existing_urls:
            skipped_reasons["already_in_registry"] += 1
            continue
        if normalize_name(repo.get("name", "")) in existing_names:
            skipped_reasons["name_collision"] += 1
            continue

        passed, reason = pre_filter(repo)
        if passed:
            filtered.append(repo)
        else:
            skipped_reasons[reason] += 1
            log_skip(repo, reason, skip_log)

    print(f"\n  {len(filtered)} pass pre-filter")
    print(f"  Skipped breakdown:")
    for reason, count in skipped_reasons.most_common():
        print(f"    {reason}: {count}")

    # ── Stage 3: Entrypoint Detection ──
    print("\n=== Stage 3: Entrypoint Detection ===")
    with_entrypoints = []
    entrypoint_failures = 0

    for i, repo in enumerate(filtered):
        if i % 50 == 0 and i > 0:
            print(f"  Detecting entrypoints... {i}/{len(filtered)}")

        language = repo["language"].lower()
        if language == "rust":
            entrypoint = "."
        elif language == "typescript":
            entrypoint = detect_typescript_entrypoint(repo["full_name"])
        else:
            entrypoint = detect_python_entrypoint(repo["full_name"])

        if entrypoint:
            repo["_entrypoint"] = entrypoint
            with_entrypoints.append(repo)
        else:
            entrypoint_failures += 1
            log_skip(repo, "no_entrypoint", skip_log)

    print(f"\n  {len(with_entrypoints)} with resolved entrypoints")
    print(f"  {entrypoint_failures} entrypoint detection failures")

    # ── Stage 4: Pin Commits + Detect Domain ──
    print("\n=== Stage 4: Registry Ingestion ===")
    entries = []
    pin_failures = 0

    for i, repo in enumerate(with_entrypoints):
        if i % 50 == 0 and i > 0:
            print(f"  Pinning commits... {i}/{len(with_entrypoints)}")

        sha = resolve_commit_sha(repo["full_name"])
        if not sha:
            pin_failures += 1
            log_skip(repo, "pin_failed", skip_log)
            continue

        language = repo["language"].lower()
        deps = extract_dependencies(repo["full_name"], language)

        # Detect if this is a published package
        has_pyproject = gh_path_exists(repo["full_name"], "pyproject.toml") if language == "python" else False
        has_cargo_lib = False
        if language == "rust":
            cargo_content = gh_get_file(repo["full_name"], "Cargo.toml")
            if cargo_content:
                try:
                    has_cargo_lib = "lib" in tomllib.loads(cargo_content)
                except Exception:
                    pass

        domain = detect_domain(deps, language, has_pyproject, has_cargo_lib)

        entries.append({
            "repo": repo,
            "sha": sha,
            "entrypoint": repo["_entrypoint"],
            "domain": domain,
            "deps": deps,
        })

    print(f"\n  {len(entries)} entries ready")
    print(f"  {pin_failures} pin failures")

    # Apply org cap (max 8 per org)
    before_cap = len(entries)
    entries = apply_org_cap(entries, max_per_org=8)
    if len(entries) < before_cap:
        print(f"  Org cap removed {before_cap - len(entries)} entries")

    # Take batch_size
    entries = entries[:args.batch_size]
    print(f"\n  Final batch: {len(entries)} repos")

    # Domain distribution preview
    domain_dist: Counter[str] = Counter()
    lang_dist: Counter[str] = Counter()
    for e in entries:
        domain_dist[e["domain"]] += 1
        lang_dist[e["repo"]["language"].lower()] += 1
    print(f"\n  Domain distribution:")
    for domain, count in domain_dist.most_common():
        print(f"    {domain}: {count}")
    print(f"  Language distribution:")
    for lang, count in lang_dist.most_common():
        print(f"    {lang}: {count}")

    if args.dry_run:
        print(f"\n=== Dry Run: {len(entries)} candidates ===")
        for e in entries:
            print(f"  {e['repo']['full_name']:<40} [{e['domain']:<10}] @ {e['sha']} → {e['entrypoint']}")
        return

    # Append to registry
    append_to_registry(entries, REGISTRY, batch=next_batch)
    print(f"\nAdded {len(entries)} repos to registry.toml")
    print(f"Skip log: {skip_log}")

    # Cache the processed entries
    entries_cache = cache_dir / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    serializable = [
        {
            "name": e["repo"]["name"],
            "full_name": e["repo"]["full_name"],
            "sha": e["sha"],
            "entrypoint": e["entrypoint"],
            "domain": e["domain"],
            "language": e["repo"]["language"].lower(),
        }
        for e in entries
    ]
    entries_cache.write_text(json.dumps(serializable, indent=2))
    print(f"Batch log: {entries_cache}")


if __name__ == "__main__":
    main()
