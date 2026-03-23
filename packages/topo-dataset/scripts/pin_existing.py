#!/usr/bin/env python3
"""
Pin all HEAD-committed repos in registry.toml to concrete SHAs.

Resolves each entry with commit = "HEAD" to the current default branch HEAD
SHA via the GitHub API (using `gh api`). Uses tomlkit for round-trip-safe
TOML editing (preserves comments and formatting).

Usage:
    python pin_existing.py              # Pin all HEAD entries
    python pin_existing.py --dry-run    # Preview without modifying registry
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import tomlkit

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
REGISTRY = PROJECT_ROOT / "examples" / "registry.toml"


def gh_api(endpoint: str) -> dict | None:
    """Call GitHub API via gh CLI. Returns None on failure."""
    try:
        result = subprocess.run(
            ["gh", "api", endpoint],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        pass
    return None


def extract_owner_repo(url: str) -> str:
    """Extract 'owner/repo' from GitHub URL."""
    url = url.rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    parts = url.split("/")
    return f"{parts[-2]}/{parts[-1]}"


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


def main():
    parser = argparse.ArgumentParser(
        description="Pin all HEAD-committed repos in registry.toml to concrete SHAs."
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without modifying registry")
    args = parser.parse_args()

    doc = tomlkit.parse(REGISTRY.read_text())
    examples = doc.get("example", [])

    unpinned = [e for e in examples if e.get("commit") == "HEAD"]
    if not unpinned:
        print("All repos already pinned.")
        return

    print(f"Found {len(unpinned)} unpinned repos (commit = \"HEAD\")")

    pinned_count = 0
    failed = []

    for entry in unpinned:
        name = entry["name"]
        owner_repo = extract_owner_repo(entry["repo"])
        sha = resolve_commit_sha(owner_repo)

        if sha:
            if args.dry_run:
                print(f"  [dry-run] {name} → {sha}")
            else:
                entry["commit"] = sha
                print(f"  Pinned {name} → {sha}")
            pinned_count += 1
        else:
            print(f"  FAILED {name} ({owner_repo})")
            failed.append(name)

    if args.dry_run:
        print(f"\nDry run: would pin {pinned_count} repos, {len(failed)} failed")
        return

    # Verify round-trip: parse the output and check it matches
    output = tomlkit.dumps(doc)
    verify = tomlkit.parse(output)
    if len(verify["example"]) != len(doc["example"]):
        print("ERROR: TOML round-trip lost entries! Aborting.", file=sys.stderr)
        sys.exit(1)

    REGISTRY.write_text(output)
    print(f"\nDone: pinned {pinned_count} repos, {len(failed)} failed")
    if failed:
        print(f"Failed repos: {', '.join(failed)}")


if __name__ == "__main__":
    main()
