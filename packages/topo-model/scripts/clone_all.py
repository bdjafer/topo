#!/usr/bin/env python3
"""Clone all repos from registry.toml for embedding extraction.

Separate from harvest_corpus.py — only clones, does NOT parse.
Used to ensure all repos have cloned source code available for
the CodeLM embedding pipeline.

Usage:
    python clone_all.py              # Clone all missing repos
    python clone_all.py flask click  # Clone specific repos
"""

import subprocess
import sys
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
REGISTRY = PROJECT_ROOT / "examples" / "registry.toml"
CLONE_DIR = Path("/tmp/topo-corpus")


def load_manifest() -> list[dict]:
    with open(REGISTRY, "rb") as f:
        reg = tomllib.load(f)
    return [
        {
            "name": ex["name"],
            "url": ex["repo"],
            "commit": ex.get("commit", "HEAD"),
        }
        for ex in reg.get("example", [])
    ]


def clone_repo(entry: dict) -> bool:
    name = entry["name"]
    clone_path = CLONE_DIR / name

    if clone_path.exists():
        print(f"  [cached] {name}")
        return True

    url = entry["url"]
    commit = entry.get("commit", "HEAD")

    try:
        if commit == "HEAD":
            subprocess.run(
                ["git", "clone", "--depth", "1", url, str(clone_path)],
                capture_output=True, timeout=120, check=True,
            )
        else:
            subprocess.run(
                ["git", "clone", "--depth", "50", url, str(clone_path)],
                capture_output=True, timeout=120, check=True,
            )
            subprocess.run(
                ["git", "checkout", commit],
                cwd=clone_path, capture_output=True, timeout=10,
            )
        print(f"  [cloned] {name}")
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"  [FAILED] {name}: {e}")
        return False


def main():
    manifest = load_manifest()
    targets = [a for a in sys.argv[1:] if not a.startswith("-")]
    if targets:
        manifest = [e for e in manifest if e["name"] in targets]

    CLONE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Cloning {len(manifest)} repos to {CLONE_DIR}")

    ok = 0
    fail = 0
    for entry in manifest:
        if clone_repo(entry):
            ok += 1
        else:
            fail += 1

    print(f"\nDone: {ok} ok, {fail} failed")


if __name__ == "__main__":
    main()
