#!/usr/bin/env python3
"""
Generate stratified train/val/test splits for the topo dataset.

Reads examples/registry.toml for tag metadata and examples/*/features.npz
for availability. Produces examples/splits/{train,val,test}.txt.

Usage:
    python split.py                    # Default 80/10/10 split
    python split.py --seed 42          # Reproducible split
    python split.py --train 0.8 --val 0.1 --test 0.1
"""

import argparse
import json
import sys
import tomllib
from pathlib import Path

from sklearn.model_selection import StratifiedShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples"
REGISTRY = EXAMPLES_DIR / "registry.toml"


def load_registry() -> dict[str, dict]:
    """Load registry entries keyed by name."""
    with open(REGISTRY, "rb") as f:
        reg = tomllib.load(f)
    return {ex["name"]: ex for ex in reg.get("example", [])}


def classify_repo(entry: dict) -> str:
    """Classify a repo into an architectural style category for stratification.

    Uses tags from registry.toml. Falls back to language-based classification.
    """
    tags = entry.get("tags", {})
    patterns = tags.get("pattern", [])
    if isinstance(patterns, str):
        patterns = [patterns]

    # Pattern-based classification
    pattern_set = set(p.lower() for p in patterns)

    web_patterns = {"web", "api", "rest", "http", "server", "layered", "sansio"}
    lib_patterns = {"library", "framework", "plugin-system", "extensible"}
    cli_patterns = {"cli", "command-line", "terminal"}
    data_patterns = {"data", "ml", "pipeline", "etl", "scientific"}
    systems_patterns = {"systems", "async", "runtime", "network", "low-level"}
    mono_patterns = {"monorepo", "workspace", "multi-package"}

    if pattern_set & mono_patterns:
        return "monorepo"
    if pattern_set & web_patterns:
        return "web"
    if pattern_set & cli_patterns:
        return "cli"
    if pattern_set & data_patterns:
        return "data"
    if pattern_set & systems_patterns:
        return "systems"
    if pattern_set & lib_patterns:
        return "library"

    # Fallback: classify by language
    lang = entry.get("language", "unknown")
    return f"library_{lang}"


def main():
    parser = argparse.ArgumentParser(description="Generate stratified train/val/test splits.")
    parser.add_argument("--train", type=float, default=0.8, help="Train fraction (default: 0.8)")
    parser.add_argument("--val", type=float, default=0.1, help="Validation fraction (default: 0.1)")
    parser.add_argument("--test", type=float, default=0.1, help="Test fraction (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--freeze-test",
        type=str,
        default=None,
        help="Path to existing test.txt — freeze these repos in test set",
    )
    args = parser.parse_args()

    # Validate ratios
    for name, val in [("train", args.train), ("val", args.val), ("test", args.test)]:
        if val < 0:
            print(f"ERROR: --{name} must be non-negative, got {val}", file=sys.stderr)
            sys.exit(1)
    total = args.train + args.val + args.test
    if abs(total - 1.0) > 1e-6:
        print(f"ERROR: ratios must sum to 1.0, got {total}", file=sys.stderr)
        sys.exit(1)

    registry = load_registry()

    # Find repos with preprocessed features
    available = sorted(
        d.name
        for d in EXAMPLES_DIR.iterdir()
        if d.is_dir()
        and (d / "features.npz").exists()
        and (d / "features.meta.json").exists()
    )

    if len(available) < 3:
        print(f"ERROR: Need at least 3 preprocessed repos, found {len(available)}", file=sys.stderr)
        sys.exit(1)

    print(f"Available repos: {len(available)}")

    # Load frozen test set if specified
    frozen_test: set[str] = set()
    if args.freeze_test and Path(args.freeze_test).exists():
        frozen_test = set(Path(args.freeze_test).read_text().strip().split("\n"))
        frozen_test = {r for r in frozen_test if r in set(available)}
        print(f"Frozen test set: {len(frozen_test)} repos")

    # Classify repos for stratification
    repo_categories: list[str] = []
    repo_names: list[str] = []
    for name in available:
        if name in frozen_test:
            continue  # will be added to test set directly
        entry = registry.get(name, {"language": "unknown"})
        cat = classify_repo(entry)
        repo_categories.append(cat)
        repo_names.append(name)

    # Show category distribution
    cat_counts: dict[str, int] = {}
    for cat in repo_categories:
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    print(f"Categories: {cat_counts}")

    n = len(repo_names)
    if n < 3:
        print(f"ERROR: Need at least 3 non-frozen repos, found {n}", file=sys.stderr)
        sys.exit(1)

    # If categories are too sparse for stratification, fall back to random
    min_cat_count = min(cat_counts.values()) if cat_counts else 0
    use_stratified = min_cat_count >= 2 and len(cat_counts) >= 2

    # Compute split sizes
    n_test = max(1, int(n * args.test))
    n_val = max(1, int(n * args.val))
    n_train = n - n_test - n_val

    if n_train < 1:
        print(f"ERROR: Not enough repos for split (n={n})", file=sys.stderr)
        sys.exit(1)

    if use_stratified:
        # Two-stage stratified split: first split off test, then val from remainder
        test_ratio = n_test / n
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=args.seed)
        indices = list(range(n))

        remain_idx, test_idx = next(splitter.split(indices, repo_categories))

        # Second split: val from remainder
        remain_cats = [repo_categories[i] for i in remain_idx]
        val_ratio = n_val / len(remain_idx)
        val_ratio = min(val_ratio, 0.5)  # safety cap

        # Check if stratification is possible for second split
        remain_cat_counts: dict[str, int] = {}
        for cat in remain_cats:
            remain_cat_counts[cat] = remain_cat_counts.get(cat, 0) + 1
        can_stratify_val = min(remain_cat_counts.values()) >= 2 if remain_cat_counts else False

        if can_stratify_val:
            splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=args.seed + 1)
            train_sub_idx, val_sub_idx = next(splitter2.split(list(range(len(remain_idx))), remain_cats))
            train_idx = [remain_idx[i] for i in train_sub_idx]
            val_idx = [remain_idx[i] for i in val_sub_idx]
        else:
            # Random split for val
            import random
            rng = random.Random(args.seed + 1)
            remain_list = list(remain_idx)
            rng.shuffle(remain_list)
            val_idx = remain_list[:n_val]
            train_idx = remain_list[n_val:]
    else:
        # Simple random split
        import random
        rng = random.Random(args.seed)
        indices = list(range(n))
        rng.shuffle(indices)
        test_idx = indices[:n_test]
        val_idx = indices[n_test:n_test + n_val]
        train_idx = indices[n_test + n_val:]

    train_repos = sorted([repo_names[i] for i in train_idx])
    val_repos = sorted([repo_names[i] for i in val_idx])
    test_repos = sorted([repo_names[i] for i in test_idx])

    # Add frozen test repos
    test_repos = sorted(set(test_repos) | frozen_test)

    # Write split files
    splits_dir = EXAMPLES_DIR / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    (splits_dir / "train.txt").write_text("\n".join(train_repos) + "\n")
    (splits_dir / "val.txt").write_text("\n".join(val_repos) + "\n")
    (splits_dir / "test.txt").write_text("\n".join(test_repos) + "\n")

    print(f"\nSplits written to {splits_dir}/")
    print(f"  train: {len(train_repos)} repos")
    print(f"  val:   {len(val_repos)} repos")
    print(f"  test:  {len(test_repos)} repos")

    # Write split metadata
    split_meta = {
        "seed": args.seed,
        "ratios": {"train": args.train, "val": args.val, "test": args.test},
        "counts": {"train": len(train_repos), "val": len(val_repos), "test": len(test_repos)},
        "frozen_test": sorted(frozen_test),
        "stratified": use_stratified,
        "categories": cat_counts,
    }
    with open(splits_dir / "split_meta.json", "w") as f:
        json.dump(split_meta, f, indent=2)


if __name__ == "__main__":
    main()
