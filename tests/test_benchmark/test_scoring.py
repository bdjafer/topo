"""Tests for scoring primitives."""

from __future__ import annotations

import math

from topo_benchmark.scoring import (
    compute_ari,
    compute_average_precision,
    compute_boundary_f1,
    compute_brier,
    compute_coverage,
    compute_cross_directory_recovery,
    compute_ece,
    compute_nmi,
    compute_precision_at_k,
    compute_v_measure,
    geometric_mean,
)


def test_geometric_mean_basic():
    assert geometric_mean(4.0, 9.0) == 6.0
    assert geometric_mean(1.0, 1.0, 1.0) == 1.0


def test_geometric_mean_zero():
    assert geometric_mean(0.0, 5.0) == 0.0
    assert geometric_mean(0.0) == 0.0


def test_ari_perfect_match():
    labels = {"a": 0, "b": 0, "c": 1, "d": 1}
    assert compute_ari(labels, labels) == 1.0


def test_ari_different_ids():
    left = {"a": 0, "b": 0, "c": 1, "d": 1}
    right = {"a": "x", "b": "x", "c": "y", "d": "y"}
    assert compute_ari(left, right) == 1.0


def test_nmi_perfect_match():
    labels = {"a": 0, "b": 0, "c": 1, "d": 1}
    assert abs(compute_nmi(labels, labels) - 1.0) < 0.001


def test_nmi_completely_different():
    left = {"a": 0, "b": 1, "c": 0, "d": 1}
    right = {"a": 0, "b": 0, "c": 1, "d": 1}
    # Not zero but less than 1
    assert compute_nmi(left, right) < 1.0


def test_boundary_f1_perfect():
    edges = [("a", "b"), ("c", "d"), ("a", "c")]
    pred = {"a": 0, "b": 0, "c": 1, "d": 1}
    gold = {"a": 0, "b": 0, "c": 1, "d": 1}
    assert compute_boundary_f1(edges, pred, gold) == 1.0


def test_coverage_full():
    pred = {"a": 0, "b": 1, "c": 2}
    assert compute_coverage(pred, {"a", "b", "c"}) == 1.0


def test_coverage_partial():
    pred = {"a": 0, "b": 1}
    assert compute_coverage(pred, {"a", "b", "c"}) == 2 / 3


def test_average_precision():
    # Perfect ranking: all positives first
    scores = [0.9, 0.8, 0.7, 0.6]
    labels = [True, True, False, False]
    assert compute_average_precision(scores, labels) == 1.0


def test_precision_at_k():
    scores = [0.9, 0.8, 0.7, 0.6]
    labels = [True, False, True, False]
    assert compute_precision_at_k(scores, labels, k=2) == 0.5


def test_ece_perfectly_calibrated():
    # Confidence = accuracy
    confs = [0.9, 0.9, 0.1, 0.1]
    correct = [True, True, False, False]
    ece = compute_ece(confs, correct)
    assert ece < 0.2  # Should be close to 0


def test_brier_perfect():
    confs = [1.0, 0.0]
    correct = [True, False]
    assert compute_brier(confs, correct) == 0.0


# --- V-measure tests ---


def test_v_measure_perfect_match():
    labels = {"a": 0, "b": 0, "c": 1, "d": 1}
    h, c, v = compute_v_measure(labels, labels)
    assert abs(v - 1.0) < 0.001
    assert abs(h - 1.0) < 0.001
    assert abs(c - 1.0) < 0.001


def test_v_measure_different_label_names():
    gold = {"a": "X", "b": "X", "c": "Y", "d": "Y"}
    pred = {"a": 0, "b": 0, "c": 1, "d": 1}
    _, _, v = compute_v_measure(gold, pred)
    assert abs(v - 1.0) < 0.001


def test_v_measure_finer_clustering():
    """Finer-grained prediction: each gold class split into 2 clusters.

    Homogeneity should be 1.0 (each cluster is pure).
    Completeness should be < 1.0 (gold classes are split).
    V-measure should be between 0 and 1.
    """
    gold = {"a": "X", "b": "X", "c": "X", "d": "Y", "e": "Y", "f": "Y"}
    pred = {"a": 0, "b": 0, "c": 1, "d": 2, "e": 2, "f": 3}
    h, c, v = compute_v_measure(gold, pred)
    assert abs(h - 1.0) < 0.001  # Each cluster is pure
    assert c < 1.0  # Gold classes are split
    assert 0 < v < 1.0


def test_v_measure_single_gold_class():
    """Flat package: all nodes in one gold class.

    Unlike NMI, V-measure should still give a meaningful result.
    """
    gold = {"a": "all", "b": "all", "c": "all", "d": "all"}
    pred = {"a": 0, "b": 0, "c": 1, "d": 1}
    h, c, v = compute_v_measure(gold, pred)
    # Homogeneity: H(C) = 0, so homogeneity = 1.0 (trivially)
    # Completeness: gold class split across 2 clusters, so < 1.0
    assert h == 1.0


def test_v_measure_empty():
    h, c, v = compute_v_measure({}, {"a": 0})
    assert v == 0.0


# --- Cross-directory recovery tests ---


def test_cross_dir_recovery_perfect():
    """Spectral correctly co-clusters nodes from same module, different dirs."""
    gold = {"a": "mod1", "b": "mod1", "c": "mod2", "d": "mod2"}
    dir_labels = {"a": "dir1", "b": "dir2", "c": "dir1", "d": "dir2"}
    # a and b are in same gold module but different dirs
    # Spectral co-clusters them
    pred = {"a": 0, "b": 0, "c": 1, "d": 1}
    rate = compute_cross_directory_recovery(pred, gold, dir_labels)
    assert rate == 1.0


def test_cross_dir_recovery_zero():
    """Spectral fails to co-cluster cross-directory same-module pairs."""
    gold = {"a": "mod1", "b": "mod1", "c": "mod2", "d": "mod2"}
    dir_labels = {"a": "dir1", "b": "dir2", "c": "dir1", "d": "dir2"}
    # Spectral groups by directory instead
    pred = {"a": 0, "b": 1, "c": 0, "d": 1}
    rate = compute_cross_directory_recovery(pred, gold, dir_labels)
    assert rate == 0.0


def test_cross_dir_recovery_directory_partition_is_zero():
    """Directory grouping always scores 0% on cross-directory recovery."""
    gold = {"a": "mod1", "b": "mod1", "c": "mod2", "d": "mod2"}
    dir_labels = {"a": "dir1", "b": "dir2", "c": "dir1", "d": "dir2"}
    # Directory partition = dir_labels
    rate = compute_cross_directory_recovery(dir_labels, gold, dir_labels)
    assert rate == 0.0


def test_cross_dir_recovery_no_cross_dir_pairs():
    """When all same-module nodes are in the same directory, metric is NaN."""
    gold = {"a": "mod1", "b": "mod1", "c": "mod2", "d": "mod2"}
    dir_labels = {"a": "dir1", "b": "dir1", "c": "dir2", "d": "dir2"}
    pred = {"a": 0, "b": 0, "c": 1, "d": 1}
    rate = compute_cross_directory_recovery(pred, gold, dir_labels)
    assert math.isnan(rate)
