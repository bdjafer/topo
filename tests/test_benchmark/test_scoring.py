"""Tests for scoring primitives."""

from __future__ import annotations

from topo_benchmark.scoring import (
    compute_ari,
    compute_average_precision,
    compute_boundary_f1,
    compute_brier,
    compute_coverage,
    compute_ece,
    compute_nmi,
    compute_precision_at_k,
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
