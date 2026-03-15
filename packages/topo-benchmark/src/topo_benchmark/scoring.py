"""Scoring primitives for benchmark evaluation."""

from __future__ import annotations

from collections import defaultdict
from math import comb, log, prod


def geometric_mean(*values: float) -> float:
    """Geometric mean of values. Returns 0 if any value is 0."""
    if not values or any(v <= 0 for v in values):
        return 0.0
    return prod(values) ** (1.0 / len(values))


def compute_ari(
    left: dict[str, object],
    right: dict[str, object],
) -> float:
    """Adjusted Rand Index between two label mappings on shared keys."""
    common = sorted(set(left) & set(right))
    if len(common) < 2:
        return 1.0 if common else 0.0

    labels_a = [left[k] for k in common]
    labels_b = [right[k] for k in common]
    n = len(common)

    joint: dict[tuple[object, object], int] = defaultdict(int)
    count_a: dict[object, int] = defaultdict(int)
    count_b: dict[object, int] = defaultdict(int)
    for a, b in zip(labels_a, labels_b):
        joint[(a, b)] += 1
        count_a[a] += 1
        count_b[b] += 1

    total_pairs = comb(n, 2)
    sum_joint = sum(comb(c, 2) for c in joint.values() if c >= 2)
    sum_a = sum(comb(c, 2) for c in count_a.values() if c >= 2)
    sum_b = sum(comb(c, 2) for c in count_b.values() if c >= 2)

    expected = (sum_a * sum_b) / total_pairs if total_pairs else 0.0
    max_index = 0.5 * (sum_a + sum_b)
    denom = max_index - expected
    if denom == 0:
        return 1.0
    return (sum_joint - expected) / denom


def compute_nmi(
    left: dict[str, object],
    right: dict[str, object],
) -> float:
    """Normalized Mutual Information between two label mappings on shared keys."""
    common = sorted(set(left) & set(right))
    if not common:
        return 0.0
    n = len(common)

    labels_a = [left[k] for k in common]
    labels_b = [right[k] for k in common]

    joint: dict[tuple[object, object], int] = defaultdict(int)
    count_a: dict[object, int] = defaultdict(int)
    count_b: dict[object, int] = defaultdict(int)
    for a, b in zip(labels_a, labels_b):
        joint[(a, b)] += 1
        count_a[a] += 1
        count_b[b] += 1

    mi = 0.0
    for (a, b), nij in joint.items():
        if nij == 0:
            continue
        pij = nij / n
        pi = count_a[a] / n
        pj = count_b[b] / n
        mi += pij * log(pij / (pi * pj))

    ha = -sum((c / n) * log(c / n) for c in count_a.values() if c > 0)
    hb = -sum((c / n) * log(c / n) for c in count_b.values() if c > 0)

    if ha + hb == 0:
        return 1.0
    return 2 * mi / (ha + hb)


def compute_boundary_f1(
    edges: list[tuple[str, str]],
    predicted: dict[str, object],
    gold: dict[str, object],
) -> float:
    """F1 on the cross-module edge class.

    For each edge between nodes in both partitions, classify as intra/cross-module
    in both predicted and gold. Compute F1 on the cross-module class.
    """
    tp = fp = fn = 0
    for src, tgt in edges:
        if src not in predicted or tgt not in predicted:
            continue
        if src not in gold or tgt not in gold:
            continue
        pred_cross = predicted[src] != predicted[tgt]
        gold_cross = gold[src] != gold[tgt]
        if gold_cross and pred_cross:
            tp += 1
        elif pred_cross and not gold_cross:
            fp += 1
        elif gold_cross and not pred_cross:
            fn += 1
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def compute_coverage(
    predicted: dict[str, object],
    gold_nodes: set[str],
    unassigned_label: object = None,
) -> float:
    """Fraction of gold nodes that received a non-unassigned predicted module."""
    if not gold_nodes:
        return 0.0
    assigned = sum(
        1 for n in gold_nodes
        if n in predicted and predicted[n] != unassigned_label
    )
    return assigned / len(gold_nodes)


def compute_average_precision(
    scores: list[float],
    labels: list[bool],
) -> float:
    """Average Precision from ranked predictions."""
    if not scores or not any(labels):
        return 0.0
    paired = sorted(zip(scores, labels), key=lambda x: -x[0])
    tp = 0
    ap_sum = 0.0
    for i, (_, is_pos) in enumerate(paired):
        if is_pos:
            tp += 1
            ap_sum += tp / (i + 1)
    total_pos = sum(labels)
    return ap_sum / total_pos if total_pos else 0.0


def compute_precision_at_k(
    scores: list[float],
    labels: list[bool],
    k: int,
) -> float:
    """Precision among the top-k ranked predictions."""
    if not scores or k <= 0:
        return 0.0
    paired = sorted(zip(scores, labels), key=lambda x: -x[0])
    top_k = paired[:k]
    return sum(1 for _, is_pos in top_k if is_pos) / len(top_k)


def compute_ece(
    confidences: list[float],
    correctness: list[bool],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error over confidence bins."""
    if not confidences:
        return 0.0
    n = len(confidences)
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for conf, correct in zip(confidences, correctness):
        idx = min(int(conf * n_bins), n_bins - 1)
        bins[idx].append((conf, correct))

    ece = 0.0
    for bin_items in bins:
        if not bin_items:
            continue
        avg_conf = sum(c for c, _ in bin_items) / len(bin_items)
        avg_acc = sum(1 for _, correct in bin_items if correct) / len(bin_items)
        ece += (len(bin_items) / n) * abs(avg_conf - avg_acc)
    return ece


def compute_brier(
    confidences: list[float],
    correctness: list[bool],
) -> float:
    """Brier score: mean squared error between confidence and correctness."""
    if not confidences:
        return 0.0
    return sum(
        (conf - float(correct)) ** 2
        for conf, correct in zip(confidences, correctness)
    ) / len(confidences)


def compute_v_measure(
    left: dict[str, object],
    right: dict[str, object],
) -> tuple[float, float, float]:
    """V-measure between two label mappings on shared keys.

    Returns (homogeneity, completeness, v_measure).
    Homogeneity: each predicted cluster contains only members of a single gold class.
    Completeness: all members of a gold class are assigned to the same cluster.
    V-measure: harmonic mean of homogeneity and completeness.

    Unlike NMI, V-measure does not degenerate when one partition has very few groups.
    """
    common = sorted(set(left) & set(right))
    if not common:
        return (0.0, 0.0, 0.0)
    n = len(common)

    labels_c = [left[k] for k in common]   # gold classes
    labels_k = [right[k] for k in common]   # predicted clusters

    joint: dict[tuple[object, object], int] = defaultdict(int)
    count_c: dict[object, int] = defaultdict(int)
    count_k: dict[object, int] = defaultdict(int)
    for c, k in zip(labels_c, labels_k):
        joint[(c, k)] += 1
        count_c[c] += 1
        count_k[k] += 1

    # H(C) — entropy of gold classes
    h_c = -sum((cnt / n) * log(cnt / n) for cnt in count_c.values() if cnt > 0)
    # H(K) — entropy of predicted clusters
    h_k = -sum((cnt / n) * log(cnt / n) for cnt in count_k.values() if cnt > 0)

    # H(C|K) — conditional entropy of gold classes given predicted clusters
    h_c_given_k = 0.0
    for k_label, k_count in count_k.items():
        for c_label in count_c:
            n_ck = joint.get((c_label, k_label), 0)
            if n_ck > 0:
                h_c_given_k -= (n_ck / n) * log(n_ck / k_count)

    # H(K|C) — conditional entropy of predicted clusters given gold classes
    h_k_given_c = 0.0
    for c_label, c_count in count_c.items():
        for k_label in count_k:
            n_ck = joint.get((c_label, k_label), 0)
            if n_ck > 0:
                h_k_given_c -= (n_ck / n) * log(n_ck / c_count)

    # Homogeneity: 1 - H(C|K) / H(C)
    homogeneity = 1.0 - (h_c_given_k / h_c) if h_c > 0 else 1.0
    # Completeness: 1 - H(K|C) / H(K)
    completeness = 1.0 - (h_k_given_c / h_k) if h_k > 0 else 1.0

    if homogeneity + completeness == 0:
        v_measure = 0.0
    else:
        v_measure = 2 * homogeneity * completeness / (homogeneity + completeness)

    return (homogeneity, completeness, v_measure)


def compute_cross_directory_recovery(
    predicted: dict[str, object],
    gold: dict[str, object],
    directory_labels: dict[str, object],
) -> float:
    """Fraction of cross-directory same-module pairs correctly co-clustered.

    Finds pairs of nodes that share the same gold module but are in different
    directories. Returns the fraction of such pairs that are also co-clustered
    by the predicted partition.

    This is the decisive metric for marginal value: directory grouping scores 0%
    by definition, since cross-directory pairs are never co-clustered by directory.
    """
    common = sorted(set(predicted) & set(gold) & set(directory_labels))
    if len(common) < 2:
        return 0.0

    cross_dir_same_module = 0
    cross_dir_same_module_recovered = 0

    for i, a in enumerate(common):
        for b in common[i + 1:]:
            if gold[a] != gold[b]:
                continue
            if directory_labels[a] == directory_labels[b]:
                continue
            # This pair shares a gold module but is in different directories
            cross_dir_same_module += 1
            if predicted[a] == predicted[b]:
                cross_dir_same_module_recovered += 1

    if cross_dir_same_module == 0:
        return float("nan")  # No cross-directory pairs exist — metric not applicable
    return cross_dir_same_module_recovered / cross_dir_same_module
