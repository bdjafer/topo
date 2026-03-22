"""Hermetic fixture benchmarks that stand in for real architectural patterns."""

from __future__ import annotations

from tests.test_validation.benchmark_utils import (
    analyze_fixture,
    compute_nmi_from_mappings,
    labels_by_top_package,
    labels_from_modules,
)


def test_layered_app_aligns_with_package_structure():
    """A layered fixture should cluster close to the obvious package baseline."""
    result = analyze_fixture("layered_app", n_modules=3)
    predicted = labels_from_modules(result)
    baseline = labels_by_top_package(list(predicted))
    nmi = compute_nmi_from_mappings(predicted, baseline)

    spectral = result.raw.get("spectral", {})
    assert spectral.get("fiedler_value") is not None
    assert nmi > 0.6
    assert not any(
        f["kind"] == "layer_violation"
        for f in result.issues
    )


def test_reverse_flow_fixture_has_bidirectional_dependency():
    """A fixture with reverse package flow should have bidirectional module dependencies."""
    result = analyze_fixture("reverse_flow_app")

    mod_labels = {m.id: m.label for m in result.modules}
    dependency_pairs = {
        (mod_labels.get(dep["source"], ""), mod_labels.get(dep["target"], ""))
        for dep in result.cross_package_dependencies
    }
    # Rust backend may produce finer-grained labels (e.g. "data.store" vs "data").
    # Check bidirectional flow between api-prefixed and data-prefixed modules.
    api_to_data = any(s.startswith("api") and t.startswith("data") for s, t in dependency_pairs)
    data_to_api = any(s.startswith("data") and t.startswith("api") for s, t in dependency_pairs)
    assert api_to_data, f"Expected api→data dependency in {dependency_pairs}"
    assert data_to_api, f"Expected data→api dependency in {dependency_pairs}"
    # Note: layer_violation only fires when minority_ratio < 0.4 (clear asymmetry).
    # Balanced bidirectional flow is intentionally not flagged.
