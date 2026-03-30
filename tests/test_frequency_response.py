import numpy as np

from tetrabz import fermi_golden_rule_weights
from tests.legacy_cases import brillouin_zone_volume
from tests.legacy_cases import exact_fermi_golden_rule_weighted_integrals
from tests.legacy_cases import fermi_golden_rule_energy_points
from tests.legacy_cases import legacy_16x8_fermi_golden_rule_weighted_integrals
from tests.legacy_cases import legacy_8x8_fermi_golden_rule_weighted_integrals
from tests.legacy_cases import legacy_free_electron_response_case


def test_fermi_golden_rule_weights_exposes_energy_first_pair_band_last_layout() -> None:
    bvec, eigenvalues_1, eigenvalues_2, _ = legacy_free_electron_response_case((8, 8, 8), (8, 8, 8))

    weights = fermi_golden_rule_weights(
        bvec,
        eigenvalues_1 - 0.5,
        eigenvalues_2 - 0.5,
        fermi_golden_rule_energy_points(),
        weight_grid_shape=(8, 8, 8),
        method="optimized",
    )

    assert weights.shape == (3, 8, 8, 8, 2, 2)
    assert weights.dtype == np.float64


def test_fermi_golden_rule_weights_matches_legacy_8x8_response_reference() -> None:
    bvec, eigenvalues_1, eigenvalues_2, weight_metric = legacy_free_electron_response_case((8, 8, 8), (8, 8, 8))

    weights = fermi_golden_rule_weights(
        bvec,
        eigenvalues_1 - 0.5,
        eigenvalues_2 - 0.5,
        fermi_golden_rule_energy_points(),
        weight_grid_shape=(8, 8, 8),
        method="optimized",
    )
    weighted = _weighted_energy_matrix(weights, weight_metric, bvec)

    np.testing.assert_allclose(weighted, legacy_8x8_fermi_golden_rule_weighted_integrals(), rtol=8.0e-4, atol=1.0e-6)


def test_fermi_golden_rule_weights_matches_legacy_16x8_interpolation_reference() -> None:
    bvec, eigenvalues_1, eigenvalues_2, weight_metric = legacy_free_electron_response_case((16, 16, 16), (8, 8, 8))

    weights = fermi_golden_rule_weights(
        bvec,
        eigenvalues_1 - 0.5,
        eigenvalues_2 - 0.5,
        fermi_golden_rule_energy_points(),
        weight_grid_shape=(8, 8, 8),
        method="optimized",
    )
    weighted = _weighted_energy_matrix(weights, weight_metric, bvec)

    np.testing.assert_allclose(weighted, legacy_16x8_fermi_golden_rule_weighted_integrals(), rtol=2.0e-4, atol=1.0e-5)


def test_fermi_golden_rule_weights_tracks_exact_integrals_on_16_grid() -> None:
    bvec, eigenvalues_1, eigenvalues_2, weight_metric = legacy_free_electron_response_case((16, 16, 16), (16, 16, 16))

    weights = fermi_golden_rule_weights(
        bvec,
        eigenvalues_1 - 0.5,
        eigenvalues_2 - 0.5,
        fermi_golden_rule_energy_points(),
        weight_grid_shape=(16, 16, 16),
        method="optimized",
    )
    weighted = _weighted_energy_matrix(weights, weight_metric, bvec)

    np.testing.assert_allclose(weighted, exact_fermi_golden_rule_weighted_integrals(), rtol=5.0e-3, atol=1.0e-5)


def _weighted_energy_matrix(weights: np.ndarray, metric: np.ndarray, reciprocal_vectors: np.ndarray) -> np.ndarray:
    return (weights * metric[None, ..., None, None]).sum(axis=(1, 2, 3)) * brillouin_zone_volume(reciprocal_vectors)
