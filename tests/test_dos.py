import numpy as np

from tetrabz import dos
from tetrabz import fermieng
from tetrabz import intdos
from tetrabz import occ
from tests.legacy_cases import brillouin_zone_volume
from tests.legacy_cases import cubic_tight_binding_band
from tests.legacy_cases import exact_free_electron_dos_weighted_integrals
from tests.legacy_cases import exact_free_electron_intdos_weighted_integrals
from tests.legacy_cases import legacy_8x8_dos_weighted_integrals
from tests.legacy_cases import legacy_8x8_intdos_weighted_integrals
from tests.legacy_cases import legacy_dos_energy_points
from tests.legacy_cases import legacy_free_electron_case
from tests.legacy_cases import load_legacy_example_dataset
from tests.legacy_cases import tight_binding_dos_energy_points


def test_intdos_matches_occ_at_same_energy_cutoff() -> None:
    bvec, eigenvalues, _ = legacy_free_electron_case((4, 4, 4), (4, 4, 4))

    occupation = occ(bvec, eigenvalues, weight_grid_shape=(4, 4, 4), method="optimized", fermi_energy=0.5)
    integrated = intdos(bvec, eigenvalues, np.array([0.5]), weight_grid_shape=(4, 4, 4), method="optimized")[0]

    np.testing.assert_allclose(integrated, occupation, rtol=1.0e-12, atol=1.0e-12)


def test_dos_matches_legacy_8x8_reference_integrals() -> None:
    bvec, eigenvalues, weight_metric = legacy_free_electron_case((8, 8, 8), (8, 8, 8))
    sample_energies = legacy_dos_energy_points()

    weights = dos(bvec, eigenvalues, sample_energies, weight_grid_shape=(8, 8, 8), method="optimized")
    weighted = _weighted_integrals(weights, weight_metric, bvec)

    np.testing.assert_allclose(weighted, legacy_8x8_dos_weighted_integrals(), rtol=6.0e-4, atol=1.0e-5)


def test_intdos_matches_legacy_8x8_reference_integrals() -> None:
    bvec, eigenvalues, weight_metric = legacy_free_electron_case((8, 8, 8), (8, 8, 8))
    sample_energies = legacy_dos_energy_points()

    weights = intdos(bvec, eigenvalues, sample_energies, weight_grid_shape=(8, 8, 8), method="optimized")
    weighted = _weighted_integrals(weights, weight_metric, bvec)

    np.testing.assert_allclose(weighted, legacy_8x8_intdos_weighted_integrals(), rtol=7.0e-4, atol=1.0e-6)


def test_dos_tracks_exact_free_electron_integrals_on_24_grid() -> None:
    bvec, eigenvalues, weight_metric = legacy_free_electron_case((24, 24, 24), (24, 24, 24))
    sample_energies = legacy_dos_energy_points()

    weights = dos(bvec, eigenvalues, sample_energies, weight_grid_shape=(24, 24, 24), method="optimized")
    weighted = _weighted_integrals(weights, weight_metric, bvec)

    np.testing.assert_allclose(
        weighted,
        exact_free_electron_dos_weighted_integrals(sample_energies),
        rtol=4.0e-2,
        atol=1.0e-6,
    )


def test_intdos_tracks_exact_free_electron_integrals_on_24_grid() -> None:
    bvec, eigenvalues, weight_metric = legacy_free_electron_case((24, 24, 24), (24, 24, 24))
    sample_energies = legacy_dos_energy_points()

    weights = intdos(bvec, eigenvalues, sample_energies, weight_grid_shape=(24, 24, 24), method="optimized")
    weighted = _weighted_integrals(weights, weight_metric, bvec)

    np.testing.assert_allclose(
        weighted,
        exact_free_electron_intdos_weighted_integrals(sample_energies),
        rtol=2.0e-2,
        atol=1.0e-6,
    )


def test_linear_tight_binding_dos_matches_legacy_example_curve() -> None:
    reciprocal_vectors, eigenvalues = cubic_tight_binding_band((8, 8, 8))
    sample_energies = tight_binding_dos_energy_points()

    weights = dos(
        reciprocal_vectors,
        eigenvalues,
        sample_energies,
        weight_grid_shape=(8, 8, 8),
        method="linear",
    )
    current_curve = weights.sum(axis=(1, 2, 3, 4))
    legacy_curve = load_legacy_example_dataset("dos1_8.dat")[:, 1]

    np.testing.assert_allclose(current_curve, legacy_curve, rtol=3.0e-4, atol=1.0e-7)


def test_optimized_tight_binding_dos_matches_legacy_example_curve() -> None:
    reciprocal_vectors, eigenvalues = cubic_tight_binding_band((8, 8, 8))
    sample_energies = tight_binding_dos_energy_points()

    weights = dos(
        reciprocal_vectors,
        eigenvalues,
        sample_energies,
        weight_grid_shape=(8, 8, 8),
        method="optimized",
    )
    current_curve = weights.sum(axis=(1, 2, 3, 4))
    legacy_curve = load_legacy_example_dataset("dos2_8.dat")[:, 1]

    np.testing.assert_allclose(current_curve, legacy_curve, rtol=5.0e-4, atol=1.0e-7)


def test_optimized_tight_binding_intdos_tracks_converged_reference_shape() -> None:
    reciprocal_vectors, eigenvalues = cubic_tight_binding_band((8, 8, 8))
    sample_energies = tight_binding_dos_energy_points()

    weights = intdos(
        reciprocal_vectors,
        eigenvalues,
        sample_energies,
        weight_grid_shape=(8, 8, 8),
        method="optimized",
    )
    integrated_curve = weights.sum(axis=(1, 2, 3, 4))

    reference_curve = load_legacy_example_dataset("dos40.dat")
    reference_intdos = _cumulative_trapezoid(reference_curve[:, 0], reference_curve[:, 1])
    reference_intdos /= reference_intdos[-1]

    np.testing.assert_allclose(integrated_curve, reference_intdos, rtol=0.0, atol=2.5e-3)


def test_optimized_tight_binding_half_filling_fermi_energy_stays_at_zero() -> None:
    reciprocal_vectors, eigenvalues = cubic_tight_binding_band((8, 8, 8))

    fermi_energy, _, _ = fermieng(
        reciprocal_vectors,
        eigenvalues,
        electrons_per_spin=0.5,
        weight_grid_shape=(8, 8, 8),
        method="optimized",
    )

    assert abs(fermi_energy) < 1.0e-12


def test_tight_binding_intdos_is_monotone_and_normalized() -> None:
    reciprocal_vectors, eigenvalues = cubic_tight_binding_band((8, 8, 8))
    sample_energies = tight_binding_dos_energy_points()

    weights = intdos(
        reciprocal_vectors,
        eigenvalues,
        sample_energies,
        weight_grid_shape=(8, 8, 8),
        method="optimized",
    )
    integrated_curve = weights.sum(axis=(1, 2, 3, 4))

    assert np.all(np.diff(integrated_curve) >= -1.0e-10)
    assert integrated_curve[0] >= -1.0e-10
    assert abs(integrated_curve[-1] - 1.0) < 3.0e-4


def _weighted_integrals(weights: np.ndarray, metric: np.ndarray, reciprocal_vectors: np.ndarray) -> np.ndarray:
    return (weights * metric[None, ..., None]).sum(axis=(1, 2, 3)) * brillouin_zone_volume(reciprocal_vectors)


def _cumulative_trapezoid(x_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
    increments = 0.5 * (y_values[1:] + y_values[:-1]) * np.diff(x_values)
    return np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(increments)))
