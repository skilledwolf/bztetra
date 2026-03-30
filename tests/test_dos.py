import numpy as np

from tetrabz import dos
from tetrabz import intdos
from tetrabz import occ
from tests.legacy_cases import brillouin_zone_volume
from tests.legacy_cases import exact_free_electron_dos_weighted_integrals
from tests.legacy_cases import exact_free_electron_intdos_weighted_integrals
from tests.legacy_cases import legacy_8x8_dos_weighted_integrals
from tests.legacy_cases import legacy_8x8_intdos_weighted_integrals
from tests.legacy_cases import legacy_dos_energy_points
from tests.legacy_cases import legacy_free_electron_case


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


def _weighted_integrals(weights: np.ndarray, metric: np.ndarray, reciprocal_vectors: np.ndarray) -> np.ndarray:
    return (weights * metric[None, ..., None]).sum(axis=(1, 2, 3)) * brillouin_zone_volume(reciprocal_vectors)
