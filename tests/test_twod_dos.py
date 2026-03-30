import numpy as np

from bztetra.twod import density_of_states_weights
from bztetra.twod import integrated_density_of_states_weights
from bztetra.twod import occupation_weights
from tests.twod_cases import exact_free_electron_dos_normalized
from tests.twod_cases import exact_free_electron_intdos_normalized
from tests.twod_cases import free_electron_case
from tests.twod_cases import free_electron_energy_points


def test_twod_integrated_density_of_states_matches_occupation_at_same_cutoff() -> None:
    reciprocal_vectors, eigenvalues = free_electron_case((16, 16))

    occupation = occupation_weights(reciprocal_vectors, eigenvalues, fermi_energy=1.5)
    integrated = integrated_density_of_states_weights(
        reciprocal_vectors,
        eigenvalues,
        np.array([1.5], dtype=np.float64),
    )[0]

    np.testing.assert_allclose(integrated, occupation, rtol=1.0e-12, atol=1.0e-12)


def test_twod_density_of_states_tracks_exact_free_electron_curve() -> None:
    reciprocal_vectors, eigenvalues = free_electron_case((96, 96))
    sample_energies = free_electron_energy_points()

    weights = density_of_states_weights(reciprocal_vectors, eigenvalues, sample_energies)
    total_curve = weights.sum(axis=(1, 2, 3))

    np.testing.assert_allclose(
        total_curve,
        exact_free_electron_dos_normalized(sample_energies),
        rtol=5.0e-2,
        atol=2.0e-3,
    )


def test_twod_integrated_density_of_states_tracks_exact_free_electron_curve() -> None:
    reciprocal_vectors, eigenvalues = free_electron_case((96, 96))
    sample_energies = free_electron_energy_points()

    weights = integrated_density_of_states_weights(reciprocal_vectors, eigenvalues, sample_energies)
    total_curve = weights.sum(axis=(1, 2, 3))

    np.testing.assert_allclose(
        total_curve,
        exact_free_electron_intdos_normalized(sample_energies),
        rtol=4.0e-2,
        atol=2.0e-3,
    )
