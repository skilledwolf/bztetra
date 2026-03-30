from __future__ import annotations

import numpy as np
import pytest

from bztetra import complex_frequency_polarization_weights
from bztetra import density_of_states_weights
from bztetra import fermi_golden_rule_weights
from bztetra import integrated_density_of_states_weights
from bztetra import nesting_function_weights
from bztetra import occupation_weights
from bztetra import phase_space_overlap_weights
from bztetra import solve_fermi_energy
from bztetra import static_polarization_weights
from tests.legacy_cases import complex_frequency_polarization_energy_points
from tests.legacy_cases import fermi_golden_rule_energy_points
from tests.legacy_cases import legacy_dos_energy_points
from tests.legacy_cases import legacy_free_electron_case
from tests.legacy_cases import legacy_free_electron_response_case
from tests.legacy_parity_helpers import legacy_electron_count_per_spin
from tests.legacy_parity_helpers import normalize_port_dos_output
from tests.legacy_parity_helpers import normalize_port_frequency_output
from tests.legacy_parity_helpers import swap_pair_axes

try:
    import libtetrabz
except ImportError:
    libtetrabz = None


DIRECT_PARITY_CASES = [
    (8, 8, 8),
    (16, 16, 16),
]

pytestmark = pytest.mark.skipif(libtetrabz is None, reason="libtetrabz is not installed")

DIRECT_ARRAY_ATOL = {
    "occupation_weights": 1.0e-12,
    "density_of_states_weights": 1.0e-12,
    "integrated_density_of_states_weights": 1.0e-12,
    "phase_space_overlap_weights": 1.0e-12,
    "nesting_function_weights": 1.0e-12,
    "static_polarization_weights": 5.0e-12,
    "fermi_golden_rule_weights": 1.0e-12,
    "complex_frequency_polarization_weights": 1.0e-7,
}


@pytest.mark.parametrize("grid_shape", DIRECT_PARITY_CASES, ids=["8_to_8", "16_to_16"])
def test_outputs_match_installed_libtetrabz_wrapper_pointwise(
    grid_shape: tuple[int, int, int],
) -> None:
    bvec, eigenvalues, _ = legacy_free_electron_case(grid_shape, grid_shape)
    _, occupied_eigenvalues, target_eigenvalues, _ = legacy_free_electron_response_case(
        grid_shape, grid_shape
    )

    comparisons = [
        (
            "occupation_weights",
            np.asarray(libtetrabz.occ(bvec, eigenvalues - 0.5)),
            np.asarray(
                occupation_weights(
                    bvec,
                    eigenvalues - 0.5,
                    weight_grid_shape=grid_shape,
                    method="optimized",
                )
            ),
        ),
        (
            "density_of_states_weights",
            np.asarray(libtetrabz.dos(bvec, eigenvalues, legacy_dos_energy_points())),
            normalize_port_dos_output(
                density_of_states_weights(
                    bvec,
                    eigenvalues,
                    legacy_dos_energy_points(),
                    weight_grid_shape=grid_shape,
                    method="optimized",
                )
            ),
        ),
        (
            "integrated_density_of_states_weights",
            np.asarray(libtetrabz.intdos(bvec, eigenvalues, legacy_dos_energy_points())),
            normalize_port_dos_output(
                integrated_density_of_states_weights(
                    bvec,
                    eigenvalues,
                    legacy_dos_energy_points(),
                    weight_grid_shape=grid_shape,
                    method="optimized",
                )
            ),
        ),
        (
            "phase_space_overlap_weights",
            np.asarray(libtetrabz.dblstep(bvec, occupied_eigenvalues - 0.5, target_eigenvalues - 0.5)),
            swap_pair_axes(
                phase_space_overlap_weights(
                    bvec,
                    occupied_eigenvalues - 0.5,
                    target_eigenvalues - 0.5,
                    weight_grid_shape=grid_shape,
                    method="optimized",
                )
            ),
        ),
        (
            "nesting_function_weights",
            np.asarray(libtetrabz.dbldelta(bvec, occupied_eigenvalues - 0.5, target_eigenvalues - 0.5)),
            swap_pair_axes(
                nesting_function_weights(
                    bvec,
                    occupied_eigenvalues - 0.5,
                    target_eigenvalues - 0.5,
                    weight_grid_shape=grid_shape,
                    method="optimized",
                )
            ),
        ),
        (
            "static_polarization_weights",
            np.asarray(libtetrabz.polstat(bvec, occupied_eigenvalues - 0.5, target_eigenvalues - 0.5)),
            swap_pair_axes(
                static_polarization_weights(
                    bvec,
                    occupied_eigenvalues - 0.5,
                    target_eigenvalues - 0.5,
                    weight_grid_shape=grid_shape,
                    method="optimized",
                )
            ),
        ),
        (
            "fermi_golden_rule_weights",
            np.asarray(
                libtetrabz.fermigr(
                    bvec,
                    occupied_eigenvalues - 0.5,
                    target_eigenvalues - 0.5,
                    fermi_golden_rule_energy_points(),
                )
            ),
            normalize_port_frequency_output(
                fermi_golden_rule_weights(
                    bvec,
                    occupied_eigenvalues - 0.5,
                    target_eigenvalues - 0.5,
                    fermi_golden_rule_energy_points(),
                    weight_grid_shape=grid_shape,
                    method="optimized",
                )
            ),
        ),
        (
            "complex_frequency_polarization_weights",
            np.asarray(
                libtetrabz.polcmplx(
                    bvec,
                    occupied_eigenvalues - 0.5,
                    target_eigenvalues - 0.5,
                    complex_frequency_polarization_energy_points(),
                )
            ),
            normalize_port_frequency_output(
                complex_frequency_polarization_weights(
                    bvec,
                    occupied_eigenvalues - 0.5,
                    target_eigenvalues - 0.5,
                    complex_frequency_polarization_energy_points(),
                    weight_grid_shape=grid_shape,
                    method="optimized",
                )
            ),
        ),
    ]

    for name, legacy_output, port_output in comparisons:
        np.testing.assert_allclose(
            port_output,
            legacy_output,
            rtol=0.0,
            atol=DIRECT_ARRAY_ATOL[name],
            err_msg=f"{name} wrapper parity failed on grid {grid_shape!r}",
        )


@pytest.mark.parametrize("grid_shape", DIRECT_PARITY_CASES, ids=["8_to_8", "16_to_16"])
def test_solve_fermi_energy_matches_installed_libtetrabz_wrapper(
    grid_shape: tuple[int, int, int],
) -> None:
    bvec, eigenvalues, _ = legacy_free_electron_case(grid_shape, grid_shape)
    electrons_per_spin = legacy_electron_count_per_spin(bvec)

    legacy_fermi_energy, legacy_weights, _ = libtetrabz.fermieng(
        bvec, eigenvalues, electrons_per_spin
    )
    port_solution = solve_fermi_energy(
        bvec,
        eigenvalues,
        electrons_per_spin,
        weight_grid_shape=grid_shape,
        method="optimized",
    )

    np.testing.assert_allclose(
        float(port_solution.fermi_energy),
        float(legacy_fermi_energy),
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(port_solution.weights),
        np.asarray(legacy_weights),
        rtol=0.0,
        atol=1.0e-12,
    )
