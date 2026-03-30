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
from tests.legacy_parity_helpers import assert_with_entrywise_atol
from tests.legacy_parity_helpers import legacy_electron_count_per_spin
from tests.legacy_parity_helpers import weighted_energy_matrix
from tests.legacy_parity_helpers import weighted_integrals
from tests.legacy_parity_helpers import weighted_matrix
from tests.legacy_shell_cases import load_legacy_shell_case


SHELL_CASES = [
    ((8, 8, 8), (8, 8, 8)),
    ((16, 16, 16), (8, 8, 8)),
    ((16, 16, 16), (16, 16, 16)),
]


@pytest.mark.parametrize(
    ("energy_grid_shape", "weight_grid_shape"),
    SHELL_CASES,
    ids=["8_to_8", "16_to_8", "16_to_16"],
)
def test_weighted_outputs_match_legacy_shell_reference_matrix(
    energy_grid_shape: tuple[int, int, int],
    weight_grid_shape: tuple[int, int, int],
) -> None:
    references = load_legacy_shell_case(energy_grid_shape, weight_grid_shape)

    bvec, eigenvalues, weight_metric = legacy_free_electron_case(
        energy_grid_shape, weight_grid_shape
    )
    _, occupied_eigenvalues, target_eigenvalues, response_metric = legacy_free_electron_response_case(
        energy_grid_shape, weight_grid_shape
    )

    occupation = occupation_weights(
        bvec,
        eigenvalues - 0.5,
        weight_grid_shape=weight_grid_shape,
        method="optimized",
    )
    assert_with_entrywise_atol(
        weighted_integrals(occupation, weight_metric, bvec),
        references["occupation_weights"].result,
        references["occupation_weights"].atol,
        label="occupation_weights",
    )

    fermi_solution = solve_fermi_energy(
        bvec,
        eigenvalues,
        legacy_electron_count_per_spin(bvec),
        weight_grid_shape=weight_grid_shape,
        method="optimized",
    )
    assert_with_entrywise_atol(
        np.asarray(fermi_solution.fermi_energy),
        references["solve_fermi_energy_fermi_energy"].result,
        references["solve_fermi_energy_fermi_energy"].atol,
        label="solve_fermi_energy.fermi_energy",
    )
    assert_with_entrywise_atol(
        weighted_integrals(fermi_solution.weights, weight_metric, bvec),
        references["solve_fermi_energy_weights"].result,
        references["solve_fermi_energy_weights"].atol,
        label="solve_fermi_energy.weights",
    )

    dos = density_of_states_weights(
        bvec,
        eigenvalues,
        legacy_dos_energy_points(),
        weight_grid_shape=weight_grid_shape,
        method="optimized",
    )
    assert_with_entrywise_atol(
        weighted_integrals(dos, weight_metric, bvec),
        references["density_of_states_weights"].result,
        references["density_of_states_weights"].atol,
        label="density_of_states_weights",
    )

    integrated_dos = integrated_density_of_states_weights(
        bvec,
        eigenvalues,
        legacy_dos_energy_points(),
        weight_grid_shape=weight_grid_shape,
        method="optimized",
    )
    assert_with_entrywise_atol(
        weighted_integrals(integrated_dos, weight_metric, bvec),
        references["integrated_density_of_states_weights"].result,
        references["integrated_density_of_states_weights"].atol,
        label="integrated_density_of_states_weights",
    )

    overlap = phase_space_overlap_weights(
        bvec,
        occupied_eigenvalues - 0.5,
        target_eigenvalues - 0.5,
        weight_grid_shape=weight_grid_shape,
        method="optimized",
    )
    assert_with_entrywise_atol(
        weighted_matrix(overlap, response_metric, bvec),
        references["phase_space_overlap_weights"].result,
        references["phase_space_overlap_weights"].atol,
        label="phase_space_overlap_weights",
    )

    nesting = nesting_function_weights(
        bvec,
        occupied_eigenvalues - 0.5,
        target_eigenvalues - 0.5,
        weight_grid_shape=weight_grid_shape,
        method="optimized",
    )
    assert_with_entrywise_atol(
        weighted_matrix(nesting, response_metric, bvec),
        references["nesting_function_weights"].result,
        references["nesting_function_weights"].atol,
        label="nesting_function_weights",
    )

    polstat = static_polarization_weights(
        bvec,
        occupied_eigenvalues - 0.5,
        target_eigenvalues - 0.5,
        weight_grid_shape=weight_grid_shape,
        method="optimized",
    )
    assert_with_entrywise_atol(
        weighted_matrix(polstat, response_metric, bvec),
        references["static_polarization_weights"].result,
        references["static_polarization_weights"].atol,
        label="static_polarization_weights",
    )

    fermigr = fermi_golden_rule_weights(
        bvec,
        occupied_eigenvalues - 0.5,
        target_eigenvalues - 0.5,
        fermi_golden_rule_energy_points(),
        weight_grid_shape=weight_grid_shape,
        method="optimized",
    )
    assert_with_entrywise_atol(
        weighted_energy_matrix(fermigr, response_metric, bvec),
        references["fermi_golden_rule_weights"].result,
        references["fermi_golden_rule_weights"].atol,
        label="fermi_golden_rule_weights",
    )

    polcmplx = complex_frequency_polarization_weights(
        bvec,
        occupied_eigenvalues - 0.5,
        target_eigenvalues - 0.5,
        complex_frequency_polarization_energy_points(),
        weight_grid_shape=weight_grid_shape,
        method="optimized",
    )
    assert_with_entrywise_atol(
        weighted_energy_matrix(polcmplx, response_metric, bvec),
        references["complex_frequency_polarization_weights"].result,
        references["complex_frequency_polarization_weights"].atol,
        label="complex_frequency_polarization_weights",
    )
