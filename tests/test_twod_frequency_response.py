from __future__ import annotations

import importlib
import importlib.util

import numpy as np
import pytest

from bztetra.twod import complex_frequency_polarization_observables
from bztetra.twod import complex_frequency_polarization_weights
from bztetra.twod import fermi_golden_rule_observables
from bztetra.twod import fermi_golden_rule_observables_batch
from bztetra.twod import fermi_golden_rule_weights
from bztetra.twod import prepare_response_evaluator
from bztetra.twod import prepare_response_sweep_evaluator
from bztetra.twod import retarded_response_observables
from bztetra.twod import retarded_response_observables_batch
from bztetra.twod import static_polarization_observables
from bztetra.twod import static_polarization_weights
from bztetra.twod._grids import interpolated_triangle_energies
from bztetra.twod._grids import normalize_eigenvalues
from bztetra.twod._response_kernels import _complex_polarization_triangle_weights
from bztetra.twod._response_kernels import _complex_polarization_weights_on_local_mesh_numba
from bztetra.twod._response_kernels import _complex_polarization_weights_on_local_mesh_pair_parallel_numba
from bztetra.twod._response_kernels import _fermi_golden_rule_triangle_weights
from bztetra.twod._response_kernels import _fermi_golden_rule_weights_on_local_mesh_numba
from bztetra.twod._response_kernels import _fermi_golden_rule_weights_on_local_mesh_pair_parallel_numba
from bztetra.twod._response_kernels import _static_polarization_triangle_weights
from bztetra.twod.geometry import cached_integration_mesh
from tests.twod_cases import fermi_golden_rule_single_triangle_case
from tests.twod_cases import fermi_golden_rule_zero_case
from tests.twod_cases import static_polarization_single_triangle_case
from tests.twod_cases import synthetic_multiband_response_case


def test_twod_fermi_golden_rule_single_triangle_reference_case() -> None:
    transfer_energies, omega, expected = fermi_golden_rule_single_triangle_case()

    np.testing.assert_allclose(transfer_energies, np.array([0.0, 1.0, 2.0], dtype=np.float64))
    np.testing.assert_allclose(omega, 1.0)
    np.testing.assert_allclose(expected, np.array([1.0 / 8.0, 1.0 / 4.0, 1.0 / 8.0], dtype=np.float64))
    np.testing.assert_allclose(
        _fermi_golden_rule_triangle_weights(
            np.zeros(3, dtype=np.float64),
            transfer_energies,
            np.array([omega], dtype=np.float64),
            0.5,
        )[0],
        expected,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_twod_fermi_golden_rule_zero_case_reference_case() -> None:
    transfer_energies, omega, expected = fermi_golden_rule_zero_case()

    np.testing.assert_allclose(transfer_energies, np.array([0.0, 1.0, 2.0], dtype=np.float64))
    np.testing.assert_allclose(omega, 10.0)
    np.testing.assert_allclose(expected, 0.0)
    np.testing.assert_allclose(
        _fermi_golden_rule_triangle_weights(
            np.zeros(3, dtype=np.float64),
            transfer_energies,
            np.array([omega], dtype=np.float64),
            0.5,
        )[0],
        expected,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_twod_complex_polarization_zero_energy_matches_static_reference() -> None:
    transfer_energies, expected_static = static_polarization_single_triangle_case()

    np.testing.assert_allclose(
        _complex_polarization_triangle_weights(
            np.zeros(3, dtype=np.float64),
            transfer_energies,
            np.array([0.0 + 0.0j], dtype=np.complex128),
            0.5,
        )[0],
        _static_polarization_triangle_weights(
            np.zeros(3, dtype=np.float64),
            transfer_energies,
            0.5,
        ),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        _complex_polarization_triangle_weights(
            np.zeros(3, dtype=np.float64),
            transfer_energies,
            np.array([0.0 + 0.0j], dtype=np.complex128),
            0.5,
        )[0].real,
        expected_static,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_twod_frequency_response_public_api_zero_case_if_implemented() -> None:
    if importlib.util.find_spec("bztetra.twod.response") is None:
        pytest.skip("2D response public API is not implemented yet")

    response = importlib.import_module("bztetra.twod.response")
    occupied = np.full((2, 2, 1), -1.0, dtype=np.float64)
    target = np.full((2, 2, 1), 1.0, dtype=np.float64)

    weights = response.fermi_golden_rule_weights(
        np.eye(2, dtype=np.float64),
        occupied,
        target,
        np.array([10.0], dtype=np.float64),
    )

    assert weights.shape == (1, 2, 2, 1, 1)
    np.testing.assert_allclose(weights, 0.0)


def test_twod_fermi_golden_rule_pair_parallel_kernel_matches_serial_local_mesh() -> None:
    reciprocal_vectors, occupied, target = synthetic_multiband_response_case(band_count=4)
    occupied_flat, energy_grid_shape = normalize_eigenvalues(occupied)
    target_flat, _ = normalize_eigenvalues(target)
    mesh = cached_integration_mesh(reciprocal_vectors, energy_grid_shape, weight_grid_shape=energy_grid_shape)
    occupied_triangles = interpolated_triangle_energies(mesh, occupied_flat)
    target_triangles = interpolated_triangle_energies(mesh, target_flat)
    triangle_area = 0.5 / float(np.prod(mesh.energy_grid_shape, dtype=np.int64))
    sample_energies = np.linspace(0.0, 1.1, 7, dtype=np.float64)

    serial = _fermi_golden_rule_weights_on_local_mesh_numba(
        mesh.local_point_indices,
        occupied_triangles,
        target_triangles,
        sample_energies,
        True,
        mesh.local_point_count,
        triangle_area,
    )
    pair_parallel = _fermi_golden_rule_weights_on_local_mesh_pair_parallel_numba(
        mesh.local_point_indices,
        occupied_triangles,
        target_triangles,
        sample_energies,
        True,
        mesh.local_point_count,
        triangle_area,
    )

    np.testing.assert_allclose(pair_parallel, serial, rtol=1.0e-12, atol=1.0e-12)


def test_twod_fermi_golden_rule_preserves_unsorted_energy_order() -> None:
    reciprocal_vectors, occupied, target = synthetic_multiband_response_case(band_count=4)
    energies = np.linspace(0.0, 1.1, 7, dtype=np.float64)
    order = np.array([3, 0, 6, 2, 1, 5, 4], dtype=np.int64)

    ordered = fermi_golden_rule_weights(
        reciprocal_vectors,
        occupied,
        target,
        energies,
        weight_grid_shape=occupied.shape[:2],
        method="linear",
    )
    shuffled = fermi_golden_rule_weights(
        reciprocal_vectors,
        occupied,
        target,
        energies[order],
        weight_grid_shape=occupied.shape[:2],
        method="linear",
    )

    np.testing.assert_allclose(shuffled, ordered[order], rtol=1.0e-12, atol=1.0e-12)


def test_twod_complex_pair_parallel_kernel_matches_serial_local_mesh() -> None:
    reciprocal_vectors, occupied, target = synthetic_multiband_response_case(band_count=4)
    occupied_flat, energy_grid_shape = normalize_eigenvalues(occupied)
    target_flat, _ = normalize_eigenvalues(target)
    mesh = cached_integration_mesh(reciprocal_vectors, energy_grid_shape, weight_grid_shape=energy_grid_shape)
    occupied_triangles = interpolated_triangle_energies(mesh, occupied_flat)
    target_triangles = interpolated_triangle_energies(mesh, target_flat)
    triangle_area = 0.5 / float(np.prod(mesh.energy_grid_shape, dtype=np.int64))
    sample_energies = 1j * np.linspace(0.1, 1.4, 6, dtype=np.float64)

    serial = _complex_polarization_weights_on_local_mesh_numba(
        mesh.local_point_indices,
        occupied_triangles,
        target_triangles,
        sample_energies,
        mesh.local_point_count,
        triangle_area,
    )
    pair_parallel = _complex_polarization_weights_on_local_mesh_pair_parallel_numba(
        mesh.local_point_indices,
        occupied_triangles,
        target_triangles,
        sample_energies,
        mesh.local_point_count,
        triangle_area,
    )

    np.testing.assert_allclose(pair_parallel, serial, rtol=1.0e-12, atol=1.0e-12)


def test_twod_complex_frequency_preserves_complex_energy_order() -> None:
    reciprocal_vectors, occupied, target = synthetic_multiband_response_case(band_count=4)
    energies = np.array(
        [0.0 + 0.4j, 0.2 + 0.9j, -0.1 + 1.1j, 0.4 + 1.6j],
        dtype=np.complex128,
    )
    order = np.array([2, 0, 3, 1], dtype=np.int64)

    ordered = complex_frequency_polarization_weights(
        reciprocal_vectors,
        occupied,
        target,
        energies,
        weight_grid_shape=occupied.shape[:2],
        method="linear",
    )
    shuffled = complex_frequency_polarization_weights(
        reciprocal_vectors,
        occupied,
        target,
        energies[order],
        weight_grid_shape=occupied.shape[:2],
        method="linear",
    )

    np.testing.assert_allclose(shuffled, ordered[order], rtol=1.0e-12, atol=1.0e-12)


def test_twod_fermi_golden_rule_observables_match_full_weight_sum() -> None:
    reciprocal_vectors, occupied, target = synthetic_multiband_response_case(band_count=4)
    energies = np.linspace(0.0, 1.1, 7, dtype=np.float64)

    weights = fermi_golden_rule_weights(
        reciprocal_vectors,
        occupied,
        target,
        energies,
        weight_grid_shape=occupied.shape[:2],
        method="linear",
    )
    observed = fermi_golden_rule_observables(
        reciprocal_vectors,
        occupied,
        target,
        energies,
        weight_grid_shape=occupied.shape[:2],
        method="linear",
    )

    np.testing.assert_allclose(observed, weights.sum(axis=(1, 2, 3, 4)), rtol=1.0e-12, atol=1.0e-12)


def test_twod_static_polarization_observables_match_full_weight_sum() -> None:
    reciprocal_vectors, occupied, target = synthetic_multiband_response_case(band_count=4)

    weights = static_polarization_weights(
        reciprocal_vectors,
        occupied,
        target,
        weight_grid_shape=occupied.shape[:2],
        method="linear",
    )
    observed = static_polarization_observables(
        reciprocal_vectors,
        occupied,
        target,
        weight_grid_shape=occupied.shape[:2],
        method="linear",
    )

    np.testing.assert_allclose(observed, weights.sum(axis=(0, 1, 2, 3)), rtol=1.0e-12, atol=1.0e-12)


def test_twod_static_polarization_observables_match_explicit_channel_contraction() -> None:
    reciprocal_vectors, occupied, target = synthetic_multiband_response_case(band_count=4)
    rng = np.random.default_rng(1357)
    matrix_elements = rng.normal(
        size=(2, 3, occupied.shape[0], occupied.shape[1], target.shape[-1], occupied.shape[-1])
    )

    weights = static_polarization_weights(
        reciprocal_vectors,
        occupied,
        target,
        weight_grid_shape=occupied.shape[:2],
        method="linear",
    )
    observed = static_polarization_observables(
        reciprocal_vectors,
        occupied,
        target,
        matrix_elements=matrix_elements,
        weight_grid_shape=occupied.shape[:2],
        method="linear",
    )
    expected = (weights[None, None, ...] * matrix_elements).sum(axis=(2, 3, 4, 5))

    assert observed.shape == (2, 3)
    np.testing.assert_allclose(observed, expected, rtol=1.0e-12, atol=1.0e-12)


def test_twod_fermi_golden_rule_observables_match_explicit_channel_contraction() -> None:
    reciprocal_vectors, occupied, target = synthetic_multiband_response_case(band_count=4)
    energies = np.linspace(0.0, 1.1, 7, dtype=np.float64)
    rng = np.random.default_rng(1234)
    matrix_elements = rng.normal(
        size=(2, 3, occupied.shape[0], occupied.shape[1], target.shape[-1], occupied.shape[-1])
    )

    weights = fermi_golden_rule_weights(
        reciprocal_vectors,
        occupied,
        target,
        energies,
        weight_grid_shape=occupied.shape[:2],
        method="linear",
    )
    observed = fermi_golden_rule_observables(
        reciprocal_vectors,
        occupied,
        target,
        energies,
        matrix_elements=matrix_elements,
        weight_grid_shape=occupied.shape[:2],
        method="linear",
    )
    expected = (weights[:, None, None, ...] * matrix_elements[None, ...]).sum(axis=(3, 4, 5, 6))

    assert observed.shape == (energies.size, 2, 3)
    np.testing.assert_allclose(observed, expected, rtol=1.0e-12, atol=1.0e-12)


def test_twod_complex_frequency_observables_match_explicit_interpolated_contraction() -> None:
    reciprocal_vectors, occupied, target = synthetic_multiband_response_case(band_count=4)
    energies = np.array([0.0 + 0.4j, 0.2 + 0.9j, -0.1 + 1.1j, 0.4 + 1.6j], dtype=np.complex128)
    weight_grid_shape = (4, 4)
    rng = np.random.default_rng(5678)
    matrix_elements = (
        rng.normal(size=(2, 2, weight_grid_shape[0], weight_grid_shape[1], target.shape[-1], occupied.shape[-1]))
        + 1j
        * rng.normal(
            size=(2, 2, weight_grid_shape[0], weight_grid_shape[1], target.shape[-1], occupied.shape[-1])
        )
    )

    weights = complex_frequency_polarization_weights(
        reciprocal_vectors,
        occupied,
        target,
        energies,
        weight_grid_shape=weight_grid_shape,
        method="linear",
    )
    observed = complex_frequency_polarization_observables(
        reciprocal_vectors,
        occupied,
        target,
        energies,
        matrix_elements=matrix_elements,
        weight_grid_shape=weight_grid_shape,
        method="linear",
    )
    expected = (weights[:, None, None, ...] * matrix_elements[None, ...]).sum(axis=(3, 4, 5, 6))

    assert observed.shape == (energies.size, 2, 2)
    np.testing.assert_allclose(observed, expected, rtol=1.0e-12, atol=1.0e-12)


def test_twod_prepared_response_observables_accept_local_point_matrix_elements() -> None:
    reciprocal_vectors, occupied, target = synthetic_multiband_response_case(band_count=4)
    energies = np.linspace(0.0, 1.1, 7, dtype=np.float64)
    problem = prepare_response_evaluator(
        reciprocal_vectors,
        occupied,
        target,
        weight_grid_shape=occupied.shape[:2],
        method="linear",
    )
    rng = np.random.default_rng(2468)
    local_matrix_elements = rng.normal(
        size=(2, problem.mesh.local_point_count, target.shape[-1], occupied.shape[-1])
    )
    grid_matrix_elements = (
        local_matrix_elements.reshape((2, occupied.shape[1], occupied.shape[0], target.shape[-1], occupied.shape[-1]))
        .transpose(0, 2, 1, 3, 4)
    )

    observed = problem.fermi_golden_rule_observables(
        energies,
        matrix_elements=local_matrix_elements,
    )
    weights = problem.fermi_golden_rule_weights(energies)
    expected = (weights[:, None, ...] * grid_matrix_elements[None, ...]).sum(axis=(2, 3, 4, 5))

    assert observed.shape == (energies.size, 2)
    np.testing.assert_allclose(observed, expected, rtol=1.0e-12, atol=1.0e-12)


def test_twod_prepare_response_sweep_target_evaluator_matches_single_prepare() -> None:
    reciprocal_vectors, occupied, target = synthetic_multiband_response_case(band_count=2)
    energies = np.linspace(0.0, 1.1, 7, dtype=np.float64)

    sweep = prepare_response_sweep_evaluator(
        reciprocal_vectors,
        occupied,
        weight_grid_shape=occupied.shape[:2],
        method="linear",
    )
    prepared = prepare_response_evaluator(
        reciprocal_vectors,
        occupied,
        target,
        weight_grid_shape=occupied.shape[:2],
        method="linear",
    )
    from_sweep = sweep.prepare_target_evaluator(target)

    np.testing.assert_allclose(
        from_sweep.fermi_golden_rule_observables(energies),
        prepared.fermi_golden_rule_observables(energies),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_twod_fermi_golden_rule_observables_batch_matches_loop_and_parallel_workers() -> None:
    reciprocal_vectors, occupied, target = synthetic_multiband_response_case(band_count=2)
    energies = np.linspace(0.0, 1.1, 7, dtype=np.float64)
    target_batch = np.stack(
        (
            target,
            target + 0.05,
            target + 0.12,
        ),
        axis=0,
    )
    rng = np.random.default_rng(97531)
    matrix_elements_batch = tuple(
        rng.normal(size=(2, occupied.shape[0], occupied.shape[1], target.shape[-1], occupied.shape[-1]))
        for _ in range(target_batch.shape[0])
    )

    sweep = prepare_response_sweep_evaluator(
        reciprocal_vectors,
        occupied,
        weight_grid_shape=occupied.shape[:2],
        method="linear",
    )
    expected = np.stack(
        [
            prepare_response_evaluator(
                reciprocal_vectors,
                occupied,
                target_values,
                weight_grid_shape=occupied.shape[:2],
                method="linear",
            ).fermi_golden_rule_observables(energies, matrix_elements=target_matrix_elements)
            for target_values, target_matrix_elements in zip(target_batch, matrix_elements_batch, strict=True)
        ],
        axis=0,
    )

    serial = sweep.fermi_golden_rule_observables_batch(
        target_batch,
        energies,
        matrix_elements=matrix_elements_batch,
        workers=1,
    )
    parallel = sweep.fermi_golden_rule_observables_batch(
        target_batch,
        energies,
        matrix_elements=matrix_elements_batch,
        workers=2,
    )
    wrapped = fermi_golden_rule_observables_batch(
        reciprocal_vectors,
        occupied,
        target_batch,
        energies,
        matrix_elements=matrix_elements_batch,
        weight_grid_shape=occupied.shape[:2],
        method="linear",
        workers=2,
    )

    np.testing.assert_allclose(serial, expected, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(parallel, expected, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(wrapped, expected, rtol=1.0e-12, atol=1.0e-12)


def test_twod_retarded_response_observables_batch_matches_loop() -> None:
    reciprocal_vectors, occupied, target = synthetic_multiband_response_case(band_count=2)
    energies = np.linspace(0.0, 2.0, 129, dtype=np.float64)
    target_batch = np.stack(
        (
            target,
            target + 0.05,
            target + 0.12,
        ),
        axis=0,
    )

    sweep = prepare_response_sweep_evaluator(
        reciprocal_vectors,
        occupied,
        weight_grid_shape=occupied.shape[:2],
        method="linear",
    )
    expected = tuple(
        prepare_response_evaluator(
            reciprocal_vectors,
            occupied,
            target_values,
            weight_grid_shape=occupied.shape[:2],
            method="linear",
        ).retarded_response_observables(energies)
        for target_values in target_batch
    )

    observed = sweep.retarded_response_observables_batch(
        target_batch,
        energies,
        workers=2,
    )
    wrapped = retarded_response_observables_batch(
        reciprocal_vectors,
        occupied,
        target_batch,
        energies,
        weight_grid_shape=occupied.shape[:2],
        method="linear",
        workers=2,
    )

    assert len(observed) == len(expected)
    assert len(wrapped) == len(expected)
    for actual, reference in zip(observed, expected, strict=True):
        np.testing.assert_allclose(actual.omega, reference.omega, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(actual.imag, reference.imag, rtol=1.0e-12, atol=1.0e-12)
        np.testing.assert_allclose(actual.real, reference.real, rtol=1.0e-12, atol=1.0e-12)
    for actual, reference in zip(wrapped, expected, strict=True):
        np.testing.assert_allclose(actual.imag, reference.imag, rtol=1.0e-12, atol=1.0e-12)
        np.testing.assert_allclose(actual.real, reference.real, rtol=1.0e-12, atol=1.0e-12)


def test_twod_retarded_response_observables_match_real_axis_complex_kernel() -> None:
    reciprocal_vectors, occupied, target = synthetic_multiband_response_case(grid_shape=(12, 12), band_count=2)
    problem = prepare_response_evaluator(
        reciprocal_vectors,
        occupied,
        target,
        weight_grid_shape=occupied.shape[:2],
        method="linear",
    )
    lower_bound, upper_bound = problem.transition_energy_bounds()
    energies = np.linspace(0.0, upper_bound + 0.35, 257, dtype=np.float64)

    reconstructed = problem.retarded_response_observables(energies)
    direct = problem.complex_frequency_polarization_observables((-energies).astype(np.complex128))

    assert reconstructed.imag.shape == energies.shape
    assert reconstructed.real.shape == energies.shape
    np.testing.assert_allclose(
        reconstructed.imag,
        np.pi * problem.fermi_golden_rule_observables(energies),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(reconstructed.real[0], problem.static_polarization_observables(), rtol=1.0e-10, atol=1.0e-10)

    interior = (energies > lower_bound + 0.1) & (energies < upper_bound - 0.1)
    np.testing.assert_allclose(
        reconstructed.real[interior],
        direct.real[interior],
        rtol=4.0e-3,
        atol=4.0e-3,
    )


def test_twod_retarded_response_module_wrapper_matches_prepared_api() -> None:
    reciprocal_vectors, occupied, target = synthetic_multiband_response_case(band_count=2)
    energies = np.linspace(0.0, 2.0, 129, dtype=np.float64)

    prepared = prepare_response_evaluator(
        reciprocal_vectors,
        occupied,
        target,
        weight_grid_shape=occupied.shape[:2],
        method="linear",
    ).retarded_response_observables(energies)
    wrapped = retarded_response_observables(
        reciprocal_vectors,
        occupied,
        target,
        energies,
        weight_grid_shape=occupied.shape[:2],
        method="linear",
    )

    np.testing.assert_allclose(wrapped.omega, prepared.omega, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(wrapped.imag, prepared.imag, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(wrapped.real, prepared.real, rtol=1.0e-12, atol=1.0e-12)
