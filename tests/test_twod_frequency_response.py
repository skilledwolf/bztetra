from __future__ import annotations

import importlib
import importlib.util

import numpy as np
import pytest

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
        mesh.local_point_count,
        triangle_area,
    )
    pair_parallel = _fermi_golden_rule_weights_on_local_mesh_pair_parallel_numba(
        mesh.local_point_indices,
        occupied_triangles,
        target_triangles,
        sample_energies,
        mesh.local_point_count,
        triangle_area,
    )

    np.testing.assert_allclose(pair_parallel, serial, rtol=1.0e-12, atol=1.0e-12)


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
