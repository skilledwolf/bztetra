from __future__ import annotations

import importlib
import importlib.util

import numpy as np
import pytest

from bztetra.twod._grids import interpolated_triangle_energies
from bztetra.twod._grids import normalize_eigenvalues
from bztetra.twod._response_kernels import _nesting_function_triangle_weights
from bztetra.twod._response_kernels import _phase_space_overlap_triangle_weights
from bztetra.twod._response_kernels import _static_polarization_weights_on_local_mesh_numba
from bztetra.twod._response_kernels import _static_polarization_weights_on_local_mesh_pair_parallel_numba
from bztetra.twod._response_kernels import _static_polarization_triangle_weights
from bztetra.twod.geometry import cached_integration_mesh
from tests.twod_cases import nesting_single_triangle_case
from tests.twod_cases import phase_space_overlap_empty_triangle_case
from tests.twod_cases import phase_space_overlap_equal_triangle_case
from tests.twod_cases import phase_space_overlap_full_triangle_case
from tests.twod_cases import static_polarization_single_triangle_case
from tests.twod_cases import synthetic_multiband_response_case


def test_twod_phase_space_overlap_full_triangle_reference_case() -> None:
    occupied, target, expected = phase_space_overlap_full_triangle_case()

    np.testing.assert_allclose(occupied, np.array([-1.0, -1.0, -1.0], dtype=np.float64))
    np.testing.assert_allclose(target, np.array([-2.0, -2.0, -2.0], dtype=np.float64))
    np.testing.assert_allclose(expected, np.full(3, 1.0 / 6.0, dtype=np.float64))
    np.testing.assert_allclose(
        _phase_space_overlap_triangle_weights(occupied, target, 0.5),
        expected,
    )


def test_twod_phase_space_overlap_empty_triangle_reference_case() -> None:
    occupied, target, expected = phase_space_overlap_empty_triangle_case()

    np.testing.assert_allclose(occupied, np.array([-1.0, -1.0, -1.0], dtype=np.float64))
    np.testing.assert_allclose(target, np.array([1.0, 1.0, 1.0], dtype=np.float64))
    np.testing.assert_allclose(expected, 0.0)
    np.testing.assert_allclose(
        _phase_space_overlap_triangle_weights(occupied, target, 0.5),
        expected,
    )


def test_twod_phase_space_overlap_equal_triangle_reference_case() -> None:
    occupied, target, expected = phase_space_overlap_equal_triangle_case()

    np.testing.assert_allclose(occupied, np.array([-1.0, -1.0, -1.0], dtype=np.float64))
    np.testing.assert_allclose(target, np.array([-1.0, -1.0, -1.0], dtype=np.float64))
    np.testing.assert_allclose(expected, np.full(3, 1.0 / 12.0, dtype=np.float64))
    np.testing.assert_allclose(
        _phase_space_overlap_triangle_weights(occupied, target, 0.5),
        expected,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_twod_nesting_single_triangle_reference_case() -> None:
    source, target, expected = nesting_single_triangle_case()

    np.testing.assert_allclose(source, np.array([-1.0, 1.0, 0.0], dtype=np.float64))
    np.testing.assert_allclose(target, np.array([-1.0, 0.0, 1.0], dtype=np.float64))
    np.testing.assert_allclose(expected, np.full(3, 1.0 / 9.0, dtype=np.float64))
    np.testing.assert_allclose(
        _nesting_function_triangle_weights(source, target, 0.5),
        expected,
    )


def test_twod_static_polarization_single_triangle_reference_case() -> None:
    transfer_energies, expected = static_polarization_single_triangle_case()

    np.testing.assert_allclose(transfer_energies, np.array([0.0, 1.0, 2.0], dtype=np.float64))
    np.testing.assert_allclose(
        expected,
        0.5 * np.array(
            [
                np.log(2.0),
                2.0 * np.log(2.0) - 1.0,
                1.0 - np.log(2.0),
            ],
            dtype=np.float64,
        ),
    )
    np.testing.assert_allclose(
        _static_polarization_triangle_weights(
            np.zeros(3, dtype=np.float64),
            transfer_energies,
            0.5,
        ),
        expected,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_twod_response_public_api_zero_case_if_implemented() -> None:
    if importlib.util.find_spec("bztetra.twod.response") is None:
        pytest.skip("2D response public API is not implemented yet")

    response = importlib.import_module("bztetra.twod.response")
    occupied = np.full((2, 2, 1), -1.0, dtype=np.float64)
    target = np.full((2, 2, 1), 1.0, dtype=np.float64)

    weights = response.phase_space_overlap_weights(np.eye(2, dtype=np.float64), occupied, target)

    assert weights.shape == (2, 2, 1, 1)
    np.testing.assert_allclose(weights, 0.0)


def test_twod_response_public_api_equal_spectra_returns_half_weight_if_implemented() -> None:
    if importlib.util.find_spec("bztetra.twod.response") is None:
        pytest.skip("2D response public API is not implemented yet")

    response = importlib.import_module("bztetra.twod.response")
    occupied = np.full((2, 2, 1), -1.0, dtype=np.float64)

    weights = response.phase_space_overlap_weights(np.eye(2, dtype=np.float64), occupied, occupied)

    assert weights.shape == (2, 2, 1, 1)
    np.testing.assert_allclose(weights.sum(), 0.5, rtol=1.0e-12, atol=1.0e-12)


def test_twod_static_pair_parallel_kernel_matches_serial_local_mesh() -> None:
    reciprocal_vectors, occupied, target = synthetic_multiband_response_case(band_count=4)
    occupied_flat, energy_grid_shape = normalize_eigenvalues(occupied)
    target_flat, _ = normalize_eigenvalues(target)
    mesh = cached_integration_mesh(reciprocal_vectors, energy_grid_shape, weight_grid_shape=energy_grid_shape)
    occupied_triangles = interpolated_triangle_energies(mesh, occupied_flat)
    target_triangles = interpolated_triangle_energies(mesh, target_flat)
    triangle_area = 0.5 / float(np.prod(mesh.energy_grid_shape, dtype=np.int64))

    serial = _static_polarization_weights_on_local_mesh_numba(
        mesh.local_point_indices,
        occupied_triangles,
        target_triangles,
        mesh.local_point_count,
        triangle_area,
    )
    pair_parallel = _static_polarization_weights_on_local_mesh_pair_parallel_numba(
        mesh.local_point_indices,
        occupied_triangles,
        target_triangles,
        mesh.local_point_count,
        triangle_area,
    )

    np.testing.assert_allclose(pair_parallel, serial, rtol=1.0e-12, atol=1.0e-12)


def test_twod_nesting_public_api_exposes_pair_band_last_layout_on_multiband_case() -> None:
    reciprocal_vectors, occupied, target = synthetic_multiband_response_case(band_count=4)

    weights = importlib.import_module("bztetra.twod.response").nesting_function_weights(
        reciprocal_vectors,
        occupied,
        target,
        weight_grid_shape=occupied.shape[:2],
    )

    assert weights.shape == (occupied.shape[0], occupied.shape[1], 4, 4)
    assert np.isfinite(weights).all()
