import numpy as np
import pytest

from bztetra.geometry import build_integration_mesh
from bztetra.geometry import cached_integration_mesh
from bztetra.geometry import tetrahedron_offsets
from bztetra.geometry import tetrahedron_weight_matrix
from bztetra.geometry import trilinear_interpolation_indices
from bztetra._grids import interpolate_local_values


def test_linear_weight_matrix_matches_legacy_identity_pattern() -> None:
    expected = np.zeros((4, 20), dtype=np.float64)
    expected[np.arange(4), np.arange(4)] = 1.0
    np.testing.assert_allclose(tetrahedron_weight_matrix("linear"), expected)


def test_optimized_weight_matrix_matches_legacy_coefficients() -> None:
    matrix = tetrahedron_weight_matrix("optimized")
    np.testing.assert_allclose(matrix[0, :4], np.array([1440.0, 0.0, 30.0, 0.0]) / 1260.0)
    np.testing.assert_allclose(matrix[:, 16:], np.array(
        [
            [-18.0, -18.0, 12.0, -18.0],
            [-18.0, -18.0, -18.0, 12.0],
            [12.0, -18.0, -18.0, -18.0],
            [-18.0, 12.0, -18.0, -18.0],
        ]
    ) / 1260.0)


def test_tetrahedron_offsets_match_cubic_shortest_diagonal_choice() -> None:
    offsets = tetrahedron_offsets(np.eye(3, dtype=np.float64), (2, 2, 2))
    expected_corners = np.array(
        [
            [[1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
            [[1, 0, 0], [0, 0, 0], [0, 0, 1], [0, 1, 1]],
            [[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1]],
            [[1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1]],
            [[1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 1, 1]],
            [[1, 0, 0], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(offsets[:, :4, :], expected_corners)


def test_build_integration_mesh_without_interpolation_keeps_global_indices() -> None:
    mesh = build_integration_mesh(np.eye(3, dtype=np.float64), (2, 2, 2), method="linear")

    assert mesh.method == 1
    assert mesh.interpolation_required is False
    assert mesh.tetrahedron_count == 48
    assert mesh.local_point_count == 8
    assert mesh.fractional_kpoints is None
    assert mesh.interpolation_indices is None
    assert mesh.interpolation_weights is None
    np.testing.assert_array_equal(mesh.global_point_indices, mesh.local_point_indices)
    np.testing.assert_array_equal(mesh.global_point_indices[0, :4], np.array([1, 0, 2, 6]))


def test_build_integration_mesh_with_interpolation_tracks_unique_points() -> None:
    mesh = build_integration_mesh(
        np.eye(3, dtype=np.float64),
        (2, 2, 2),
        weight_grid_shape=(4, 4, 4),
    )

    assert mesh.interpolation_required is True
    assert mesh.local_point_count == 8
    assert mesh.fractional_kpoints is not None
    assert mesh.interpolation_indices is not None
    assert mesh.interpolation_weights is not None
    np.testing.assert_array_equal(np.unique(mesh.local_point_indices), np.arange(8, dtype=np.int64))
    expected_kpoints = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        ],
        dtype=np.float64,
    )
    actual = np.array(sorted(map(tuple, mesh.fractional_kpoints.tolist())))
    expected = np.array(sorted(map(tuple, expected_kpoints.tolist())))
    np.testing.assert_allclose(actual, expected)
    for local_index in range(mesh.local_point_count):
        expected_indices, expected_weights = trilinear_interpolation_indices(
            mesh.weight_grid_shape,
            mesh.fractional_kpoints[local_index],
        )
        np.testing.assert_array_equal(mesh.interpolation_indices[local_index], expected_indices)
        np.testing.assert_allclose(mesh.interpolation_weights[local_index], expected_weights)


def test_cached_integration_mesh_matches_direct_builder() -> None:
    reciprocal_vectors = np.diag([3.0, 3.0, 3.0]).astype(np.float64)
    direct = build_integration_mesh(
        reciprocal_vectors,
        (4, 4, 4),
        weight_grid_shape=(8, 8, 8),
        method="optimized",
    )
    cached = cached_integration_mesh(
        reciprocal_vectors,
        (4, 4, 4),
        weight_grid_shape=(8, 8, 8),
        method="optimized",
    )

    assert direct.method == cached.method
    assert direct.energy_grid_shape == cached.energy_grid_shape
    assert direct.weight_grid_shape == cached.weight_grid_shape
    assert direct.interpolation_required == cached.interpolation_required
    assert direct.local_point_count == cached.local_point_count
    assert direct.tetrahedron_count == cached.tetrahedron_count
    np.testing.assert_array_equal(direct.tetrahedron_weight_matrix, cached.tetrahedron_weight_matrix)
    np.testing.assert_array_equal(direct.tetrahedra_offsets, cached.tetrahedra_offsets)
    np.testing.assert_array_equal(direct.global_point_indices, cached.global_point_indices)
    np.testing.assert_array_equal(direct.local_point_indices, cached.local_point_indices)
    assert direct.fractional_kpoints is not None
    assert cached.fractional_kpoints is not None
    assert direct.interpolation_indices is not None
    assert cached.interpolation_indices is not None
    assert direct.interpolation_weights is not None
    assert cached.interpolation_weights is not None
    np.testing.assert_allclose(direct.fractional_kpoints, cached.fractional_kpoints)
    np.testing.assert_array_equal(direct.interpolation_indices, cached.interpolation_indices)
    np.testing.assert_allclose(direct.interpolation_weights, cached.interpolation_weights)


def test_trilinear_interpolation_indices_wrap_periodically() -> None:
    indices, weights = trilinear_interpolation_indices((2, 2, 2), np.array([0.75, 0.75, 0.75]))

    np.testing.assert_array_equal(np.sort(indices), np.arange(8, dtype=np.int64))
    np.testing.assert_allclose(weights, np.full(8, 0.125))
    np.testing.assert_allclose(weights.sum(), 1.0)


def test_interpolate_local_values_matches_on_demand_stencil_application() -> None:
    mesh = build_integration_mesh(
        np.eye(3, dtype=np.float64),
        (2, 2, 2),
        weight_grid_shape=(4, 4, 4),
    )

    local_values = np.arange(mesh.local_point_count * 3, dtype=np.float64).reshape(mesh.local_point_count, 3)
    interpolated = interpolate_local_values(mesh, local_values)
    expected = np.zeros_like(interpolated)

    for local_index, kpoint in enumerate(mesh.fractional_kpoints):
        indices, weights = trilinear_interpolation_indices(mesh.weight_grid_shape, kpoint)
        for feature_index in range(local_values.shape[1]):
            np.add.at(
                expected[:, feature_index],
                indices,
                weights * local_values[local_index, feature_index],
            )

    np.testing.assert_allclose(interpolated, expected)


def test_build_integration_mesh_rejects_two_dimensional_grid_shapes_explicitly() -> None:
    with pytest.raises(ValueError, match="supports only 3D regular grids"):
        build_integration_mesh(np.eye(3, dtype=np.float64), (16, 16))


def test_build_integration_mesh_rejects_two_dimensional_reciprocal_bases_explicitly() -> None:
    with pytest.raises(ValueError, match="supports only 3D reciprocal bases"):
        build_integration_mesh(np.eye(2, dtype=np.float64), (16, 16, 1))


def test_build_integration_mesh_rejects_strictly_2d_shapes_with_clear_message() -> None:
    with pytest.raises(ValueError, match="currently supports only 3D"):
        build_integration_mesh(np.eye(2, dtype=np.float64), (16, 16))
