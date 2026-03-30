import numpy as np

from tetrabz.geometry import build_integration_mesh
from tetrabz.geometry import tetrahedron_offsets
from tetrabz.geometry import tetrahedron_weight_matrix
from tetrabz.geometry import trilinear_interpolation_indices


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


def test_trilinear_interpolation_indices_wrap_periodically() -> None:
    indices, weights = trilinear_interpolation_indices((2, 2, 2), np.array([0.75, 0.75, 0.75]))

    np.testing.assert_array_equal(np.sort(indices), np.arange(8, dtype=np.int64))
    np.testing.assert_allclose(weights, np.full(8, 0.125))
    np.testing.assert_allclose(weights.sum(), 1.0)
