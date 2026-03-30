import numpy as np
import pytest

from bztetra.twod.geometry import bilinear_interpolation_indices
from bztetra.twod.geometry import build_integration_mesh
from bztetra.twod.geometry import triangle_offsets


def test_triangle_offsets_split_each_cell_into_two_triangles() -> None:
    offsets = triangle_offsets(np.eye(2, dtype=np.float64), (2, 2))

    assert offsets.shape == (2, 3, 2)
    assert {tuple(vertex) for vertex in offsets[0]} != {tuple(vertex) for vertex in offsets[1]}


def test_build_integration_mesh_without_interpolation_tracks_expected_counts() -> None:
    mesh = build_integration_mesh(np.eye(2, dtype=np.float64), (2, 2))

    assert mesh.triangle_count == 8
    assert mesh.local_point_count == 4
    assert mesh.interpolation_required is False
    np.testing.assert_array_equal(mesh.global_point_indices, mesh.local_point_indices)


def test_bilinear_interpolation_indices_wrap_periodically() -> None:
    indices, weights = bilinear_interpolation_indices((2, 2), np.array([0.75, 0.75]))

    np.testing.assert_array_equal(np.sort(indices), np.arange(4, dtype=np.int64))
    np.testing.assert_allclose(weights, np.full(4, 0.25))
    np.testing.assert_allclose(weights.sum(), 1.0)


def test_twod_mesh_rejects_non_linear_method() -> None:
    with pytest.raises(ValueError, match="only the linear triangle method"):
        build_integration_mesh(np.eye(2, dtype=np.float64), (8, 8), method="optimized")
