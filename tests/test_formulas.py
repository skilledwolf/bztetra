import math

import numpy as np
import pytest

from tetrabz.formulas import simplex_affine_coefficients
from tetrabz.formulas import small_tetrahedron_cut
from tetrabz.formulas import triangle_cut


def test_simplex_affine_coefficients_match_hand_computed_entries() -> None:
    coefficients = simplex_affine_coefficients(np.array([-2.0, 1.0, 3.0, 5.0]))

    expected = np.array(
        [
            [math.nan, 1.0 / 3.0, 3.0 / 5.0, 5.0 / 7.0],
            [2.0 / 3.0, math.nan, 3.0 / 2.0, 5.0 / 4.0],
            [2.0 / 5.0, -1.0 / 2.0, math.nan, 5.0 / 2.0],
            [2.0 / 7.0, -1.0 / 4.0, -3.0 / 2.0, math.nan],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(coefficients, expected, equal_nan=True)


def test_small_tetrahedron_cut_a1_matches_hand_computed_values() -> None:
    cut = small_tetrahedron_cut("a1", np.array([-2.0, 1.0, 3.0, 5.0]))

    assert cut.volume_factor == pytest.approx(8.0 / 105.0)
    expected = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0 / 3.0, 2.0 / 3.0, 0.0, 0.0],
            [3.0 / 5.0, 0.0, 2.0 / 5.0, 0.0],
            [5.0 / 7.0, 0.0, 0.0, 2.0 / 7.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(cut.coefficients, expected)


def test_small_tetrahedron_cut_b2_matches_hand_computed_values() -> None:
    cut = small_tetrahedron_cut("b2", np.array([-4.0, -1.0, 2.0, 5.0]))

    assert cut.volume_factor == pytest.approx(1.0 / 18.0)
    expected = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 2.0 / 3.0, 1.0 / 3.0, 0.0],
            [0.0, 5.0 / 6.0, 0.0, 1.0 / 6.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(cut.coefficients, expected)


def test_triangle_cut_c1_matches_hand_computed_values() -> None:
    cut = triangle_cut("c1", np.array([-5.0, -3.0, -1.0, 2.0]))

    assert cut.volume_factor == pytest.approx(4.0 / 35.0)
    expected = np.array(
        [
            [2.0 / 7.0, 0.0, 0.0, 5.0 / 7.0],
            [0.0, 2.0 / 5.0, 0.0, 3.0 / 5.0],
            [0.0, 0.0, 2.0 / 3.0, 1.0 / 3.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(cut.coefficients, expected)


@pytest.mark.parametrize(
    ("kind", "energies"),
    [
        ("a1", np.array([-2.0, 1.0, 3.0, 5.0])),
        ("b1", np.array([-4.0, -1.0, 2.0, 5.0])),
        ("b2", np.array([-4.0, -1.0, 2.0, 5.0])),
        ("b3", np.array([-4.0, -1.0, 2.0, 5.0])),
        ("c1", np.array([-5.0, -3.0, -1.0, 2.0])),
        ("c2", np.array([-5.0, -3.0, -1.0, 2.0])),
        ("c3", np.array([-5.0, -3.0, -1.0, 2.0])),
    ],
)
def test_small_tetrahedron_rows_are_barycentric(kind: str, energies: np.ndarray) -> None:
    cut = small_tetrahedron_cut(kind, energies)
    np.testing.assert_allclose(cut.coefficients.sum(axis=1), np.ones(cut.coefficients.shape[0]))
    assert cut.volume_factor > 0.0


@pytest.mark.parametrize(
    ("kind", "energies"),
    [
        ("a1", np.array([-2.0, 1.0, 3.0, 5.0])),
        ("b1", np.array([-4.0, -1.0, 2.0, 5.0])),
        ("b2", np.array([-4.0, -1.0, 2.0, 5.0])),
        ("c1", np.array([-5.0, -3.0, -1.0, 2.0])),
    ],
)
def test_triangle_rows_are_barycentric(kind: str, energies: np.ndarray) -> None:
    cut = triangle_cut(kind, energies)
    np.testing.assert_allclose(cut.coefficients.sum(axis=1), np.ones(cut.coefficients.shape[0]))
    assert cut.volume_factor > 0.0


def test_cut_helpers_require_strictly_sorted_distinct_energies() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        small_tetrahedron_cut("a1", np.array([-2.0, -2.0, 1.0, 3.0]))

    with pytest.raises(ValueError, match="strictly increasing"):
        triangle_cut("a1", np.array([1.0, -1.0, 2.0, 3.0]))
