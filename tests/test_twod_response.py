from __future__ import annotations

import importlib
import importlib.util

import numpy as np
import pytest

from tests.twod_cases import nesting_single_triangle_case
from tests.twod_cases import phase_space_overlap_empty_triangle_case
from tests.twod_cases import phase_space_overlap_full_triangle_case
from tests.twod_cases import static_polarization_single_triangle_case


def test_twod_phase_space_overlap_full_triangle_reference_case() -> None:
    occupied, target, expected = phase_space_overlap_full_triangle_case()

    np.testing.assert_allclose(occupied, np.array([-1.0, -1.0, -1.0], dtype=np.float64))
    np.testing.assert_allclose(target, np.array([1.0, 1.0, 1.0], dtype=np.float64))
    np.testing.assert_allclose(expected, np.full(3, 1.0 / 6.0, dtype=np.float64))


def test_twod_phase_space_overlap_empty_triangle_reference_case() -> None:
    occupied, target, expected = phase_space_overlap_empty_triangle_case()

    np.testing.assert_allclose(occupied, np.array([1.0, 1.0, 1.0], dtype=np.float64))
    np.testing.assert_allclose(target, np.array([-1.0, -1.0, -1.0], dtype=np.float64))
    np.testing.assert_allclose(expected, 0.0)


def test_twod_nesting_single_triangle_reference_case() -> None:
    source, target, expected = nesting_single_triangle_case()

    np.testing.assert_allclose(source, np.array([-1.0, 1.0, 0.0], dtype=np.float64))
    np.testing.assert_allclose(target, np.array([-1.0, 0.0, 1.0], dtype=np.float64))
    np.testing.assert_allclose(expected, np.full(3, 1.0 / 9.0, dtype=np.float64))


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


def test_twod_response_public_api_zero_case_if_implemented() -> None:
    if importlib.util.find_spec("bztetra.twod.response") is None:
        pytest.skip("2D response public API is not implemented yet")

    response = importlib.import_module("bztetra.twod.response")
    occupied = np.full((2, 2, 1), -1.0, dtype=np.float64)
    target = np.full((2, 2, 1), 1.0, dtype=np.float64)

    weights = response.phase_space_overlap_weights(np.eye(2, dtype=np.float64), occupied, target)

    assert weights.shape == (2, 2, 1, 1)
    np.testing.assert_allclose(weights, 0.0)
