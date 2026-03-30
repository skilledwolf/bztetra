from __future__ import annotations

import importlib
import importlib.util

import numpy as np
import pytest

from tests.twod_cases import fermi_golden_rule_single_triangle_case
from tests.twod_cases import fermi_golden_rule_zero_case


def test_twod_fermi_golden_rule_single_triangle_reference_case() -> None:
    transfer_energies, omega, expected = fermi_golden_rule_single_triangle_case()

    np.testing.assert_allclose(transfer_energies, np.array([0.0, 1.0, 2.0], dtype=np.float64))
    np.testing.assert_allclose(omega, 1.0)
    np.testing.assert_allclose(expected, np.array([1.0 / 8.0, 1.0 / 4.0, 1.0 / 8.0], dtype=np.float64))


def test_twod_fermi_golden_rule_zero_case_reference_case() -> None:
    transfer_energies, omega, expected = fermi_golden_rule_zero_case()

    np.testing.assert_allclose(transfer_energies, np.array([0.0, 1.0, 2.0], dtype=np.float64))
    np.testing.assert_allclose(omega, 10.0)
    np.testing.assert_allclose(expected, 0.0)


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
