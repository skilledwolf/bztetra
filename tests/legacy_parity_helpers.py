from __future__ import annotations

import numpy as np

from tests.legacy_cases import brillouin_zone_volume


def normalize_port_dos_output(array: np.ndarray) -> np.ndarray:
    return np.transpose(np.asarray(array), (1, 2, 3, 4, 0))


def normalize_port_frequency_output(array: np.ndarray) -> np.ndarray:
    return np.transpose(np.asarray(array), (1, 2, 3, 5, 4, 0))


def swap_pair_axes(array: np.ndarray) -> np.ndarray:
    return np.transpose(np.asarray(array), (0, 1, 2, 4, 3))


def legacy_electron_count_per_spin(bvec: np.ndarray) -> float:
    vbz = float(np.linalg.det(bvec))
    return (4.0 * np.pi / 3.0 + np.sqrt(2.0) * np.pi / 3.0) / vbz


def weighted_integrals(
    weights: np.ndarray,
    metric: np.ndarray,
    reciprocal_vectors: np.ndarray,
) -> np.ndarray:
    values = (weights * metric[None, ..., None]).sum(axis=(1, 2, 3)) * brillouin_zone_volume(
        reciprocal_vectors
    )
    return np.squeeze(values)


def weighted_matrix(
    weights: np.ndarray,
    metric: np.ndarray,
    reciprocal_vectors: np.ndarray,
) -> np.ndarray:
    return (weights * metric[..., None, None]).sum(axis=(0, 1, 2)) * brillouin_zone_volume(
        reciprocal_vectors
    )


def weighted_energy_matrix(
    weights: np.ndarray,
    metric: np.ndarray,
    reciprocal_vectors: np.ndarray,
) -> np.ndarray:
    return (weights * metric[None, ..., None, None]).sum(axis=(1, 2, 3)) * brillouin_zone_volume(
        reciprocal_vectors
    )


def assert_with_entrywise_atol(
    actual: np.ndarray | float | complex,
    expected: np.ndarray | float | complex,
    atol: np.ndarray | float,
    *,
    label: str,
) -> None:
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    atol_array = np.asarray(atol)

    if actual_array.shape != expected_array.shape:
        raise AssertionError(
            f"{label}: shape mismatch {actual_array.shape!r} vs {expected_array.shape!r}"
        )

    difference = np.abs(actual_array - expected_array)
    allowed = atol_array + 1.0e-12
    excess = difference - allowed
    if np.any(excess > 0.0):
        failing_index = np.unravel_index(int(np.argmax(excess)), excess.shape)
        raise AssertionError(
            f"{label}: max diff {difference[failing_index]!r} exceeds "
            f"allowed {allowed[failing_index]!r} at index {failing_index}"
        )
