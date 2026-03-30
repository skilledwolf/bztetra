import numpy as np

from tetrabz import PreparedResponseProblem
from tetrabz import dbldelta
from tetrabz import dblstep
from tetrabz import fermigr
from tetrabz import polcmplx
from tetrabz import polstat
from tetrabz import prepare_response_problem
from tests.legacy_cases import fermigr_energy_points
from tests.legacy_cases import legacy_free_electron_response_case
from tests.legacy_cases import polcmplx_energy_points


def _synthetic_multiband_response_case(
    grid_shape: tuple[int, int, int],
    band_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reciprocal_vectors = np.diag([3.0, 3.0, 3.0]).astype(np.float64)
    occupied = np.empty((*grid_shape, band_count), dtype=np.float64)
    target = np.empty((*grid_shape, band_count), dtype=np.float64)

    grid = np.asarray(grid_shape, dtype=np.int64)
    half_grid = grid // 2
    nx, ny, nz = grid_shape
    for x_index in range(nx):
        for y_index in range(ny):
            for z_index in range(nz):
                integer_indices = np.array((x_index, y_index, z_index), dtype=np.int64)
                centered = np.mod(integer_indices + half_grid, grid) - half_grid
                kvec = reciprocal_vectors @ (centered.astype(np.float64) / grid.astype(np.float64))
                base = 0.5 * float(np.dot(kvec, kvec)) - 0.5
                shifted = kvec.copy()
                shifted[0] += 1.0
                shifted_base = 0.5 * float(np.dot(shifted, shifted)) - 0.5
                for band_index in range(band_count):
                    occupied[x_index, y_index, z_index, band_index] = base + 0.18 * band_index
                    target[x_index, y_index, z_index, band_index] = (
                        shifted_base + 0.18 * band_index + 0.08 * (band_index % 3)
                    )

    return reciprocal_vectors, occupied, target


def test_prepare_response_problem_returns_public_type() -> None:
    bvec, eigenvalues_1, eigenvalues_2, _ = legacy_free_electron_response_case((8, 8, 8), (8, 8, 8))

    problem = prepare_response_problem(
        bvec,
        eigenvalues_1 - 0.5,
        eigenvalues_2 - 0.5,
        weight_grid_shape=(8, 8, 8),
        method="optimized",
    )

    assert isinstance(problem, PreparedResponseProblem)


def test_prepared_response_problem_matches_static_response_kernels() -> None:
    bvec, eigenvalues_1, eigenvalues_2, _ = legacy_free_electron_response_case((8, 8, 8), (8, 8, 8))

    problem = prepare_response_problem(
        bvec,
        eigenvalues_1 - 0.5,
        eigenvalues_2 - 0.5,
        weight_grid_shape=(8, 8, 8),
        method="optimized",
    )

    np.testing.assert_allclose(problem.dblstep(), dblstep(bvec, eigenvalues_1 - 0.5, eigenvalues_2 - 0.5, weight_grid_shape=(8, 8, 8), method="optimized"))
    np.testing.assert_allclose(problem.dbldelta(), dbldelta(bvec, eigenvalues_1 - 0.5, eigenvalues_2 - 0.5, weight_grid_shape=(8, 8, 8), method="optimized"))
    np.testing.assert_allclose(problem.polstat(), polstat(bvec, eigenvalues_1 - 0.5, eigenvalues_2 - 0.5, weight_grid_shape=(8, 8, 8), method="optimized"))


def test_prepared_response_problem_matches_frequency_response_kernels() -> None:
    bvec, eigenvalues_1, eigenvalues_2, _ = legacy_free_electron_response_case((8, 8, 8), (8, 8, 8))

    problem = prepare_response_problem(
        bvec,
        eigenvalues_1 - 0.5,
        eigenvalues_2 - 0.5,
        weight_grid_shape=(8, 8, 8),
        method="optimized",
    )

    np.testing.assert_allclose(
        problem.fermigr(fermigr_energy_points()),
        fermigr(
            bvec,
            eigenvalues_1 - 0.5,
            eigenvalues_2 - 0.5,
            fermigr_energy_points(),
            weight_grid_shape=(8, 8, 8),
            method="optimized",
        ),
    )


def test_prepared_response_problem_matches_interpolated_frequency_kernels() -> None:
    bvec, eigenvalues_1, eigenvalues_2, _ = legacy_free_electron_response_case((16, 16, 16), (8, 8, 8))

    problem = prepare_response_problem(
        bvec,
        eigenvalues_1 - 0.5,
        eigenvalues_2 - 0.5,
        weight_grid_shape=(8, 8, 8),
        method="optimized",
    )

    np.testing.assert_allclose(
        problem.fermigr(fermigr_energy_points()),
        fermigr(
            bvec,
            eigenvalues_1 - 0.5,
            eigenvalues_2 - 0.5,
            fermigr_energy_points(),
            weight_grid_shape=(8, 8, 8),
            method="optimized",
        ),
    )
    np.testing.assert_allclose(
        problem.polcmplx(polcmplx_energy_points()),
        polcmplx(
            bvec,
            eigenvalues_1 - 0.5,
            eigenvalues_2 - 0.5,
            polcmplx_energy_points(),
            weight_grid_shape=(8, 8, 8),
            method="optimized",
        ),
    )


def test_prepared_response_problem_matches_multiband_frequency_response_kernels() -> None:
    bvec, occupied, target = _synthetic_multiband_response_case((4, 4, 4), 5)

    problem = prepare_response_problem(
        bvec,
        occupied,
        target,
        weight_grid_shape=(4, 4, 4),
        method="optimized",
    )

    real_energies = np.linspace(0.0, 1.25, 8, dtype=np.float64)
    complex_energies = 1j * np.linspace(0.1, 1.5, 8, dtype=np.float64)

    np.testing.assert_allclose(
        problem.fermigr(real_energies),
        fermigr(
            bvec,
            occupied,
            target,
            real_energies,
            weight_grid_shape=(4, 4, 4),
            method="optimized",
        ),
    )
    np.testing.assert_allclose(
        problem.polcmplx(complex_energies),
        polcmplx(
            bvec,
            occupied,
            target,
            complex_energies,
            weight_grid_shape=(4, 4, 4),
            method="optimized",
        ),
    )
