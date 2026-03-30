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
