from tetrabz import __version__
from tetrabz import PreparedResponseEvaluator
from tetrabz import build_integration_mesh
from tetrabz import occupation_weights
from tetrabz import complex_frequency_polarization_weights
from tetrabz import prepare_response_evaluator
from tetrabz import small_tetrahedron_cut


def test_package_version_is_exposed() -> None:
    assert __version__ == "0.4.0"


def test_core_exports_are_available() -> None:
    assert callable(build_integration_mesh)
    assert callable(occupation_weights)
    assert callable(complex_frequency_polarization_weights)
    assert callable(prepare_response_evaluator)
    assert callable(PreparedResponseEvaluator)
    assert callable(small_tetrahedron_cut)
