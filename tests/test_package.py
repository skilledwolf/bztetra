from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

from bztetra import __version__
from bztetra import PreparedResponseEvaluator
from bztetra import build_integration_mesh
from bztetra import occupation_weights
from bztetra import complex_frequency_polarization_weights
from bztetra import prepare_response_evaluator
from bztetra import small_tetrahedron_cut


def test_package_version_is_exposed() -> None:
    assert __version__
    try:
        installed_version = package_version("bztetra")
    except PackageNotFoundError:
        assert __version__ == "0+unknown"
    else:
        assert __version__ == installed_version


def test_core_exports_are_available() -> None:
    assert callable(build_integration_mesh)
    assert callable(occupation_weights)
    assert callable(complex_frequency_polarization_weights)
    assert callable(prepare_response_evaluator)
    assert callable(PreparedResponseEvaluator)
    assert callable(small_tetrahedron_cut)
