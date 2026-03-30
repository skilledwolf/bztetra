from tetrabz import __version__
from tetrabz import build_integration_mesh
from tetrabz import occ
from tetrabz import small_tetrahedron_cut


def test_package_version_is_exposed() -> None:
    assert __version__ == "0.3.0"


def test_core_exports_are_available() -> None:
    assert callable(build_integration_mesh)
    assert callable(occ)
    assert callable(small_tetrahedron_cut)
