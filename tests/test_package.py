from libtetra import __version__ as libtetra_version
from libtetra import build_integration_mesh as compat_build_integration_mesh
from tetrabz import __version__
from tetrabz import build_integration_mesh


def test_package_version_is_exposed() -> None:
    assert __version__ == "0.2.0"
    assert libtetra_version == __version__


def test_compatibility_alias_exports_geometry_entrypoints() -> None:
    assert compat_build_integration_mesh is build_integration_mesh
