"""Compatibility shim while the package name transitions to ``tetrabz``."""

from tetrabz import __version__
from tetrabz import IntegrationMesh
from tetrabz import build_integration_mesh
from tetrabz import tetrahedron_offsets
from tetrabz import tetrahedron_weight_matrix
from tetrabz import trilinear_interpolation_indices

__all__ = [
    "__version__",
    "IntegrationMesh",
    "build_integration_mesh",
    "tetrahedron_offsets",
    "tetrahedron_weight_matrix",
    "trilinear_interpolation_indices",
]
