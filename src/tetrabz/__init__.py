"""Modern Python port of the legacy libtetrabz tetrahedron routines."""

from ._version import __version__
from .geometry import IntegrationMesh
from .geometry import build_integration_mesh
from .geometry import tetrahedron_offsets
from .geometry import tetrahedron_weight_matrix
from .geometry import trilinear_interpolation_indices

__all__ = [
    "IntegrationMesh",
    "__version__",
    "build_integration_mesh",
    "tetrahedron_offsets",
    "tetrahedron_weight_matrix",
    "trilinear_interpolation_indices",
]

