"""Compatibility shim while the package name transitions to ``tetrabz``."""

from tetrabz import __version__
from tetrabz import IntegrationMesh
from tetrabz import SimplexCut
from tetrabz import build_integration_mesh
from tetrabz import fermieng
from tetrabz import occ
from tetrabz import occupation_weights
from tetrabz import simplex_affine_coefficients
from tetrabz import solve_fermi_energy
from tetrabz import small_tetrahedron_cut
from tetrabz import tetrahedron_offsets
from tetrabz import tetrahedron_weight_matrix
from tetrabz import triangle_cut
from tetrabz import trilinear_interpolation_indices

__all__ = [
    "__version__",
    "IntegrationMesh",
    "SimplexCut",
    "build_integration_mesh",
    "fermieng",
    "occ",
    "occupation_weights",
    "simplex_affine_coefficients",
    "solve_fermi_energy",
    "small_tetrahedron_cut",
    "tetrahedron_offsets",
    "tetrahedron_weight_matrix",
    "triangle_cut",
    "trilinear_interpolation_indices",
]
