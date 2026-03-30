"""Compatibility shim while the package name transitions to ``tetrabz``."""

from tetrabz import __version__
from tetrabz import build_integration_mesh
from tetrabz import dblstep
from tetrabz import dbldelta
from tetrabz import density_of_states_weights
from tetrabz import dos
from tetrabz import double_delta_weights
from tetrabz import double_step_weights
from tetrabz import fermieng
from tetrabz import integrated_density_of_states_weights
from tetrabz import IntegrationMesh
from tetrabz import intdos
from tetrabz import occ
from tetrabz import occupation_weights
from tetrabz import polstat
from tetrabz import SimplexCut
from tetrabz import simplex_affine_coefficients
from tetrabz import small_tetrahedron_cut
from tetrabz import solve_fermi_energy
from tetrabz import static_polarization_weights
from tetrabz import tetrahedron_offsets
from tetrabz import tetrahedron_weight_matrix
from tetrabz import triangle_cut
from tetrabz import trilinear_interpolation_indices

__all__ = [
    "__version__",
    "IntegrationMesh",
    "SimplexCut",
    "build_integration_mesh",
    "dblstep",
    "dbldelta",
    "density_of_states_weights",
    "dos",
    "double_delta_weights",
    "double_step_weights",
    "fermieng",
    "integrated_density_of_states_weights",
    "intdos",
    "occ",
    "occupation_weights",
    "polstat",
    "simplex_affine_coefficients",
    "small_tetrahedron_cut",
    "solve_fermi_energy",
    "static_polarization_weights",
    "tetrahedron_offsets",
    "tetrahedron_weight_matrix",
    "triangle_cut",
    "trilinear_interpolation_indices",
]
