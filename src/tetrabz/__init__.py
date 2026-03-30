"""Modern Python port of the legacy libtetrabz tetrahedron routines."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

from .dos import density_of_states_weights
from .dos import integrated_density_of_states_weights
from .occupancy import FermiEnergySolution
from .response import complex_frequency_polarization_weights
from .response import fermi_golden_rule_weights
from .response import nesting_function_weights
from .response import phase_space_overlap_weights
from .response import prepare_response_evaluator
from .response import PreparedResponseEvaluator
from .response import static_polarization_weights
from .formulas import SimplexCut
from .geometry import IntegrationMesh
from .geometry import build_integration_mesh
from .occupancy import occupation_weights
from .occupancy import solve_fermi_energy
from .formulas import simplex_affine_coefficients
from .formulas import small_tetrahedron_cut
from .geometry import tetrahedron_offsets
from .geometry import tetrahedron_weight_matrix
from .formulas import triangle_cut
from .geometry import trilinear_interpolation_indices

try:
    __version__ = package_version("tetrabz")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "complex_frequency_polarization_weights",
    "density_of_states_weights",
    "FermiEnergySolution",
    "fermi_golden_rule_weights",
    "integrated_density_of_states_weights",
    "IntegrationMesh",
    "nesting_function_weights",
    "occupation_weights",
    "phase_space_overlap_weights",
    "PreparedResponseEvaluator",
    "prepare_response_evaluator",
    "SimplexCut",
    "solve_fermi_energy",
    "static_polarization_weights",
    "__version__",
    "build_integration_mesh",
    "simplex_affine_coefficients",
    "small_tetrahedron_cut",
    "tetrahedron_offsets",
    "tetrahedron_weight_matrix",
    "triangle_cut",
    "trilinear_interpolation_indices",
]
