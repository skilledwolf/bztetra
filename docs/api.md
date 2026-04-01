# API Reference

`bztetra` keeps a deliberately small public computational surface. The
functions below are grouped by workflow rather than by source file.

## Occupation And Fermi Level

::: bztetra.FermiEnergySolution

::: bztetra.occupation_weights

::: bztetra.solve_fermi_energy

## DOS

::: bztetra.density_of_states_weights

::: bztetra.integrated_density_of_states_weights

## 2D Triangle Method

`bztetra.twod` mirrors the 3D public vocabulary where the physics matches, but
with 2D shapes and the linear triangle method only.

::: bztetra.twod.FermiEnergySolution

::: bztetra.twod.occupation_weights

::: bztetra.twod.solve_fermi_energy

::: bztetra.twod.density_of_states_weights

::: bztetra.twod.integrated_density_of_states_weights

::: bztetra.twod.PreparedResponseEvaluator

::: bztetra.twod.prepare_response_evaluator

::: bztetra.twod.phase_space_overlap_weights

::: bztetra.twod.nesting_function_weights

::: bztetra.twod.static_polarization_weights

::: bztetra.twod.static_polarization_observables

::: bztetra.twod.fermi_golden_rule_weights

::: bztetra.twod.fermi_golden_rule_observables

::: bztetra.twod.complex_frequency_polarization_weights

::: bztetra.twod.complex_frequency_polarization_observables

::: bztetra.twod.retarded_response_observables

::: bztetra.twod.TriangleIntegrationMesh

::: bztetra.twod.build_integration_mesh

::: bztetra.twod.bilinear_interpolation_indices

## Response Setup

::: bztetra.PreparedResponseEvaluator

::: bztetra.prepare_response_evaluator

## Static Response

::: bztetra.phase_space_overlap_weights

::: bztetra.nesting_function_weights

::: bztetra.static_polarization_weights

## Frequency-Dependent Response

::: bztetra.fermi_golden_rule_weights

::: bztetra.complex_frequency_polarization_weights

## Causality And Reconstruction

::: bztetra.reconstruct_retarded_response

::: bztetra.RetardedResponse

::: bztetra.KramersKronigDiagnostics

## Advanced Helpers

These helpers expose the tetrahedron geometry and scalar constructions used by
the higher-level routines. They are useful for method work, debugging, and
tests, but most users should start with the routines above.

::: bztetra.IntegrationMesh

::: bztetra.build_integration_mesh

::: bztetra.tetrahedron_weight_matrix

::: bztetra.tetrahedron_offsets

::: bztetra.trilinear_interpolation_indices

::: bztetra.SimplexCut

::: bztetra.small_tetrahedron_cut

::: bztetra.triangle_cut

::: bztetra.simplex_affine_coefficients
