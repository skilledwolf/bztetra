# 2D Triangle Method

`bztetra.twod` is the package's genuine 2D public surface. The top-level
`bztetra` API still supports **3D reciprocal grids only**, and that is
deliberate: the original implementation is built around tetrahedra, trilinear
interpolation, and Brillouin-zone **volume** integration.

The current 2D linear triangle-method surface lives under `bztetra.twod`:

- regular-grid 2D geometry and triangle decomposition,
- bilinear interpolation onto a distinct output grid,
- `occupation_weights`,
- `solve_fermi_energy`,
- `density_of_states_weights`,
- `integrated_density_of_states_weights`,
- `phase_space_overlap_weights`,
- `nesting_function_weights`,
- `static_polarization_weights`,
- `fermi_golden_rule_weights`,
- `complex_frequency_polarization_weights`,
- `prepare_response_evaluator`.

This page documents what is already available and what still remains open. The
main unresolved 2D question is whether the package needs an improved/optimized
triangle-weight correction beyond the current linear kernels.

## What Not To Do

Do **not** try to treat a 2D problem as a fake `(nx, ny, 1)` 3D problem.

That shortcut is wrong for three separate reasons:

1. The measure is wrong: a 2D Brillouin-zone integral uses **area**, not
   volume.
2. The cell decomposition is wrong: 2D cells split into triangles, not
   tetrahedra.
3. The interpolation path is wrong: 2D remapping needs bilinear stencils, not
   the current trilinear machinery.

## Recommended Strategy

2D support should remain a **separate triangle-method path** that lives beside
the current 3D implementation rather than inside it behind shape-dependent
branches.

The current design is:

- keep the current package behavior explicit and 3D-only;
- add a dedicated 2D mesh and triangle decomposition;
- keep the public routine vocabulary aligned where the physics matches;
- avoid mixed 2D/3D Numba hot kernels.

## Capability Assessment

The 2D work splits naturally into a straightforward part and a derivation-heavy
part.

### Implemented In The Current Tree

These pieces are now live in `bztetra.twod`:

- 2D regular-grid geometry and indexing on `(nx, ny)` meshes
- triangle decomposition of each 2D cell
- bilinear interpolation onto a distinct output grid
- `occupation_weights`
- `solve_fermi_energy`
- `density_of_states_weights`
- `integrated_density_of_states_weights`
- `phase_space_overlap_weights`
- `nesting_function_weights`
- `static_polarization_weights`
- `fermi_golden_rule_weights`
- `complex_frequency_polarization_weights`
- `prepare_response_evaluator`
- analytic checks such as constant 2D free-electron DOS
- a plot-first 2D square-lattice DOS review example
- exact single-triangle regression checks for the response-family formulas
- plot-first 2D free-electron overlap and Lindhard review examples

### The Expert Derivation Was Needed For

The response-family routines were not implemented from intuition alone. They
needed a clean 2D triangle-method derivation and edge-case strategy first:

- `phase_space_overlap_weights`
- `nesting_function_weights`
- `static_polarization_weights`
- `fermi_golden_rule_weights`
- `complex_frequency_polarization_weights`

That derivation is now folded into the implementation. The remaining caution
areas are optimized-correction design, public documentation of the `dblstep`
semantics, and broader plot-first review coverage for the dynamic kernels.

## Next 2D Steps

1. Decide whether the 2D path should remain linear-only or gain an
   improved/optimized triangle-weight scheme.
2. Add more 2D review cases for the frequency-dependent kernels if we want the
   same plot coverage depth as the 3D path.
3. Keep the exact single-triangle regression cases in sync with any future 2D
   optimization work.

## Expert Handoff Prompt

If a stronger computational-physics model is available, this is the prompt I
used before implementing the 2D response family.

```text
You are helping design a true 2D triangle-method extension for an existing
Python + Numba Brillouin-zone integration package called `bztetra`.

Context:
- The current package is 3D-only and implements the modernized equivalents of
  libtetrabz’s public routine surface:
  occupation_weights
  solve_fermi_energy
  density_of_states_weights
  integrated_density_of_states_weights
  phase_space_overlap_weights
  nesting_function_weights
  static_polarization_weights
  fermi_golden_rule_weights
  complex_frequency_polarization_weights
- Public inputs are regular-grid band energies on arrays shaped like:
  3D single-spectrum: (nx, ny, nz, nbands)
  3D two-spectrum: occupied and target arrays each shaped (nx, ny, nz, nbands)
- Public outputs are k-resolved weights on a regular output grid, not already
  integrated observables.
- The current 3D code uses:
  - tetrahedral decomposition of each cell,
  - trilinear interpolation when output and energy grids differ,
  - Numba hot kernels,
  - explicit k-resolved weights that are summed by the caller with det(bvec).
- For 2D we do NOT want to fake things with nz=1. We want a real triangle
  method with Brillouin-zone area normalization.

What I need from you:
1. A mathematically clean specification for a 2D regular-grid triangle method
   that is structurally analogous to the current 3D package but genuinely 2D.
2. A recommended decomposition of each rectangular 2D cell into triangles,
   including any preferred orientation rules and periodic-grid indexing notes.
3. The per-triangle piecewise formulas for k-resolved weights, suitable for
   distribution to triangle vertices or local interpolation points, for the
   following routines:
   - occupation_weights
   - density_of_states_weights
   - integrated_density_of_states_weights
   - phase_space_overlap_weights
   - nesting_function_weights
   - static_polarization_weights
   - fermi_golden_rule_weights
   - complex_frequency_polarization_weights
4. For each routine, I need the formulas written in a way that can be turned
   directly into fixed-size Numba kernels:
   - assume one triangle with sorted vertex energies
   - identify all piecewise cases
   - give explicit scalar prefactors
   - describe the affine/barycentric coefficient matrices needed to map local
     triangle contributions back to vertex weights
5. I need careful handling guidance for numerically delicate cases:
   - equal or nearly equal triangle-vertex energies
   - zero or nearly zero energy denominators
   - grazing cuts at a triangle edge or vertex
   - delta-function support collapsing to a point or line segment
   - static-polarization and complex-frequency degeneracies
6. I need validation guidance:
   - exact or highly trusted analytic checks for 2D free electrons
   - known 2D Lindhard / nesting behaviors that should appear in plots
   - minimal regression cases that would catch sign errors or wrong prefactors
7. Please separate what is fully standard/known from what is your own
   recommended interpretation, and call out any places where multiple
   conventions are possible.

Desired output format:
- Section 1: recommended overall 2D API/shape conventions
- Section 2: geometry / interpolation design
- Section 3: occupation + DOS family formulas
- Section 4: response-family formulas
- Section 5: degeneracy handling and numerical-stability rules
- Section 6: validation plan
- Section 7: implementation-oriented pseudocode sketches for each kernel

Important:
- The answer must be self-contained. Do not assume access to the current repo.
- Optimize for implementation readiness in Python + Numba, not symbolic
  elegance.
- If a routine is too risky to specify exactly without a literature cross-check,
  say so explicitly and give the safest next step.
```
