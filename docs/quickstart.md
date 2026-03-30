# bztetra Quickstart

This is the shortest path from "I have band energies on a regular k-grid" to a
correct `bztetra` call.

Until the first full public release, validate important production calculations
against the original
[`libtetrabz`](https://github.com/mitsuaki1987/libtetrabz) implementation or
the parity checks in [validation.md](validation.md).

The top-level `bztetra` routines on this page are strictly 3D. For 2D
occupation, DOS, or response work, use `bztetra.twod`. Do not fake a flat
third axis with `nz=1`; see [2D Triangle Method](two_dimensional_plan.md).

## Four Things To Know

1. `reciprocal_vectors` is a `(3, 3)` array with reciprocal basis vectors in
   columns.
2. Every eigenvalue input uses shape `(nx, ny, nz, nbands)`.
3. `weight_grid_shape` is optional and controls the output grid. Leave it unset
   unless you explicitly want the final weights on another regular grid.
4. Brillouin-zone integrals come from summing over the k-grid axes and
   multiplying by `np.linalg.det(bvec)` for a right-handed basis.

`method="optimized"` is the default and the right choice unless you are
reproducing legacy linear-tetrahedron results.

If you want the physics before the call signatures, read
[Physics Guide](physics.md) first.

## Which Function Should I Call?

| Task | Function | Return shape |
| --- | --- | --- |
| Occupations at a fixed Fermi level | `occupation_weights` | `(wx, wy, wz, nbands)` |
| Fermi level search plus occupations | `solve_fermi_energy` | `FermiEnergySolution` |
| DOS at sampled energies | `density_of_states_weights` | `(nenergy, wx, wy, wz, nbands)` |
| Integrated DOS at sampled energies | `integrated_density_of_states_weights` | `(nenergy, wx, wy, wz, nbands)` |
| Static overlap between source and target manifolds | `phase_space_overlap_weights` | `(wx, wy, wz, ntarget, nsource)` |
| Static nesting function | `nesting_function_weights` | `(wx, wy, wz, ntarget, nsource)` |
| Static polarization | `static_polarization_weights` | `(wx, wy, wz, ntarget, nsource)` |
| Real-frequency response | `fermi_golden_rule_weights` | `(nenergy, wx, wy, wz, ntarget, nsource)` |
| Complex-frequency response | `complex_frequency_polarization_weights` | `(nenergy, wx, wy, wz, ntarget, nsource)` |
| Many response sweeps on the same band pair | `prepare_response_evaluator` | `PreparedResponseEvaluator` |

`(wx, wy, wz)` is `weight_grid_shape` when you pass it, otherwise it is the
same as the eigenvalue grid `(nx, ny, nz)`.

## First Calculation

```python
import numpy as np
from bztetra import density_of_states_weights

bvec = 2.0 * np.pi * np.eye(3)
nx = ny = nz = 16
energies = np.linspace(-1.0, 1.0, 200, dtype=np.float64)

eigenvalues = np.empty((nx, ny, nz, 1), dtype=np.float64)
for ix in range(nx):
    for iy in range(ny):
        for iz in range(nz):
            kfrac = np.array([ix / nx, iy / ny, iz / nz], dtype=np.float64) - 0.5
            kcart = bvec @ kfrac
            eigenvalues[ix, iy, iz, 0] = 0.5 * np.dot(kcart, kcart)

weights = density_of_states_weights(bvec, eigenvalues, energies)
total_dos = weights.sum(axis=(1, 2, 3, 4)) * np.linalg.det(bvec)
```

The DOS routines return k-resolved weights, not already-summed curves. That is
intentional: it lets you apply additional matrix elements or observables before
integrating.

## Repeated Response Sweeps

```python
import numpy as np
from bztetra import prepare_response_evaluator

bvec = 2.0 * np.pi * np.eye(3)
occupied = np.random.default_rng(0).normal(size=(16, 16, 16, 3))
target = np.random.default_rng(1).normal(size=(16, 16, 16, 5))

problem = prepare_response_evaluator(bvec, occupied, target)

static = problem.static_polarization_weights()
real_freq = problem.fermi_golden_rule_weights(np.linspace(0.0, 3.0, 64))
imag_freq = problem.complex_frequency_polarization_weights(1j * np.linspace(0.2, 3.0, 32))
```

## Two Gotchas

- The first call to a Numba-backed routine includes JIT compilation time.
  Time the second call if you care about steady-state performance.
- Response outputs are ordered `[..., target_band, source_band]`.

## Useful Examples

See [Examples](examples.md) for the repository scripts to run next, including
the recommended DOS, occupation, and Lindhard entry points.
