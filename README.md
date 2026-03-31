# bztetra

`bztetra` is a Python + Numba package for single-process tetrahedron
integration on regular k-grids.

It is aimed at users who already have band energies on a regular mesh and want
k-resolved weights for occupations, DOS, and Lindhard-style response functions
without going through the legacy `libtetrabz` wrapper.

The top-level `bztetra` API remains strictly 3D. A parallel 2D linear
triangle-method implementation now lives under `bztetra.twod` for the
occupation, DOS, and response families. Do not fake a flat third axis with
`nz=1`; see
[docs/two_dimensional_plan.md](docs/two_dimensional_plan.md).

> [!IMPORTANT]
> `bztetra` is a public preview in the `0.x` series. The package is usable, but
> the API and validation envelope may still evolve before `1.0`. Validate
> important production results against the original
> [`libtetrabz`](https://github.com/mitsuaki1987/libtetrabz) implementation or
> run the parity checks described in [docs/validation.md](docs/validation.md).

- NumPy arrays in, NumPy arrays out.
- Optimized and legacy-linear tetrahedron schemes.
- Validation against legacy shell outputs, the legacy Python wrapper, and
  analytic free-electron reference cases.

## Install

Requires Python 3.11+.

```bash
pip install bztetra
```

Project docs live at [skilledwolf.github.io/bztetra](https://skilledwolf.github.io/bztetra/).

Plotting examples use Matplotlib:

```bash
pip install "bztetra[plot]"
```

From a source checkout:

```bash
pip install .
```

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

`weights` has shape `(nenergy, nx, ny, nz, nbands)`. For band-resolved DOS,
sum only over the k-grid axes.

## Core Conventions

- `reciprocal_vectors` is a `(3, 3)` matrix with reciprocal basis vectors in
  columns.
- `eigenvalues` always has shape `(nx, ny, nz, nbands)`.
- `weight_grid_shape` is optional. Leave it unset unless you intentionally want
  weights interpolated onto another regular grid.
- Response outputs use the last axes `(target_band, source_band)`.
- To integrate over the Brillouin zone, sum over the explicit k-grid axes and
  multiply by `np.linalg.det(bvec)` for a right-handed basis.

## Choose The Right Routine

| Need | Call |
| --- | --- |
| Occupations at a known Fermi level | `occupation_weights` |
| Fermi level search plus occupations | `solve_fermi_energy` |
| DOS or integrated DOS on a sampled energy grid | `density_of_states_weights`, `integrated_density_of_states_weights` |
| Static overlap, nesting, or polarization between two band manifolds | `phase_space_overlap_weights`, `nesting_function_weights`, `static_polarization_weights` |
| Repeated real- or complex-frequency response sweeps | `prepare_response_evaluator`, then `fermi_golden_rule_weights` or `complex_frequency_polarization_weights` |

See [docs/physics.md](docs/physics.md) for the key formulas and
[docs/examples.md](docs/examples.md) for worked examples with output plots.

## 2D Triangle Method

For genuinely 2D problems, switch namespaces and use `bztetra.twod`.
`bztetra.twod` keeps the same physics-facing routine names for occupations,
DOS, overlap, Lindhard-style polarization, and real- or complex-frequency
response, but on `(nx, ny, nbands)` arrays with Brillouin-zone **area**
normalization and the linear triangle method.

Start with [docs/two_dimensional_plan.md](docs/two_dimensional_plan.md) for
the current 2D surface, then run one of the plot-backed example scripts:

- [examples/plot_twod_square_lattice_dos.py](examples/plot_twod_square_lattice_dos.py)
- [examples/plot_twod_phase_space_overlap.py](examples/plot_twod_phase_space_overlap.py)
- [examples/plot_twod_lindhard.py](examples/plot_twod_lindhard.py)

## Examples

- [docs/quickstart.md](docs/quickstart.md): the shortest path from eigenvalues
  on a regular grid to a correct `bztetra` call.
- [examples/plot_tight_binding_dos.py](examples/plot_tight_binding_dos.py):
  cubic tight-binding DOS figure.
- [examples/review_occupancy.py](examples/review_occupancy.py): occupation
  weights and Fermi-level search on a small reference problem.
- [examples/plot_lindhard.py](examples/plot_lindhard.py): static Lindhard
  response figure.
- [examples/plot_twod_square_lattice_dos.py](examples/plot_twod_square_lattice_dos.py):
  2D square-lattice DOS and integrated DOS via `bztetra.twod`.
- [examples/plot_twod_phase_space_overlap.py](examples/plot_twod_phase_space_overlap.py):
  2D free-electron `dblstep` phase-space overlap with the exact circular-segment curve.
- [examples/plot_twod_lindhard.py](examples/plot_twod_lindhard.py):
  2D free-electron static Lindhard plateau and Kohn cusp.

## Validation

`bztetra` is checked against tracked legacy shell/example fixtures, optional
direct parity with the legacy `libtetrabz` Python wrapper, and analytic
reference cases. See
[docs/validation.md](docs/validation.md) for the exact coverage and
reproduction commands.

## Acknowledgement

`bztetra` is a clean-room Python + Numba port informed by the original
[`libtetrabz`](https://github.com/mitsuaki1987/libtetrabz) project by Mitsuaki
Kawamura and collaborators. If `bztetra` is useful in research, users should
also acknowledge the original method and implementation:

- M. Kawamura, Y. Gohda, and S. Tsuneyuki, "Improved tetrahedron method for the
  Brillouin-zone integration applicable to response functions,"
  [Phys. Rev. B 89, 094515 (2014)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.89.094515)
- Original repository:
  [github.com/mitsuaki1987/libtetrabz](https://github.com/mitsuaki1987/libtetrabz)

## Development

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
python -m pytest -q
```

For local docs work:

```bash
pip install -e '.[dev,docs]'
mkdocs serve
```

Benchmark entry points live under `benchmarks/`:

```bash
PYTHONPATH=src python benchmarks/benchmark_hotpaths.py
PYTHONPATH=src python benchmarks/benchmark_twod_hotpaths.py --profile --profile-task complex_frequency_polarization_weights
PYTHONPATH=src python benchmarks/benchmark_twod_response_multiband.py --profile --profile-task all
```

Package versions are derived from git tags at build and install time.
