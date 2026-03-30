# tetrabz

`tetrabz` is a clean-room Python and Numba port of the legacy `libtetrabz`
tetrahedron-integration library. The root package is being built around a
modern Python API, reproducible validation fixtures, and performance-oriented
array layouts instead of a literal line-by-line Fortran translation.

The legacy source tree remains in `libtetra_original/` for reference and is
intentionally ignored by the root git repository.

Package versions are derived from git tags at build/install time. Untagged
checkouts use a `0.0.0` fallback base, which Hatch VCS expands into a
development version until the first release tag is created.

The implementation strategy is:

- NumPy for array orchestration and test/reference helpers.
- Numba for hot scalar and tensor kernels once the numerical contracts are
  locked down. The `occupation_weights`,
  `density_of_states_weights`, `integrated_density_of_states_weights`,
  `phase_space_overlap_weights`, `nesting_function_weights`,
  `static_polarization_weights`, `fermi_golden_rule_weights`, and
  `complex_frequency_polarization_weights` paths are now compiled.
- Shared setup is also optimized now: direct repeated calls reuse cached mesh
  geometry internally, and tetrahedron interpolation is compiled instead of
  running in Python loops.
- Multiband dynamic-response kernels now parallelize over the band axes, with
  larger pair-count workloads using a higher-parallelism band-pair path for
  `fermi_golden_rule_weights` and
  `complex_frequency_polarization_weights`.
- A prepared response evaluator for repeated sweeps on fixed bands, so
  real- and complex-frequency response users can reuse mesh and tetrahedron
  setup instead of rebuilding it for every energy grid.
- SciPy only where it materially helps validation, reference calculations, or
  tooling; it is not a planned dependency for the core runtime path right now.

## Dev Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
python -m pytest -q
```

For plot-only installs without the rest of the dev tooling, use
`pip install -e '.[plot]'`.

For a quick human-review snapshot of the current geometry/indexing layer, the
shared cut-formula helpers, and one occupation-weight result, run:

```bash
.venv/bin/python examples/review_geometry_and_cuts.py
```

For a focused occupation / Fermi-search review against the legacy 8x8 toy
system, run:

```bash
.venv/bin/python examples/review_occupancy.py
```

For a physically meaningful DOS review plot, reproduce the legacy cubic
tight-binding example and write a figure under `build/review_plots/`:

```bash
.venv/bin/python examples/plot_tight_binding_dos.py
```

For a physically meaningful static-polarization review plot, reproduce the
legacy 3D free-electron Lindhard example and write a figure under
`build/review_plots/`:

```bash
.venv/bin/python examples/plot_lindhard.py
```

For a physically meaningful real-frequency review plot, reproduce the
free-electron Fermi-golden-rule toy model with the legacy `k^2` matrix element
and write a figure under `build/review_plots/`:

```bash
.venv/bin/python examples/plot_fermi_golden_rule.py
```

For a physically meaningful complex-frequency review plot, reproduce the
free-electron interband polarization on the positive Matsubara axis with exact
comparison channels and write a figure under `build/review_plots/`:

```bash
.venv/bin/python examples/plot_complex_frequency_polarization.py
```

For a physically meaningful phase-space review plot, compare the free-
electron phase-space-overlap and nesting-function sweeps against their exact
overlap / nesting curves and write a two-panel figure under
`build/review_plots/`:

```bash
.venv/bin/python examples/plot_phase_space_and_nesting.py
```

For a numeric DOS / integrated-DOS review that compares the current 8x8
free-electron fixture against the analytic continuum target, run:

```bash
.venv/bin/python examples/review_dos.py
```

For a quick local timing baseline on the current hot paths
(`occupation_weights`, `density_of_states_weights`,
`integrated_density_of_states_weights`, `phase_space_overlap_weights`,
`nesting_function_weights`, `static_polarization_weights`,
`fermi_golden_rule_weights`, `complex_frequency_polarization_weights`, and
their prepared-response counterparts), run:

```bash
PYTHONPATH=src .venv/bin/python benchmarks/benchmark_hotpaths.py
```

For repeated response sweeps on a fixed pair of band manifolds, prepare the
mesh/tetrahedra once and reuse them explicitly. Warm direct calls already
reuse the mesh cache internally, but this API also skips repeated tetrahedron
interpolation:

```python
from tetrabz import prepare_response_evaluator

response = prepare_response_evaluator(bvec, occupied_bands, target_bands, weight_grid_shape=(16, 16, 16))
weights_real = response.fermi_golden_rule_weights(sample_energies)
weights_complex = response.complex_frequency_polarization_weights(1j * matsubara_frequencies)
```

For a more focused static-polarization timing sweep that isolates the multiband free-
electron case, the Lindhard small-`q` / `2k_F` branches, and the interpolation
path, run:

```bash
PYTHONPATH=src .venv/bin/python benchmarks/benchmark_static_polarization.py --grid 16 --weight-grid 8 --q-values 0.125,2.0
```

For a focused multiband frequency-response benchmark that compares direct and
prepared real- and complex-frequency polarization sweeps, run:

```bash
PYTHONPATH=src .venv/bin/python benchmarks/benchmark_response_multiband.py --grid 16 --bands 6 --energy-count 16
```

For a routine-by-routine comparison against the legacy `libtetrabz` Python
wrapper, first install the local legacy package into `.venv`:

```bash
.venv/bin/pip install -e ./libtetra_original/python
```

Then run:

```bash
PYTHONPATH=src .venv/bin/python benchmarks/benchmark_compare_libtetrabz.py --grid 8 --repeats 3
```

For the full release-parity gate against both the legacy shell references
(`test2_8_8`, `test2_16_8`, `test2_16_16`) and the installed
`libtetrabz` Python wrapper, run:

```bash
.venv/bin/pip install -e ./libtetra_original/python
.venv/bin/python -m pytest -q tests/test_legacy_shell_matrix.py tests/test_legacy_wrapper_parity.py
```

GitHub Actions now runs that gate automatically on Ubuntu in addition to the
normal lint and pytest jobs.

The public API now uses one descriptive name per routine. Each public
function docstring names the corresponding legacy `libtetrabz_*` routine it
replaces.
