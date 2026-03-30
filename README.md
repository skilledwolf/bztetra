# tetrabz

`tetrabz` is a clean-room Python and Numba port of the legacy `libtetrabz`
tetrahedron-integration library. The root package is being built around a
modern Python API, reproducible validation fixtures, and performance-oriented
array layouts instead of a literal line-by-line Fortran translation.

The legacy source tree remains in `libtetra_original/` for reference and is
intentionally ignored by the root git repository.

The implementation strategy is:

- NumPy for array orchestration and test/reference helpers.
- Numba for hot scalar and tensor kernels once the numerical contracts are
  locked down.
- SciPy only where it materially helps validation, reference calculations, or
  tooling; it is not a planned dependency for the core runtime path right now.
