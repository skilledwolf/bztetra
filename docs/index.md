# bztetra

`bztetra` is a Python + Numba package for tetrahedron integration on regular
k-grids.

It is designed for users who already have eigenvalues on a structured mesh and
want k-resolved weights for occupations, DOS, and Lindhard-style response
functions without going through the legacy `libtetrabz` wrapper.

!!! warning "Pre-release status"

    `bztetra` is still pre-release. Until the first full public release,
    validate important production calculations against the original
    [`libtetrabz`](https://github.com/mitsuaki1987/libtetrabz) implementation
    or the parity checks documented in [Validation](validation.md).

<div class="grid cards" markdown>

-   __Quickstart__

    ---

    Array conventions, a first calculation, and the repeated-response workflow.

    [Open quickstart](quickstart.md)

-   __API Reference__

    ---

    The stable public surface, rendered from the package docstrings.

    [Browse the API](api.md)

-   __Examples__

    ---

    Which example script to run for DOS, occupations, Lindhard response, and
    more.

    [See examples](examples.md)

-   __Validation__

    ---

    Legacy shell parity, wrapper parity, analytic checks, and exact commands.

    [See validation](validation.md)

</div>

## What bztetra Gives You

- NumPy arrays in, NumPy arrays out.
- Optimized and legacy-linear tetrahedron schemes.
- A band-last public API with shape contracts written in Python terms.
- Validation against the original shell references, the legacy Python wrapper,
  and analytic free-electron cases.

## The Four Rules That Matter

1. `reciprocal_vectors` is a `(3, 3)` matrix with reciprocal basis vectors in
   columns.
2. Eigenvalue inputs always use shape `(nx, ny, nz, nbands)`.
3. `weight_grid_shape` changes the output grid, not the grid on which your
   eigenvalues were sampled.
4. Response outputs end in `(target_band, source_band)`, and
   frequency-dependent response adds a leading energy axis.

## Original Project

`bztetra` is a clean-room Python port informed by the original
[`libtetrabz`](https://github.com/mitsuaki1987/libtetrabz) project by Mitsuaki
Kawamura and collaborators. If you use `bztetra` in research, also acknowledge
the original implementation and cite the method paper linked from
[Validation](validation.md#original-project).
