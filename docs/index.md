# bztetra

`bztetra` is a Python + Numba package for tetrahedron integration on regular
k-grids, aimed at electronic-structure users who want physically meaningful
Brillouin-zone weights without going through the legacy `libtetrabz` wrapper.
The top-level API is 3D-only, while the parallel 2D linear triangle-method
surface now lives under `bztetra.twod`; see
[2D Triangle Method](two_dimensional_plan.md).

!!! warning "Public preview status"

    `bztetra` is in its public `0.x` preview series. Validate important
    production calculations against the original
    [`libtetrabz`](https://github.com/mitsuaki1987/libtetrabz) implementation
    or the parity checks documented in [Validation](validation.md) while the
    API and validation envelope continue to mature.

<div class="grid cards" markdown>

-   __Physics Guide__

    ---

    What each routine computes, with the key formulas in display math.

    [Open the physics guide](physics.md)

-   __Worked Examples__

    ---

    Minimal examples with actual output plots for 3D and 2D DOS and response functions.

    [See the examples](examples.md)

-   __2D Triangle Method__

    ---

    Use `bztetra.twod` for genuinely 2D occupations, DOS, and response work.

    [Open the 2D guide](two_dimensional_plan.md)

-   __Quickstart__

    ---

    The array conventions, output shapes, and first correct calls.

    [Open quickstart](quickstart.md)

-   __Validation__

    ---

    Legacy shell parity, wrapper parity, analytic checks, and exact commands.

    [See validation](validation.md)

</div>

## What It Computes

The public routines cover the usual tetrahedron-method objects a condensed-
matter user cares about:

\[
\Theta(E_F - \varepsilon_n(\mathbf{k})),
\qquad
\delta(E - \varepsilon_n(\mathbf{k})),
\qquad
\chi_0(\mathbf{q}),
\qquad
\Pi_0(\mathbf{q}, z).
\]

The package returns the corresponding **k-resolved weights**, so you can still
apply matrix elements or projectors before the final Brillouin-zone sum.

## Start Here

If you want the docs in physicist order rather than package order:

1. Read [Physics Guide](physics.md) to see the actual quantities and formulas.
2. Read [Examples](examples.md) to see minimal scripts and output plots.
3. Use [Quickstart](quickstart.md) only to check array conventions and shapes.
4. Use [API Reference](api.md) when you need signatures, not when you need the
   physical idea first.
5. Read [2D Triangle Method](two_dimensional_plan.md) before attempting any
   genuinely 2D use case.

## Original Project

`bztetra` is a clean-room Python port informed by the original
[`libtetrabz`](https://github.com/mitsuaki1987/libtetrabz) project by Mitsuaki
Kawamura and collaborators. If you use `bztetra` in research, also acknowledge
the original implementation and cite the method paper linked from
[Validation](validation.md#original-project).
