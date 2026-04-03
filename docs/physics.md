# Physics Guide

This page is the physics-facing map of the package: what each routine computes,
how the returned weights relate to standard Brillouin-zone formulas, and how to
turn those weights into observables.

Most formulas on this page are written in the 3D tetrahedron notation for
compactness. The current 2D triangle implementation mirrors the same response
objects with Brillouin-zone area normalization instead of 3D volume
normalization.

## One Core Idea

`bztetra` does **not** try to hide the k-resolved weight field from you. The
public routines return tetrahedron-integrated weights on a regular output grid,
so you can still apply matrix elements, orbital projectors, or other
k-dependent factors before the final Brillouin-zone sum.

For a right-handed reciprocal basis \(bvec\), the final integral is obtained by
summing over the explicit k-grid axes and multiplying by
\(\det(bvec)\).

## Occupation And Fermi Level

At a fixed Fermi level, `occupation_weights` approximates the occupation field

\[
n_n(\mathbf{k}; E_F) = \Theta\!\left(E_F - \varepsilon_n(\mathbf{k})\right),
\]

where \(\varepsilon_n(\mathbf{k})\) is the band energy for band \(n\).

`solve_fermi_energy` finds the value of \(E_F\) such that

\[
\sum_n \int_{\mathrm{BZ}} n_n(\mathbf{k}; E_F)\, d^3k
= N_{\mathrm{e}, \uparrow},
\]

with `electrons_per_spin = N_{e,\uparrow}` in the package API.

## Density Of States

`density_of_states_weights` approximates the k-resolved DOS weight field for

\[
D(E) = \sum_n \int_{\mathrm{BZ}}
\delta\!\left(E - \varepsilon_n(\mathbf{k})\right)\, d^3k,
\]

and `integrated_density_of_states_weights` approximates

\[
N(E) = \sum_n \int_{\mathrm{BZ}}
\Theta\!\left(E - \varepsilon_n(\mathbf{k})\right)\, d^3k.
\]

The returned arrays keep the k-grid explicit:

- `density_of_states_weights` returns `(nenergy, wx, wy, wz, nbands)`
- `integrated_density_of_states_weights` returns
  `(nenergy, wx, wy, wz, nbands)`

To obtain a scalar DOS curve, sum over the k-grid and band axes and multiply by
\(\det(bvec)\).

## Static Response Quantities

For two band manifolds, `bztetra` exposes three useful static objects.

### Phase-Space Overlap

`phase_space_overlap_weights` follows the reference `dblstep` semantics: occupied
phase space filtered by a second Heaviside on the source-minus-target energy
difference

\[
W_{\mathrm{overlap}}(\mathbf{q}) =
\sum_{nm} \int_{\mathrm{BZ}}
\Theta\!\left(-\varepsilon_n(\mathbf{k})\right)
\Theta\!\left(\varepsilon_n(\mathbf{k}) - \varepsilon_m(\mathbf{k}+\mathbf{q})\right)
\, d^3k,
\]

with energies typically supplied relative to the Fermi level.

### Nesting Function

`nesting_function_weights` approximates the Fermi-surface intersection weight

\[
W_{\mathrm{nest}}(\mathbf{q}) =
\sum_{nm} \int_{\mathrm{BZ}}
\delta\!\left(\varepsilon_n(\mathbf{k})\right)
\delta\!\left(\varepsilon_m(\mathbf{k}+\mathbf{q})\right)
\, d^3k.
\]

### Static Polarization

`static_polarization_weights` corresponds to the zero-frequency polarization

\[
\chi_0(\mathbf{q}) =
\sum_{nm} \int_{\mathrm{BZ}}
\frac{
f\!\left(\varepsilon_n(\mathbf{k})\right)
- f\!\left(\varepsilon_m(\mathbf{k}+\mathbf{q})\right)
}{
\varepsilon_n(\mathbf{k}) - \varepsilon_m(\mathbf{k}+\mathbf{q})
}
\, d^3k.
\]

For simple symmetric free-electron review plots, the full Lindhard curve is
the sum of the occupied-to-empty and empty-to-occupied channels, so examples
at nonzero \(q\) often multiply the kernel by 2.

All three static response routines return arrays of shape

\[
(wx, wy, wz, n_{\mathrm{target}}, n_{\mathrm{source}}),
\]

with the final two axes ordered `(target_band, source_band)`.

## Frequency-Dependent Response

`fermi_golden_rule_weights` approximates the real-frequency transition weight

\[
S(\mathbf{q}, \omega) =
\sum_{nm} \int_{\mathrm{BZ}}
\left[
f\!\left(\varepsilon_n(\mathbf{k})\right)
- f\!\left(\varepsilon_m(\mathbf{k}+\mathbf{q})\right)
\right]
\delta\!\left(
\omega - \varepsilon_m(\mathbf{k}+\mathbf{q}) + \varepsilon_n(\mathbf{k})
\right)
\, d^3k,
\]

while `complex_frequency_polarization_weights` approximates the complex-energy
response

\[
\Pi_0(\mathbf{q}, z) =
\sum_{nm} \int_{\mathrm{BZ}}
\frac{
f\!\left(\varepsilon_n(\mathbf{k})\right)
- f\!\left(\varepsilon_m(\mathbf{k}+\mathbf{q})\right)
}{
z + \varepsilon_n(\mathbf{k}) - \varepsilon_m(\mathbf{k}+\mathbf{q})
}
\, d^3k.
\]

These routines return arrays of shape

\[
(n_E, wx, wy, wz, n_{\mathrm{target}}, n_{\mathrm{source}}).
\]

Use `prepare_response_evaluator` when the occupied and target band sets are
fixed but you want to sweep over many frequencies.

## How To Contract Weights Into Observables

If your observable carries a matrix element \(M_{nm}(\mathbf{k})\), form the
contraction before the final k-point sum. For example, for a DOS-like quantity
weighted by a scalar factor \(g_n(\mathbf{k})\),

\[
A(E_\alpha) =
\sum_n \int_{\mathrm{BZ}}
g_n(\mathbf{k})\,
\delta\!\left(E - \varepsilon_n(\mathbf{k})\right)\, d^3k
\approx
\det(bvec)\,
\sum_{\mathbf{k}, n}
g_n(\mathbf{k})\, w_{\alpha,\mathbf{k},n}.
\]

The same pattern applies to response functions with
\(M_{nm}(\mathbf{k})\,w_{E,\mathbf{k},m,n}\).

For dense 2D \((q_x, q_y, \omega)\) sweeps where you only need the final
contracted observable, `bztetra.twod` also exposes direct contracted response
evaluators via `fermi_golden_rule_observables` and
`complex_frequency_polarization_observables`. These accept matrix elements with
arbitrary leading channel axes, so you can evaluate operator-resolved response
matrices such as \(\chi_{AB}\) without materializing the full k-resolved weight
tensor.

For the same occupied-to-empty branch, the positive-frequency spectral weights
also determine the negative-real-axis continuation of the complex kernel. The
public causality helper `reconstruct_retarded_response` and the 2D convenience
wrapper `retarded_response_observables` automate that reconstruction from
`fermi_golden_rule_observables`, using the static observable as the
zero-frequency anchor and the source-to-target transition-energy bounds as an
automatic compact-support window. In the current package conventions, this
matches evaluating `complex_frequency_polarization_observables(-omega + 0j)` on
the real axis.

## Causality And Kramers-Kronig Reconstruction

For the default occupied-to-empty branch used by the contracted real-frequency
APIs, the positive-frequency spectrum determines the imaginary part as

\[
\operatorname{Im}\Pi_{\mathrm{oe}}(-\omega + i0^+) = \pi S(\omega),
\qquad \omega \ge 0.
\]

`reconstruct_retarded_response` takes that one-sided imaginary part on
`\omega >= 0` and reconstructs the corresponding real part on the same sampled
grid.

There are two supported branch conventions:

- default (`assume_hermitian=False`): the unspecified negative-frequency branch
  is taken to be zero. This matches
  `complex_frequency_polarization_observables(-omega + 0j)` on the same
  occupied-to-empty branch.
- Hermitian (`assume_hermitian=True`): the negative-frequency branch is
  generated by odd extension,
  \(\operatorname{Im}\chi(-\omega) = -\operatorname{Im}\chi(\omega)\), which is
  the appropriate convention for full Hermitian self-responses such as density
  or spin susceptibilities.

The optional `static_anchor` has different roles in those two modes:

- in the default non-Hermitian branch, it is only valid when the requested
  frequency grid contains `omega = 0`, because only that returned zero-frequency
  point is pinned;
- in the Hermitian branch, it fixes the zero-frequency constant of the full
  reconstructed curve.

The optional `support_bounds=(\omega_{\min}, \omega_{\max})` describe a compact
window where the imaginary part is expected to live. When a support edge falls
between two requested frequency samples, the reconstruction inserts an explicit
zero-valued sample at that edge on its internal working grid so the clipped
piecewise-linear model is exact on that grid. The returned `RetardedResponse`
still uses the original requested `omega` values.

`RetardedResponse.diagnostics` exposes the operational details that matter in
practice:

- whether a synthetic `omega = 0` point was inserted internally,
- whether the Hermitian odd-extension logic had to force `Im chi(0) = 0`,
- whether support clipping was applied and whether support-edge samples were
  inserted,
- the minimum and maximum working-grid spacing,
- the size of the augmented Kramers-Kronig operator,
- and whether that dense operator came from the bounded small-grid cache.

### Kramers-Kronig Reconstruction

`reconstruct_retarded_response` reconstructs the real part from an imaginary
part sampled on a positive-frequency grid. The helper supports two distinct
continuations.

For the default occupied-to-empty branch,

\[
\operatorname{Im}\Pi_{\mathrm{branch}}(\omega > 0)
= \text{input},\qquad
\operatorname{Im}\Pi_{\mathrm{branch}}(\omega < 0) = 0.
\]

This is the convention used by the contracted 2D spectral routines and it
matches the negative-real-axis evaluation
`complex_frequency_polarization_observables(-omega + 0j)`.

For full Hermitian self-responses, set `assume_hermitian=True`. The helper then
uses the odd extension

\[
\operatorname{Im}\chi(-\omega) = -\operatorname{Im}\chi(\omega),
\]

which is the usual causal symmetry for density-density or spin-spin response
functions.

The zero-frequency anchor has different semantics in those two modes:

- Default non-Hermitian branch: `static_anchor` pins only the returned
  \(\omega = 0\) sample, so it is only valid when the requested grid includes
  `omega[0] == 0`.
- Hermitian mode: `static_anchor` fixes the additive constant of the full
  reconstructed curve by matching \(\operatorname{Re}\chi(0)\).

`support_bounds=(ω_{\min}, ω_{\max})` describes a compact-support window for
the supplied imaginary part. The implementation clips the spectrum on its
working piecewise-linear grid and inserts zero-valued edge samples whenever a
support boundary falls between two requested frequencies. That means the
requested output grid is preserved, but the internal quadrature still sees the
exact zero crossing at the specified support edge.

The public diagnostics report the operational details of that reconstruction:

- `minimum_spacing`, `maximum_spacing`: smallest and largest working-grid
  spacing after support clipping and any synthetic zero insertion.
- `inserted_zero_frequency`: whether the helper added an internal `ω = 0`
  sample because the request started at `ω > 0`.
- `support_boundary_insertions`: how many zero-valued support-edge samples were
  inserted internally.
- `static_anchor_applied`: whether the supplied anchor actually modified the
  reconstruction.
- `cached_operator`: whether the dense Kramers-Kronig operator came from the
  small-grid cache rather than being built as an uncached one-off matrix.

## Where To Go Next

- [Examples](examples.md) for worked examples with plots.
- [Quickstart](quickstart.md) for the shortest correct calling patterns.
- [API Reference](api.md) for signatures and shape contracts.
