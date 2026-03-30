# Physics Guide

This page is the physics-facing map of the package: what each routine computes,
how the returned weights relate to standard Brillouin-zone formulas, and how to
turn those weights into observables.

All formulas on this page refer to the current 3D package. Genuine 2D support
will require a separate triangle-method implementation with Brillouin-zone area
normalization rather than the present tetrahedron path with `nz=1`.

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

`phase_space_overlap_weights` follows the legacy `dblstep` semantics: occupied
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

## Where To Go Next

- [Examples](examples.md) for worked examples with plots.
- [Quickstart](quickstart.md) for the shortest correct calling patterns.
- [API Reference](api.md) for signatures and shape contracts.
