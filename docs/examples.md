# Examples

Run examples from the repository root. Plotting scripts need Matplotlib:

```bash
pip install -e '.[plot]'
```

## Recommended Starting Points

| If you want... | Script | Run |
| --- | --- | --- |
| A first DOS figure against a legacy reference | `examples/plot_tight_binding_dos.py` | `python examples/plot_tight_binding_dos.py` |
| A small occupation and Fermi-level sanity check | `examples/review_occupancy.py` | `python examples/review_occupancy.py` |
| A single static-response example with a known physical curve | `examples/plot_lindhard.py` | `python examples/plot_lindhard.py` |

## Additional Examples

| Script | What it shows | Run |
| --- | --- | --- |
| `examples/review_dos.py` | Numeric DOS and integrated-DOS checks against a simple analytic free-electron case | `python examples/review_dos.py` |
| `examples/plot_phase_space_and_nesting.py` | Phase-space overlap and nesting curves for the free-electron response setup | `python examples/plot_phase_space_and_nesting.py` |
| `examples/plot_fermi_golden_rule.py` | Real-frequency transition weights on a free-electron toy model | `python examples/plot_fermi_golden_rule.py` |
| `examples/plot_complex_frequency_polarization.py` | Imaginary-frequency polarization on the Matsubara axis | `python examples/plot_complex_frequency_polarization.py` |
| `examples/review_geometry_and_cuts.py` | Low-level geometry and scalar-cut review for debugging or method work | `python examples/review_geometry_and_cuts.py` |

## Choosing Quickly

- Start with `plot_tight_binding_dos.py` if you want a release-style visual
  sanity check.
- Start with `review_occupancy.py` if you need to confirm the occupation or
  Fermi-energy path before moving on to response calculations.
- Start with `plot_lindhard.py` if your main use case is static response.
