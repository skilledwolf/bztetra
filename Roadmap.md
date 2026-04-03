# Roadmap

## Kramers-Kronig Hardening

- [x] Capture the current response/Kramers-Kronig state in a baseline commit.
- [x] Fix non-Hermitian `static_anchor` handling when the input grid starts at `omega > 0`.
- [x] Replace the size-blind dense operator cache with a bounded policy that does not accumulate multi-GB state.
- [x] Remove dead padding/error-control API knobs and placeholder diagnostics.
- [x] Tighten support clipping so diagnostics and behavior match the implemented piecewise-linear clipping model.
- [x] Add regression tests for nonzero-start anchors, support-bound clipping, positive-start 2D wrappers, and cache behavior.
- [x] Run targeted validation and record final status here.

## Kramers-Kronig Documentation

- [x] Audit the current KK docs against the shipped implementation.
- [x] Add a dedicated physics-facing explanation of the default branch, Hermitian mode, anchors, support clipping, and diagnostics.
- [x] Add a direct `reconstruct_retarded_response(...)` usage example and practical caveats.
- [x] Re-check API/examples/docstrings for consistency and record final status here.

## Publication Readiness

- [x] State clearly that the mesh topology is periodic-only and that open-boundary conditions are not supported.
- [x] Mark the free-electron / 2DEG review scripts as periodic-box approximations rather than true open-boundary calculations.
- [x] Run the full repository test suite after the publish-readiness docs patch.
- [x] Record final publication-readiness status here.

## Progress Notes

- 2026-04-03: Started KK implementation review and cleanup pass.
- 2026-04-03: Added full `(q_x, q_y, omega)` 2DEG benchmark example.
- 2026-04-03: Hardened the KK backend, simplified the public diagnostics, and bounded the operator cache.
- 2026-04-03: Validation passed for `38` targeted tests plus the 1D and full-q-grid retarded-response examples.
- 2026-04-03: Started a documentation hardening pass for the KK / causality layer.
- 2026-04-03: Expanded the KK docs in physics/examples/API text and public docstrings to match the shipped branch, anchor, support, and diagnostics behavior.
- 2026-04-03: Started a publish-readiness pass for boundary-condition and example-interpretation docs.
- 2026-04-03: Clarified the periodic-only mesh model in the README and docs, and marked the free-electron / 2DEG examples as periodic-box approximations.
- 2026-04-03: Full validation passed with `128 passed, 4 skipped`, and `mkdocs build` completed successfully.
