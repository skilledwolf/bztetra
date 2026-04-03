# Roadmap

## Kramers-Kronig Hardening

- [x] Capture the current response/Kramers-Kronig state in a baseline commit.
- [x] Fix non-Hermitian `static_anchor` handling when the input grid starts at `omega > 0`.
- [x] Replace the size-blind dense operator cache with a bounded policy that does not accumulate multi-GB state.
- [x] Remove dead padding/error-control API knobs and placeholder diagnostics.
- [x] Tighten support clipping so diagnostics and behavior match the implemented piecewise-linear clipping model.
- [x] Add regression tests for nonzero-start anchors, support-bound clipping, positive-start 2D wrappers, and cache behavior.
- [x] Run targeted validation and record final status here.

## Progress Notes

- 2026-04-03: Started KK implementation review and cleanup pass.
- 2026-04-03: Added full `(q_x, q_y, omega)` 2DEG benchmark example.
- 2026-04-03: Hardened the KK backend, simplified the public diagnostics, and bounded the operator cache.
- 2026-04-03: Validation passed for `38` targeted tests plus the 1D and full-q-grid retarded-response examples.
