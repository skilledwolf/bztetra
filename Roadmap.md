# Roadmap

## Kramers-Kronig Hardening

- [x] Capture the current response/Kramers-Kronig state in a baseline commit.
- [ ] Fix non-Hermitian `static_anchor` handling when the input grid starts at `omega > 0`.
- [ ] Replace the size-blind dense operator cache with a bounded policy that does not accumulate multi-GB state.
- [ ] Remove or reimplement dead padding/error-control API knobs and placeholder diagnostics.
- [ ] Tighten support clipping so diagnostics and behavior match the documented approximation.
- [ ] Add regression tests for nonzero-start anchors, support-bound clipping, and cache behavior.
- [ ] Run targeted validation and record final status here.

## Progress Notes

- 2026-04-03: Started KK implementation review and cleanup pass.
- 2026-04-03: Added full `(q_x, q_y, omega)` 2DEG benchmark example.
