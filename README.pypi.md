# bztetra

`bztetra` is a Python + Numba package for tetrahedron integration on regular
k-grids.

This is the public `0.x` preview series. Validate important production results
against the original
[`libtetrabz`](https://github.com/mitsuaki1987/libtetrabz) implementation or
the parity guidance in the
[validation docs](https://skilledwolf.github.io/bztetra/validation/).

## Install

```bash
pip install bztetra
```

Plot-backed examples use Matplotlib:

```bash
pip install "bztetra[plot]"
```

## What It Provides

- Occupation, DOS, and integrated DOS weights on regular 3D k-grids.
- Static overlap, nesting, and polarization weights.
- Real- and complex-frequency Lindhard-style response functions.
- A separate `bztetra.twod` namespace for genuine 2D triangle-method problems.

## Documentation

- Documentation: <https://skilledwolf.github.io/bztetra/>
- Quickstart: <https://skilledwolf.github.io/bztetra/quickstart/>
- Physics guide: <https://skilledwolf.github.io/bztetra/physics/>
- Examples: <https://skilledwolf.github.io/bztetra/examples/>
- Validation: <https://skilledwolf.github.io/bztetra/validation/>
- Repository: <https://github.com/skilledwolf/bztetra>
- Issues: <https://github.com/skilledwolf/bztetra/issues>
