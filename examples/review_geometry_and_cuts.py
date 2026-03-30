from __future__ import annotations

import numpy as np

from tetrabz import build_integration_mesh
from tetrabz import small_tetrahedron_cut
from tetrabz import triangle_cut


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)

    mesh = build_integration_mesh(np.eye(3, dtype=np.float64), (2, 2, 2), method="optimized")
    print("Mesh summary")
    print("  tetrahedron_count:", mesh.tetrahedron_count)
    print("  local_point_count:", mesh.local_point_count)
    print("  first tetrahedron corner offsets:")
    print(mesh.tetrahedra_offsets[0, :4])
    print("  first tetrahedron global point indices:")
    print(mesh.global_point_indices[0, :4])
    print()

    a1_cut = small_tetrahedron_cut("a1", np.array([-2.0, 1.0, 3.0, 5.0]))
    print("Small tetrahedron A1 example")
    print("  volume_factor:", a1_cut.volume_factor)
    print("  coefficients:")
    print(a1_cut.coefficients)
    print()

    c1_triangle = triangle_cut("c1", np.array([-5.0, -3.0, -1.0, 2.0]))
    print("Triangle C1 example")
    print("  volume_factor:", c1_triangle.volume_factor)
    print("  coefficients:")
    print(c1_triangle.coefficients)


if __name__ == "__main__":
    main()
