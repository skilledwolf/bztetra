from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


LEGACY_TEST_DIR = Path(__file__).resolve().parent / "data" / "legacy" / "test"


@dataclass(frozen=True)
class LegacyShellValue:
    result: np.ndarray
    atol: np.ndarray


@lru_cache(maxsize=None)
def load_legacy_shell_case(
    energy_grid_shape: tuple[int, int, int],
    weight_grid_shape: tuple[int, int, int],
) -> dict[str, LegacyShellValue]:
    if len(set(energy_grid_shape)) != 1 or len(set(weight_grid_shape)) != 1:
        raise ValueError("legacy shell references only exist for cubic energy and weight grids")

    energy_grid = energy_grid_shape[0]
    weight_grid = weight_grid_shape[0]
    path = LEGACY_TEST_DIR / f"test2_{energy_grid}_{weight_grid}.sh"
    sections = _parse_shell_reference_sections(path)

    return {
        "occupation_weights": _parse_real_value(sections["libtetrabz_occ"], (2,)),
        "solve_fermi_energy_fermi_energy": _parse_real_value(
            sections["libtetrabz_fermieng"][:1], ()
        ),
        "solve_fermi_energy_weights": _parse_real_value(
            sections["libtetrabz_fermieng"][1:], (2,)
        ),
        "density_of_states_weights": _parse_real_value(sections["libtetrabz_dos"], (5, 2)),
        "integrated_density_of_states_weights": _parse_real_value(
            sections["libtetrabz_intdos"], (5, 2)
        ),
        "phase_space_overlap_weights": _parse_real_value(
            sections["libtetrabz_dblstep"], (2, 2)
        ),
        "nesting_function_weights": _parse_real_value(sections["libtetrabz_dbldelta"], (2, 2)),
        "static_polarization_weights": _parse_real_value(
            sections["libtetrabz_polstat"], (2, 2)
        ),
        "fermi_golden_rule_weights": _parse_real_value(
            sections["libtetrabz_fermigr"], (3, 2, 2)
        ),
        "complex_frequency_polarization_weights": _parse_complex_value(
            sections["libtetrabz_polcmplx"], (3, 2, 2)
        ),
    }


def _parse_shell_reference_sections(path: Path) -> dict[str, list[tuple[str, str]]]:
    sections: dict[str, list[tuple[str, str]]] = {}
    current_section: str | None = None
    in_reference = False

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if line == "cat > test.out.ref <<EOF":
            in_reference = True
            continue
        if not in_reference:
            continue
        if line == "EOF":
            break
        if not line or line == "#          Ideal          Result":
            continue
        if line.startswith("# libtetrabz_"):
            current_section = line[2:].strip()
            sections[current_section] = []
            continue
        if current_section is None:
            continue
        ideal_token, result_token = line.split()
        sections[current_section].append((ideal_token, result_token))

    return sections


def _parse_real_value(
    entries: list[tuple[str, str]],
    shape: tuple[int, ...],
) -> LegacyShellValue:
    values = np.array([_parse_fortran_float(result) for _, result in entries], dtype=np.float64)
    tolerances = np.array([_printed_decimal_ulp(result) for _, result in entries], dtype=np.float64)
    return LegacyShellValue(
        result=values.reshape(shape, order="F"),
        atol=tolerances.reshape(shape, order="F"),
    )


def _parse_complex_value(
    entries: list[tuple[str, str]],
    shape: tuple[int, ...],
) -> LegacyShellValue:
    real_values = np.array(
        [_parse_fortran_float(result) for _, result in entries[0::2]],
        dtype=np.float64,
    )
    imag_values = np.array(
        [_parse_fortran_float(result) for _, result in entries[1::2]],
        dtype=np.float64,
    )
    real_tolerances = np.array(
        [_printed_decimal_ulp(result) for _, result in entries[0::2]],
        dtype=np.float64,
    )
    imag_tolerances = np.array(
        [_printed_decimal_ulp(result) for _, result in entries[1::2]],
        dtype=np.float64,
    )
    return LegacyShellValue(
        result=(real_values + 1.0j * imag_values).reshape(shape, order="F"),
        atol=np.maximum(real_tolerances, imag_tolerances).reshape(shape, order="F"),
    )


def _parse_fortran_float(token: str) -> float:
    return float(token.replace("D", "E"))


def _printed_decimal_ulp(token: str) -> float:
    mantissa, exponent = token.upper().split("E")
    _, _, fraction = mantissa.partition(".")
    return 10.0 ** (int(exponent) - len(fraction))
