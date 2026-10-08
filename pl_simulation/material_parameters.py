"""User-editable unstrained wurtzite material parameters for the Python model.

This is a scalar effective-mass model, not a multiband calculation. The
bundled literature values and the reductions used for hole masses are
identified in configs/materials.json. No nextnano executable or database
is consulted.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

import numpy as np


_BINARY_KEYS = {
    "band_gap_0k_ev", "varshni_alpha_ev_k", "varshni_beta_k",
    "electron_mass_z", "electron_mass_xy", "hole_mass_z", "hole_mass_xy",
    "source_id", "hole_mass_derivation",
}
_NUMERIC_KEYS = _BINARY_KEYS - {"source_id", "hole_mass_derivation"}


def _finite_number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive.")
    return float(value)


def _validate_materials(data: Mapping[str, Any]) -> None:
    if not isinstance(data, Mapping):
        raise ValueError("Material data must be a JSON object.")
    required = {
        "schema_version", "crystal", "energy_reference", "gap_bowing_ev",
        "mass_interpolation", "materials", "sources", "notes",
    }
    if set(data) != required:
        raise ValueError(f"Material data must contain exactly these keys: {sorted(required)}")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("Only material schema_version 1 is supported.")
    if data["crystal"] != "wurtzite":
        raise ValueError("Only wurtzite material parameters are supported.")
    if data["energy_reference"] != "unstrained_GaN_VBM":
        raise ValueError("Material energy_reference must be unstrained_GaN_VBM.")
    if data["mass_interpolation"] != "linear_mass":
        raise ValueError("Only linear_mass interpolation is implemented.")
    if _finite_number(data["gap_bowing_ev"], "gap_bowing_ev") < 0:
        raise ValueError("gap_bowing_ev must be nonnegative.")
    binaries = data["materials"]
    if not isinstance(binaries, Mapping) or set(binaries) != {"GaN", "InN"}:
        raise ValueError("Material data must define exactly the GaN and InN binaries.")
    sources = data["sources"]
    if not isinstance(sources, Mapping) or not sources:
        raise ValueError("Material sources must be a nonempty object.")
    for source_id, source in sources.items():
        if not isinstance(source_id, str) or not source_id.strip() or not isinstance(source, Mapping):
            raise ValueError("Each material source must have a nonempty name and an object.")
        for field in ("authors", "title", "url"):
            if not isinstance(source.get(field), str) or not source[field].strip():
                raise ValueError(f"Material source {source_id} requires {field}.")
    for name, binary in binaries.items():
        if not isinstance(binary, Mapping) or set(binary) != _BINARY_KEYS:
            raise ValueError(f"{name} must contain exactly these keys: {sorted(_BINARY_KEYS)}")
        for key in _NUMERIC_KEYS:
            _finite_number(binary[key], f"{name}.{key}", positive=True)
        source_id = binary["source_id"]
        if not isinstance(source_id, str) or source_id not in sources:
            raise ValueError(f"{name}.source_id must identify a listed material source.")
        if not isinstance(binary["hole_mass_derivation"], str) or not binary["hole_mass_derivation"].strip():
            raise ValueError(f"{name}.hole_mass_derivation must explain the selected scalar masses.")
    notes = data["notes"]
    if not isinstance(notes, list) or any(not isinstance(note, str) for note in notes):
        raise ValueError("Material notes must be a list of strings.")


def load_materials(path: str | Path) -> dict[str, Any]:
    """Load and validate a complete material JSON file without solver access."""
    with Path(path).expanduser().open(encoding="utf-8") as stream:
        data = json.load(stream)
    _validate_materials(data)
    return data


def _temperature_gap(binary: Mapping[str, Any], temperature_k: float) -> float:
    gap = binary["band_gap_0k_ev"] - (
        binary["varshni_alpha_ev_k"] * temperature_k**2
        / (temperature_k + binary["varshni_beta_k"])
    )
    if not math.isfinite(gap) or gap <= 0:
        raise ValueError("Material gap must remain positive at the configured temperature.")
    return float(gap)


def build_band_profiles(
    structure: Any,
    settings: Any,
    materials: Mapping[str, Any],
    x_nm: np.ndarray,
) -> dict[str, Any]:
    """Construct bands in eV and scalar masses m/m0 on an increasing nm grid.

    GaN's unstrained VBM is the zero-energy reference before applying the
    prescribed field. With Qc the conduction offset ratio and x the In
    fraction, Ev=(1-Qc)*(Eg_GaN-Eg_alloy), Ec=Ev+Eg_alloy.
    The field adds +1e-4*F_kV_per_cm*(z_nm-L_nm/2) to both electron
    band edges. Interior material interfaces belong to their right-hand
    layer; the final structure endpoint belongs to the right barrier.
    """
    structure.validate()
    settings.validate()
    _validate_materials(materials)
    if getattr(settings, "include_strain", False) or getattr(settings, "include_polarization", False):
        raise ValueError("This material model does not calculate strain or automatic polarization.")
    normal = tuple(settings.x_hkl)
    if normal[0] != 0 or normal[1] != 0 or normal[2] == 0:
        raise ValueError("Scalar A-like material masses currently support c-axis growth only.")
    temperature = _finite_number(settings.temperature_k, "temperature_k", positive=True)
    ratio = _finite_number(settings.conduction_band_offset_ratio, "conduction_band_offset_ratio")
    if not 0 < ratio < 1:
        raise ValueError("conduction_band_offset_ratio must be between zero and one.")
    field = _finite_number(settings.electric_field_kv_cm, "electric_field_kv_cm")
    positions = np.asarray(x_nm, dtype=float)
    if positions.ndim != 1 or len(positions) < 2 or not np.all(np.isfinite(positions)):
        raise ValueError("x_nm must contain at least two finite positions in a one-dimensional array.")
    if not np.all(np.diff(positions) > 0):
        raise ValueError("x_nm must be strictly increasing.")
    ranges = structure.get_layer_ranges()
    total_length = float(ranges[-1]["end_nm"])
    tolerance = 1e-10 * max(1.0, total_length)
    if positions[0] < -tolerance or positions[-1] > total_length + tolerance:
        raise ValueError("x_nm positions must lie within the configured structure.")
    positions = np.clip(positions, 0.0, total_length)
    gan, inn = materials["materials"]["GaN"], materials["materials"]["InN"]
    gan_gap = _temperature_gap(gan, temperature)
    inn_gap = _temperature_gap(inn, temperature)
    if gan_gap <= inn_gap:
        raise ValueError("This GaN/InGaN well model requires the GaN gap to exceed the InN gap.")
    arrays = {
        key: np.empty_like(positions)
        for key in (
            "conduction_ev", "valence_ev", "electron_mass_z", "hole_mass_z",
            "electron_mass_xy", "hole_mass_xy",
        )
    }
    layer_metadata = []
    for index, layer in enumerate(ranges):
        fraction = float(layer["indium_fraction"])
        gap = ((1 - fraction) * gan_gap + fraction * inn_gap
               - materials["gap_bowing_ev"] * fraction * (1 - fraction))
        if not math.isfinite(gap) or gap <= 0:
            raise ValueError(f"The configured alloy gap is not positive in layer {layer['name']}.")
        valence = (1 - ratio) * (gan_gap - gap)
        conduction = valence + gap
        mask = (positions >= layer["start_nm"]) & (
            positions <= layer["end_nm"] if index == len(ranges) - 1
            else positions < layer["end_nm"]
        )
        arrays["conduction_ev"][mask] = conduction
        arrays["valence_ev"][mask] = valence
        masses = {}
        for key in ("electron_mass_z", "hole_mass_z", "electron_mass_xy", "hole_mass_xy"):
            masses[key] = (1 - fraction) * gan[key] + fraction * inn[key]
            arrays[key][mask] = masses[key]
        layer_metadata.append({
            **layer, "band_gap_ev": gap, "conduction_unshifted_ev": conduction,
            "valence_unshifted_ev": valence, **masses,
        })
    field_shift = 0.0001 * field * (positions - total_length / 2)
    arrays["conduction_ev"] += field_shift
    arrays["valence_ev"] += field_shift
    if any(not np.all(np.isfinite(values)) for values in arrays.values()):
        raise ValueError("Configured parameters produce nonfinite material profiles.")
    return {
        **arrays,
        "metadata": {
            "energy_reference": materials["energy_reference"],
            "energy_reference_description": "Unstrained GaN VBM=0 before prescribed field; field gauge is zero at structure centre.",
            "temperature_k": temperature,
            "gap_bowing_ev": materials["gap_bowing_ev"],
            "conduction_band_offset_ratio": ratio,
            "electric_field_kv_cm": field,
            "field_model": "prescribed_uniform_external_field",
            "field_electron_energy_sign": "+1e-4*F_kV_cm*(z_nm-L_nm/2)",
            "hole_model": "scalar_A_like_diagonal_projection_no_valence_mixing",
            "mass_interpolation": materials["mass_interpolation"],
            "include_strain": False,
            "include_automatic_polarization": False,
            "layers": layer_metadata,
            "materials": deepcopy(dict(materials)),
        },
    }
