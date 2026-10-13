"""Normalize relative wavelength PL spectra while retaining their scale."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Mapping

import numpy as np


NORMALIZATION_METHOD = "divide_by_wavelength_maximum"
NORMALIZATION_KEYS = (
    "intensity_normalization", "normalization_reference_intensity",
    "normalization_reference_unit",
)
_RELATIVE_UNITS = {"a.u.", "relative", "relative/nm", "a.u./nm"}
NO_SIGNAL_MESSAGES = {
    "no_bound_transitions": "No bound electron-hole transitions",
    "no_signal_in_window": "No PL signal in selected range",
}


def normalize_relative_spectrum(spectrum: Mapping[str, Any]) -> dict[str, Any]:
    """Return I/max(I) for relative spectra without changing the input arrays.

    Apply this only after conversion to wavelength. Explicit physical units
    are preserved. Repeated preparation retains the original reference scale.
    Explicitly reported no-signal results stay zero, without normalization.
    """
    prepared = dict(spectrum)
    unit = prepared.get("intensity_unit")
    if not isinstance(unit, str) or unit.strip() not in _RELATIVE_UNITS:
        return prepared
    intensity = np.asarray(prepared["intensity"], dtype=float)
    if (intensity.ndim != 1 or intensity.size == 0
            or not np.all(np.isfinite(intensity)) or np.any(intensity < 0)):
        raise ValueError("PL normalization requires finite nonnegative intensities.")
    maximum = float(np.max(intensity))
    if maximum <= 0:
        if prepared.get("signal_status") not in NO_SIGNAL_MESSAGES:
            raise ValueError("PL normalization requires at least one positive intensity or an explicit no-signal status.")
        prepared.update({"intensity": intensity.copy(), "intensity_unit": "a.u."})
        for key in NORMALIZATION_KEYS:
            prepared.pop(key, None)
        return prepared
    reference = maximum
    reference_unit = unit.strip()
    if prepared.get("intensity_normalization") == NORMALIZATION_METHOD:
        previous = prepared.get("normalization_reference_intensity")
        reference_unit = prepared.get("normalization_reference_unit")
        if (isinstance(previous, bool) or not isinstance(previous, Real)
                or not math.isfinite(previous) or previous <= 0
                or not isinstance(reference_unit, str) or not reference_unit.strip()):
            raise ValueError("A normalized PL spectrum requires a positive finite reference and its unit.")
        reference = float(previous) * maximum
        if not math.isfinite(reference) or reference <= 0:
            raise ValueError("The PL normalization reference must remain positive and finite.")
    prepared.update({
        "intensity": intensity / maximum,
        "intensity_unit": "a.u.",
        "intensity_normalization": NORMALIZATION_METHOD,
        "normalization_reference_intensity": reference,
        "normalization_reference_unit": reference_unit,
    })
    return prepared
