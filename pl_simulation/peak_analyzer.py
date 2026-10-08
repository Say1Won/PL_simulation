"""Analyze and plot solver-calculated wavelength emission spectra."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


class PeakAnalyzer:
    """Keep wavelength-spectrum peaks separate from level-spacing estimates."""

    HC_EV_NM = 1239.8419843320026

    def __init__(self, spectrum: Mapping[str, Any] | None = None) -> None:
        self.spectrum = spectrum

    def _spectrum(self, spectrum: Mapping[str, Any] | None) -> tuple[np.ndarray, np.ndarray, str, Mapping[str, Any]]:
        source = self.spectrum if spectrum is None else spectrum
        if source is None:
            raise ValueError("Provide a solver-calculated wavelength spectrum.")
        if source.get("axis", "wavelength") != "wavelength" or source.get("axis_unit", "nm") != "nm":
            raise ValueError("PeakAnalyzer requires a wavelength spectrum in nm.")
        x = np.asarray(source["wavelength_nm"], dtype=float)
        y = np.asarray(source["intensity"], dtype=float)
        if x.ndim != 1 or x.size == 0 or y.shape != x.shape:
            raise ValueError("Wavelength and intensity must be nonempty one-dimensional arrays of equal length.")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("Spectrum coordinates and intensities must be finite.")
        if np.any(x <= 0) or np.any(np.diff(x) <= 0):
            raise ValueError("Wavelengths must be positive and strictly increasing.")
        if np.any(y < 0) or not np.any(y > 0):
            raise ValueError("Emission intensities must be nonnegative with at least one positive value.")
        unit = source.get("intensity_unit")
        if not isinstance(unit, str) or not unit.strip():
            raise ValueError("A declared emission intensity unit is required.")
        return x, y, unit, source

    @classmethod
    def estimate_transition(
        cls,
        electron_ev: float,
        valence_ev: float,
        *,
        energy_reference: str,
    ) -> dict[str, Any]:
        """Estimate a level-to-level transition using two electron energies.

        Both inputs must use the declared shared energy reference. This estimate
        does not include occupancies, matrix elements, broadening or excitons
        and is not a simulated PL-spectrum peak.
        """
        if not isinstance(energy_reference, str) or not energy_reference.strip():
            raise ValueError("Declare the shared energy reference for both states.")
        if not np.isfinite(electron_ev) or not np.isfinite(valence_ev):
            raise ValueError("State energies must be finite.")
        energy = float(electron_ev - valence_ev)
        if energy <= 0:
            raise ValueError("Conduction electron energy must exceed valence electron energy.")
        return {
            "energy_ev": energy, "wavelength_nm": cls.HC_EV_NM / energy,
            "energy_reference": energy_reference, "kind": "level_spacing_estimate",
        }

    @staticmethod
    def calculate_overlap(
        position_nm: Any,
        electron_wavefunction: Any,
        hole_wavefunction: Any,
    ) -> float:
        """Return normalized |∫ψe*ψh dx|² for scalar envelopes on one grid.

        This envelope overlap is not the optical transition matrix element of
        an eight-band spinor. Probability densities must not be supplied as
        wavefunction amplitudes.
        """
        x = np.asarray(position_nm, dtype=float)
        electron = np.asarray(electron_wavefunction, dtype=complex)
        hole = np.asarray(hole_wavefunction, dtype=complex)
        if x.ndim != 1 or x.size < 2 or electron.shape != x.shape or hole.shape != x.shape:
            raise ValueError("Two scalar envelopes must share a one-dimensional grid with at least two points.")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(electron)) or not np.all(np.isfinite(hole)):
            raise ValueError("Grid and wavefunction values must be finite.")
        if np.any(np.diff(x) <= 0):
            raise ValueError("Wavefunction positions must be strictly increasing.")
        widths = np.diff(x)
        electron_density = np.abs(electron) ** 2
        hole_density = np.abs(hole) ** 2
        electron_norm = float(np.sum((electron_density[:-1] + electron_density[1:]) * widths / 2))
        hole_norm = float(np.sum((hole_density[:-1] + hole_density[1:]) * widths / 2))
        if electron_norm <= 0 or hole_norm <= 0:
            raise ValueError("Wavefunctions must have nonzero norm.")
        product = np.conj(electron) * hole
        integral = np.sum((product[:-1] + product[1:]) * widths / 2)
        return float(np.clip(np.abs(integral) ** 2 / (electron_norm * hole_norm), 0.0, 1.0))

    @staticmethod
    def select_bound_states(
        energies_ev: Any,
        well_probability: Any,
        threshold: float = 0.5,
    ) -> dict[str, np.ndarray]:
        """Select well-localized candidates by supplied integrated probability.

        Localization alone is not proof of a true bound state: verify energies
        against the appropriate barrier continuum, particularly under fields.
        """
        energies = np.asarray(energies_ev, dtype=float)
        weights = np.asarray(well_probability, dtype=float)
        if energies.ndim != 1 or weights.shape != energies.shape or energies.size == 0:
            raise ValueError("State energies and well probabilities must have equal nonzero lengths.")
        if not np.all(np.isfinite(energies)) or not np.all(np.isfinite(weights)):
            raise ValueError("Energies and well probabilities must be finite.")
        if np.any(weights < 0) or np.any(weights > 1) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Probabilities and threshold must lie between 0 and 1.")
        indices = np.flatnonzero(weights >= threshold)
        return {"indices": indices, "energies_ev": energies[indices], "well_probability": weights[indices]}

    def calculate_fwhm(
        self,
        spectrum: Mapping[str, Any] | None = None,
        peak_index: int | None = None,
    ) -> float | None:
        """Return the selected peak lobe's FWHM in nm, or None if truncated.

        Crossings are linearly interpolated around the contiguous interval
        above half the selected maximum, with no baseline subtraction. A lobe
        reaching a spectrum boundary has an unresolved width.
        """
        x, y, _, _ = self._spectrum(spectrum)
        index = int(np.argmax(y)) if peak_index is None else peak_index
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or not 0 <= index < x.size:
            raise ValueError("peak_index must be an integer inside the spectrum.")
        if y[index] <= 0 or index == 0 or index == x.size - 1:
            return None
        if y[index] < y[index - 1] or y[index] < y[index + 1]:
            raise ValueError("peak_index must identify a local maximum.")
        half = float(y[index] / 2)
        left = right = index
        while left > 0 and y[left - 1] >= half:
            left -= 1
        while right < x.size - 1 and y[right + 1] >= half:
            right += 1
        if left == 0 or right == x.size - 1:
            return None
        left_cross = x[left - 1] + (half - y[left - 1]) * (x[left] - x[left - 1]) / (y[left] - y[left - 1])
        right_cross = x[right] + (half - y[right]) * (x[right + 1] - x[right]) / (y[right + 1] - y[right])
        return float(right_cross - left_cross)

    def find_spectrum_peak(self, spectrum: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Report the sampled global wavelength maximum; ties choose the first."""
        x, y, unit, source = self._spectrum(spectrum)
        index = int(np.argmax(y))
        return {
            "wavelength_nm": float(x[index]), "energy_ev": self.HC_EV_NM / float(x[index]),
            "intensity": float(y[index]), "intensity_unit": unit, "peak_index": index,
            "fwhm_nm": self.calculate_fwhm(source, index),
            "axis": "wavelength", "axis_unit": "nm", "peak_at_boundary": index in (0, x.size - 1),
            "peak_method": "sampled_global_maximum_first_tie",
            "fwhm_method": "contiguous_half_maximum_linear_interpolation_no_baseline_subtraction",
        }

    def plot_pl_spectrum(self, path: str | Path, spectrum: Mapping[str, Any] | None = None) -> Path:
        """Save the wavelength spectrum and annotate its sampled maximum."""
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        x, y, unit, source = self._spectrum(spectrum)
        peak = self.find_spectrum_peak(source)
        figure = Figure(figsize=(8, 5), layout="constrained")
        FigureCanvasAgg(figure)
        ax = figure.subplots()
        ax.plot(x, y, label="Spontaneous emission")
        ax.scatter([peak["wavelength_nm"]], [peak["intensity"]], color="tab:red", zorder=3)
        ax.annotate(f"Peak: {peak['wavelength_nm']:.2f} nm", (peak["wavelength_nm"], peak["intensity"]), xytext=(8, 8), textcoords="offset points", fontsize=9)
        ax.set(xlabel="Wavelength (nm)", ylabel=f"Emission intensity ({unit})", title="PL emission spectrum")
        ax.grid(alpha=0.2)
        ax.legend()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=180)
        return destination

    def export_csv(self, path: str | Path, spectrum: Mapping[str, Any] | None = None) -> Path:
        """Save the same wavelength density used for peak analysis."""
        x, y, unit, _ = self._spectrum(spectrum)
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["wavelength_nm", f"intensity [{unit}]"])
            writer.writerows(zip(x, y))
        return destination

    def export_summary(self, path: str | Path, spectrum: Mapping[str, Any] | None = None) -> Path:
        """Save the peak result and relevant spectrum conversion metadata."""
        _, _, _, source = self._spectrum(spectrum)
        summary = self.find_spectrum_peak(source)
        for key in ("source", "source_axis", "source_axis_unit", "spectral_density", "conversion"):
            if key in source:
                summary[key] = source[key]
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return destination
