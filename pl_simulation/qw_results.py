"""Validate native calculation results or read explicitly selected solver data."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping
from zipfile import BadZipFile

import numpy as np


class QWResults:
    """Post-process native results or explicitly mapped external output files.

    Native calculations use :meth:`from_calculation` or :meth:`from_archive`
    and need no nextnanopy installation. For optional nextnano file imports,
    ``files`` maps ``band_edges``, ``states``, ``wavefunctions`` and ``spectrum``
    to specifications. Each source needs ``path`` or ``glob`` and exact column
    names (or deliberately configured zero-based indices). State sources are
    specified separately as ``states.electron`` and ``states.valence``; valence
    energies remain electron energies on the same reference as the bands.
    """

    HC_EV_NM = 1239.8419843320026
    _ARRAY_KEYS = (
        "position_nm", "conduction_ev", "valence_ev",
        "electron_energies_ev", "valence_energies_ev",
        "electron_wavefunctions", "hole_wavefunctions",
        "energy_ev", "energy_intensity",
    )

    def __init__(
        self,
        output_directory: str | Path,
        files: Mapping[str, Any] | None = None,
    ) -> None:
        self.output_directory = Path(output_directory).expanduser().resolve()
        self.files = deepcopy(dict(files or {}))
        self._data: dict[str, Any] = {}
        self._calculation: dict[str, Any] | None = None
        self._metadata: dict[str, Any] = {}

    @property
    def metadata(self) -> dict[str, Any]:
        """Return an independent copy of native model and provenance metadata."""
        return deepcopy(self._metadata)

    @classmethod
    def from_calculation(cls, calculation: Mapping[str, Any]) -> "QWResults":
        """Validate an effective-mass calculation on one shared energy reference.

        Required arrays are spatial band edges, state electron energies,
        scalar envelope wavefunctions and emission density per eV. Hole
        wavefunctions correspond to ``valence_energies_ev`` (valence electron
        energies), not positive hole confinement energies. Additional numeric
        arrays are preserved in archives without object serialization.
        """
        if not isinstance(calculation, Mapping):
            raise ValueError("A calculation must be a mapping of arrays and metadata.")
        missing = [key for key in cls._ARRAY_KEYS if key not in calculation]
        if missing:
            raise ValueError(f"Calculation is missing required arrays: {', '.join(missing)}.")
        raw_metadata = calculation.get("metadata", {})
        if not isinstance(raw_metadata, Mapping):
            raise ValueError("Calculation metadata must be a JSON object.")
        try:
            metadata = json.loads(json.dumps(dict(raw_metadata), allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError("Calculation metadata must contain finite JSON-compatible values.") from exc
        reference = metadata.setdefault("energy_reference", "unstrained_GaN_VBM")
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError("Calculation metadata requires a shared nonempty energy_reference.")
        unit = metadata.setdefault("energy_intensity_unit", "relative/eV")
        if not isinstance(unit, str) or not unit.endswith("/eV") or unit == "/eV":
            raise ValueError("energy_intensity_unit must explicitly describe density per eV.")

        arrays: dict[str, np.ndarray] = {}
        for key, values in calculation.items():
            if key == "metadata":
                continue
            if not isinstance(key, str) or not key or key in ("metadata_json", "format_version"):
                raise ValueError("Calculation array names must be nonempty, unreserved strings.")
            try:
                array = np.asarray(values)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Calculation array {key!r} cannot be converted to a numeric array.") from exc
            if array.dtype.kind not in "biufc" or not np.all(np.isfinite(array)):
                raise ValueError(f"Calculation array {key!r} must contain finite numeric values.")
            if key in cls._ARRAY_KEYS:
                if array.dtype.kind in "bc":
                    raise ValueError(f"Calculation array {key!r} must contain real numeric values.")
                array = np.asarray(array, dtype=float)
            arrays[key] = array.copy()

        for key in ("position_nm", "energy_ev"):
            axis = arrays[key]
            if axis.ndim != 1 or axis.size < 2 or np.any(np.diff(axis) <= 0):
                raise ValueError(f"{key} must be a strictly increasing one-dimensional axis with at least two points.")
        position = arrays["position_nm"]
        for key in ("conduction_ev", "valence_ev"):
            if arrays[key].shape != position.shape:
                raise ValueError(f"{key} must match position_nm.")
        for energy_key, wave_key in (
            ("electron_energies_ev", "electron_wavefunctions"),
            ("valence_energies_ev", "hole_wavefunctions"),
        ):
            energies = arrays[energy_key]
            if energies.ndim != 1 or energies.size == 0:
                raise ValueError(f"{energy_key} must be a nonempty one-dimensional array.")
            waves = arrays[wave_key]
            if waves.shape != (energies.size, position.size):
                raise ValueError(f"{wave_key} must have shape (state count, position count).")
            if np.any(np.max(np.abs(waves), axis=1) == 0):
                raise ValueError(f"{wave_key} cannot contain a zero envelope.")
        energy = arrays["energy_ev"]
        density = arrays["energy_intensity"]
        if np.any(energy <= 0) or density.shape != energy.shape or np.any(density < 0):
            raise ValueError("Energy spectrum requires positive energies and matching nonnegative density.")

        instance = cls(Path.cwd())
        instance._metadata = metadata
        instance._calculation = {**arrays, "metadata": deepcopy(metadata)}
        instance._data["band_edges"] = {
            "position_nm": position, "conduction_ev": arrays["conduction_ev"],
            "valence_ev": arrays["valence_ev"], "energy_reference": reference,
            "conduction_label": "effective mass", "valence_label": "effective mass",
            "valence_components_ev": {"effective mass": arrays["valence_ev"]},
        }
        states: dict[str, Any] = {"energy_reference": reference}
        waves: dict[str, Any] = {}
        for kind, energy_key, wave_key in (
            ("electron", "electron_energies_ev", "electron_wavefunctions"),
            ("valence", "valence_energies_ev", "hole_wavefunctions"),
        ):
            labels = [f"{kind} {index + 1}" for index in range(arrays[energy_key].size)]
            states[f"{kind}_ev"] = arrays[energy_key]
            states[f"{kind}_labels"] = labels
            waves[kind] = {
                "position_nm": position, "values": arrays[wave_key],
                "representation": "envelope", "labels": labels,
            }
        instance._data["states"] = states
        instance._data["wavefunctions"] = waves
        wavelength = cls.HC_EV_NM / energy
        wavelength, intensity = cls._sort_rows(
            wavelength, density * cls.HC_EV_NM / wavelength**2,
        )
        instance._data["spectrum"] = {
            "wavelength_nm": wavelength, "intensity": intensity,
            "intensity_unit": unit[:-3] + "/nm", "axis": "wavelength", "axis_unit": "nm",
            "source_axis": "energy", "source_axis_unit": "eV", "spectral_density": True,
            "conversion": "intensity_per_eV * hc_eV_nm / wavelength_nm**2",
            "source": "native_calculation",
        }
        return instance

    def save_calculation(self, path: str | Path) -> Path:
        """Save native arrays and JSON metadata as a pickle-free NumPy archive."""
        if self._calculation is None:
            raise ValueError("Only native calculation results can be saved as a calculation archive.")
        destination = Path(path).expanduser()
        if destination.suffix.lower() != ".npz":
            raise ValueError("Calculation archive path must end in .npz.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {key: value for key, value in self._calculation.items() if key != "metadata"}
        payload["metadata_json"] = np.asarray(json.dumps(self._metadata, allow_nan=False))
        payload["format_version"] = np.asarray(1, dtype=np.int64)
        # Exclusive creation preserves prior run results and prevents the suffix
        # rewriting performed by np.savez_compressed when given a path string.
        stream = destination.open("xb")
        try:
            with stream:
                np.savez_compressed(stream, **payload)
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        return destination

    @classmethod
    def from_archive(cls, path: str | Path) -> "QWResults":
        """Load and validate an archive without enabling pickle deserialization."""
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Calculation archive does not exist: {source}")
        try:
            archive = np.load(source, allow_pickle=False)
            if not isinstance(archive, np.lib.npyio.NpzFile):
                raise ValueError("Expected a .npz calculation archive.")
            with archive:
                keys = set(archive.files)
                required = {*cls._ARRAY_KEYS, "metadata_json", "format_version"}
                if not required.issubset(keys):
                    raise ValueError(f"Archive is missing required arrays: {', '.join(sorted(required - keys))}.")
                version = archive["format_version"]
                if version.shape != () or version.dtype.kind not in "iu" or int(version) != 1:
                    raise ValueError("Unsupported calculation archive format_version.")
                raw_metadata = archive["metadata_json"]
                if raw_metadata.shape != () or raw_metadata.dtype.kind != "U":
                    raise ValueError("Archive metadata_json must be a Unicode scalar.")
                metadata = json.loads(str(raw_metadata))
                calculation = {
                    key: archive[key].copy() for key in archive.files
                    if key not in ("metadata_json", "format_version")
                }
                calculation["metadata"] = metadata
            result = cls.from_calculation(calculation)
        except (OSError, ValueError, TypeError, KeyError, BadZipFile) as exc:
            raise ValueError(f"Invalid calculation archive {source}: {exc}") from exc
        result.output_directory = source.parent
        result._data["spectrum"]["source"] = str(source)
        return result

    def _spec(self, kind: str) -> dict[str, Any]:
        spec = self.files.get(kind)
        if not isinstance(spec, Mapping):
            raise ValueError(
                f"No output mapping for {kind!r}. Configure output_files.{kind} "
                "with a path/glob and explicit column selectors."
            )
        return dict(spec)

    def _load_file(self, spec: Mapping[str, Any]) -> Any:
        if bool(spec.get("path")) == bool(spec.get("glob")):
            raise ValueError("An output source needs exactly one of 'path' or 'glob'.")
        if spec.get("glob"):
            matches = sorted(p for p in self.output_directory.glob(str(spec["glob"])) if p.is_file())
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"Expected one output for {spec['glob']!r}, found {len(matches)} "
                    f"under {self.output_directory}. Use a more precise mapping."
                )
            path = matches[0].resolve()
        else:
            path = (self.output_directory / str(spec["path"])).resolve()
        if not path.is_relative_to(self.output_directory):
            raise ValueError("Output mappings must refer to files inside output_directory.")
        if not path.is_file():
            raise FileNotFoundError(f"Required nextnano output does not exist: {path}")
        try:
            from nextnanopy import DataFile
        except ImportError as exc:
            raise ImportError("Install nextnanopy to read nextnano output files.") from exc
        data = DataFile(str(path), product="nextnano++")
        if spec.get("require_single_coordinate") and len(data.coords) != 1:
            raise ValueError(f"Expected exactly one coordinate in {path}, found {len(data.coords)}.")
        if spec.get("require_single_variable") and len(data.variables) != 1:
            raise ValueError(f"Expected exactly one intensity variable in {path}, found {len(data.variables)}.")
        return data

    @staticmethod
    def _column(data: Any, selector: str | int, *, coordinate: bool = False) -> Any:
        if isinstance(selector, bool) or not isinstance(selector, (str, int)):
            raise ValueError("A column selector must be an exact name or zero-based index.")
        columns = data.coords if coordinate else data.variables
        try:
            return columns[selector]
        except (KeyError, IndexError) as exc:
            raise ValueError(
                f"Column {selector!r} was not found. Available columns: {list(columns.keys())}"
            ) from exc

    @staticmethod
    def _values(column: Any, name: str) -> np.ndarray:
        values = np.atleast_1d(np.asarray(column.value, dtype=float).squeeze())
        if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain a nonempty, finite one-dimensional array.")
        return values

    @staticmethod
    def _unit(column: Any, override: Any = None) -> str:
        unit = override if override is not None else getattr(column, "unit", "")
        return str(unit).strip().strip("[]")

    @staticmethod
    def _length_factor(unit: str) -> float:
        factors = {
            "nm": 1.0, "um": 1e3, "µm": 1e3, "μm": 1e3,
            "m": 1e9, "cm": 1e7, "mm": 1e6, "angstrom": 0.1,
            "Angstrom": 0.1, "Å": 0.1,
        }
        if unit not in factors:
            raise ValueError(f"Unsupported length unit {unit!r}; configure an explicit unit.")
        return factors[unit]

    @staticmethod
    def _energy_factor(unit: str) -> float:
        factors = {"eV": 1.0, "meV": 1e-3, "keV": 1e3, "J": 1 / 1.602176634e-19}
        if unit not in factors:
            raise ValueError(f"Unsupported energy unit {unit!r}; configure an explicit unit.")
        return factors[unit]

    @staticmethod
    def _sort_rows(axis: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
        if any(array.shape != axis.shape for array in arrays):
            raise ValueError("Output coordinates and values must have matching lengths.")
        if not np.all(np.isfinite(axis)) or any(not np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("Converted output coordinates and values must be finite.")
        order = np.argsort(axis)
        sorted_axis = axis[order]
        if sorted_axis.size > 1 and np.any(np.diff(sorted_axis) <= 0):
            raise ValueError("Output coordinates must be distinct.")
        return (sorted_axis, *(array[order] for array in arrays))

    def load_band_edges(self) -> dict[str, Any]:
        """Return position and band electron energies on the solver reference.

        If ``valence`` is a list of branch selectors, ``valence_ev`` is their
        pointwise maximum; the individual branches remain available in
        ``valence_components_ev``. This avoids assuming HH is always the top
        valence branch for every orientation or strain condition.
        """
        if "band_edges" in self._data:
            return self._data["band_edges"]
        spec = self._spec("band_edges")
        data = self._load_file(spec)
        x = self._column(data, spec["coordinate"], coordinate=True)
        conduction = self._column(data, spec["conduction"])
        position = self._values(x, "position") * self._length_factor(self._unit(x, spec.get("position_unit")))
        ec = self._values(conduction, "conduction band") * self._energy_factor(self._unit(conduction, spec.get("energy_unit")))
        selectors = spec["valence"] if isinstance(spec["valence"], list) else [spec["valence"]]
        if not selectors:
            raise ValueError("Configure at least one valence-band column.")
        valence_branches = []
        for selector in selectors:
            valence = self._column(data, selector)
            valence_branches.append(self._values(valence, f"valence band {selector}") * self._energy_factor(self._unit(valence, spec.get("energy_unit"))))
        sorted_rows = self._sort_rows(position, ec, *valence_branches)
        position, ec = sorted_rows[:2]
        components = {str(selector): branch for selector, branch in zip(selectors, sorted_rows[2:])}
        ev = np.max(np.vstack(sorted_rows[2:]), axis=0)
        result = {
            "position_nm": position, "conduction_ev": ec, "valence_ev": ev,
            "energy_reference": spec.get("energy_reference"),
            "conduction_label": str(spec["conduction"]),
            "valence_label": str(selectors[0]) if len(selectors) == 1 else f"max({', '.join(map(str, selectors))})",
            "valence_components_ev": components,
        }
        self._data["band_edges"] = result
        return result

    def load_states(self) -> dict[str, Any]:
        """Read explicitly identified conduction-like and valence-like states.

        For a coupled kp model, identify the groups from solver spinor output
        before configuring these selectors. A positive 'hole confinement energy'
        is not a valence electron energy and must not be supplied here.
        """
        if "states" in self._data:
            return self._data["states"]
        spec = self._spec("states")
        reference = spec.get("energy_reference")
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError("State mapping needs a nonempty shared 'energy_reference'.")
        result: dict[str, Any] = {"energy_reference": reference}
        for kind in ("electron", "valence"):
            source = spec.get(kind)
            if not isinstance(source, Mapping):
                raise ValueError(f"State mapping requires an explicit {kind!r} source.")
            if source.get("energy_reference", reference) != reference:
                raise ValueError("Electron and valence states must share the same energy reference.")
            data = self._load_file(source)
            column = self._column(data, source["variable"])
            energies = self._values(column, f"{kind} states") * self._energy_factor(self._unit(column, source.get("energy_unit")))
            if "indices" in source:
                indices = np.asarray(source["indices"])
                if indices.ndim != 1 or indices.size == 0 or indices.dtype.kind not in "iu":
                    raise ValueError("State indices must be a nonempty list of nonnegative integers.")
                if np.any(indices < 0) or np.any(indices >= energies.size):
                    raise ValueError("State index is outside the selected energy column.")
                energies = energies[indices]
            labels = source.get("labels", [f"{kind} {i + 1}" for i in range(energies.size)])
            if len(labels) != energies.size:
                raise ValueError("State labels and energies must have matching lengths.")
            result[f"{kind}_ev"] = energies
            result[f"{kind}_labels"] = list(labels)
        self._data["states"] = result
        return result

    def load_wavefunctions(self) -> dict[str, Any]:
        """Read explicitly configured scalar envelopes or probability densities.

        An eight-band spinor cannot be reduced to a scalar envelope by choosing
        an arbitrary component. Configure probability data for visualization or
        use a dedicated spinor treatment for physical overlap calculations.
        """
        if "wavefunctions" in self._data:
            return self._data["wavefunctions"]
        spec = self._spec("wavefunctions")
        result: dict[str, Any] = {}
        for kind in ("electron", "valence"):
            source = spec.get(kind)
            if source is None:
                continue
            data = self._load_file(source)
            x = self._column(data, source["coordinate"], coordinate=True)
            position = self._values(x, "position") * self._length_factor(self._unit(x, source.get("position_unit")))
            selectors = source.get("variables")
            if not isinstance(selectors, list) or not selectors:
                raise ValueError("Wavefunction sources require an explicit nonempty 'variables' list.")
            arrays = [self._values(self._column(data, item), f"wavefunction {item}") for item in selectors]
            sorted_rows = self._sort_rows(position, *arrays)
            representation = source.get("representation", "envelope")
            if representation not in ("envelope", "probability"):
                raise ValueError("Wavefunction representation must be 'envelope' or 'probability'.")
            if representation == "probability" and any(np.any(row < 0) for row in sorted_rows[1:]):
                raise ValueError("Probability densities must be nonnegative.")
            result[kind] = {
                "position_nm": sorted_rows[0], "values": np.vstack(sorted_rows[1:]),
                "representation": representation, "labels": [str(item) for item in selectors],
            }
        if not result:
            raise ValueError("No electron or valence wavefunction sources are configured.")
        self._data["wavefunctions"] = result
        return result

    def load_spectrum(self) -> dict[str, Any]:
        """Read a solver spectrum; convert density per eV to density per nm.

        An energy-axis spectrum requires ``spectral_density: true`` and its
        ``density_axis_unit``. The transformation uses |dE/dλ| = hc/λ², so its
        wavelength peak can differ from the energy-spectrum peak.
        """
        if "spectrum" in self._data:
            return self._data["spectrum"]
        spec = self._spec("spectrum")
        if spec.get("axis_container", "coords") not in ("coords", "variables"):
            raise ValueError("Spectrum axis_container must be 'coords' or 'variables'.")
        if "spectral_density" in spec and not isinstance(spec["spectral_density"], bool):
            raise ValueError("spectral_density must be a JSON boolean.")
        data = self._load_file(spec)
        axis_container = spec.get("axis_container", "coords")
        if axis_container not in ("coords", "variables"):
            raise ValueError("Spectrum axis_container must be 'coords' or 'variables'.")
        axis_column = self._column(data, spec["coordinate"], coordinate=axis_container == "coords")
        intensity_column = self._column(data, spec["intensity"])
        axis = self._values(axis_column, "spectrum axis")
        intensity = self._values(intensity_column, "emission intensity")
        if axis.shape != intensity.shape or np.any(axis <= 0) or np.any(intensity < 0):
            raise ValueError("Spectrum needs matching positive coordinates and nonnegative intensities.")
        unit = self._unit(axis_column, spec.get("axis_unit"))
        intensity_unit = self._unit(intensity_column, spec.get("intensity_unit"))
        kind = spec.get("axis_kind")
        conversion = "none"
        if kind == "wavelength":
            wavelength = axis * self._length_factor(unit)
            if spec.get("spectral_density"):
                density_unit = spec.get("density_axis_unit", unit)
                density_factor = self._length_factor(str(density_unit))
                if density_factor != 1 and not spec.get("wavelength_intensity_unit"):
                    raise ValueError("Wavelength density conversion requires an explicit 'wavelength_intensity_unit'.")
                intensity = intensity / density_factor
                intensity_unit = str(spec.get("wavelength_intensity_unit", intensity_unit))
                if density_factor != 1:
                    conversion = f"intensity_per_{density_unit} / {density_factor:g}"
        elif kind == "energy":
            if spec.get("spectral_density") is not True or not spec.get("density_axis_unit"):
                raise ValueError("Energy spectra require spectral_density=true and density_axis_unit to conserve density.")
            energy = axis * self._energy_factor(unit)
            wavelength = self.HC_EV_NM / energy
            intensity_per_ev = intensity / self._energy_factor(str(spec["density_axis_unit"]))
            intensity = intensity_per_ev * self.HC_EV_NM / wavelength**2
            intensity_unit = spec.get("wavelength_intensity_unit")
            if not isinstance(intensity_unit, str) or not intensity_unit:
                raise ValueError("Energy conversion requires an explicit 'wavelength_intensity_unit'.")
            conversion = "intensity_per_eV * hc_eV_nm / wavelength_nm**2"
        else:
            raise ValueError("Spectrum mapping axis_kind must be 'wavelength' or 'energy'.")
        if not intensity_unit:
            raise ValueError("The emission intensity unit must be present or explicitly configured.")
        wavelength, intensity = self._sort_rows(wavelength, intensity)
        result = {
            "wavelength_nm": wavelength, "intensity": intensity,
            "intensity_unit": intensity_unit, "axis": "wavelength", "axis_unit": "nm",
            "source_axis": kind, "source_axis_unit": unit,
            "spectral_density": bool(spec.get("spectral_density", False)),
            "conversion": conversion, "source": str(spec.get("path", spec.get("glob"))),
        }
        self._data["spectrum"] = result
        return result

    def plot_band_diagram(
        self,
        path: str | Path,
        structure: Any = None,
        include_states: bool = True,
        include_wavefunctions: bool = False,
    ) -> Path:
        """Save band edges with optional state levels and scaled wavefunctions.

        State lines are drawn inside the supplied well regions only. Their
        energy reference must match the band's declared reference.
        """
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        bands = self._data.get("band_edges") or self.load_band_edges()
        figure = Figure(figsize=(8, 5), layout="constrained")
        FigureCanvasAgg(figure)
        ax = figure.subplots()
        ax.plot(bands["position_nm"], bands["conduction_ev"], label=f"Conduction ({bands['conduction_label']})")
        ax.plot(bands["position_nm"], bands["valence_ev"], label=f"Valence ({bands['valence_label']})")
        regions = [] if structure is None else structure.get_well_regions()
        well_ranges: list[tuple[float, float]] = []
        for region in regions:
            start, end = (region["start_nm"], region["end_nm"]) if isinstance(region, Mapping) else region[:2]
            well_ranges.append((float(start), float(end)))
            ax.axvspan(start, end, color="gray", alpha=0.12)
        states = None
        if include_states or include_wavefunctions:
            if not well_ranges:
                raise ValueError("Provide a structure with well regions to overlay state energies.")
            states = self._data.get("states") or self.load_states()
            if not bands.get("energy_reference") or states["energy_reference"] != bands["energy_reference"]:
                raise ValueError("Band and state overlays require matching declared energy references.")
            if include_states:
                for kind, color in (("electron", "tab:blue"), ("valence", "tab:orange")):
                    for energy, label in zip(states[f"{kind}_ev"], states[f"{kind}_labels"]):
                        for index, (start, end) in enumerate(well_ranges):
                            ax.hlines(energy, start, end, color=color, linestyle="--", alpha=0.7, label=label if index == 0 else None)
        if include_wavefunctions:
            waves = self._data.get("wavefunctions") or self.load_wavefunctions()
            span = float(np.ptp(np.concatenate((bands["conduction_ev"], bands["valence_ev"]))))
            scale = 0.04 * max(span, 1.0)
            for kind, group in waves.items():
                energies = states[f"{kind}_ev"]
                if len(energies) != len(group["values"]):
                    raise ValueError("Each wavefunction must correspond to one configured state energy.")
                for values, energy in zip(group["values"], energies):
                    peak = float(np.max(np.abs(values)))
                    if peak == 0:
                        raise ValueError("A zero wavefunction cannot be scaled for visualization.")
                    ax.plot(group["position_nm"], energy + scale * values / peak, linewidth=1, alpha=0.8)
            ax.text(0.02, 0.02, "Wavefunction/probability display uses arbitrary vertical scaling", transform=ax.transAxes, fontsize=8)
        ax.set(xlabel="Position (nm)", ylabel="Energy (eV)", title="Single quantum well")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=180)
        return destination
