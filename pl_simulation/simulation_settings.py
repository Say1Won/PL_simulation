"""Controls for the standalone Python single-quantum-well calculation."""

from dataclasses import asdict, dataclass, fields
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class SimulationSettings:
    """Use nm, eV, K, cm^-2 and kV/cm with no external solver inputs.

    Electron and hole sheet densities determine their occupations internally;
    they are not a conversion from laser power. The current scalar effective-
    mass model supports a GaN substrate and c-plane growth only. Strain and
    automatic polarization are unsupported; a signed uniform electric field
    can be specified explicitly. ``broadening_ev`` is the Gaussian standard
    deviation in photon energy, rather than its FWHM.
    """

    temperature_k: float = 300.0
    x_hkl: tuple[int, int, int] = (0, 0, 1)
    y_hkl: tuple[int, int, int] = (1, 0, 0)
    grid_spacing_nm: float = 0.1
    electron_states: int = 4
    hole_states: int = 8
    electron_sheet_density_cm2: float = 1e12
    hole_sheet_density_cm2: float = 1e12
    conduction_band_offset_ratio: float = 0.7
    electric_field_kv_cm: float = 0.0
    spectrum_energy_min_ev: float = 2.0
    spectrum_energy_max_ev: float = 4.0
    spectrum_energy_step_ev: float = 0.002
    broadening_ev: float = 0.01
    k_integration_points: int = 256
    include_strain: bool = False
    include_polarization: bool = False
    substrate: str = "GaN"

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "x_hkl", tuple(self.x_hkl))
            object.__setattr__(self, "y_hkl", tuple(self.y_hkl))
        except TypeError as exc:
            raise ValueError("x_hkl and y_hkl must contain three integer indices.") from exc
        self.validate()

    def validate(self) -> None:
        """Reject invalid controls and physics absent from the native model."""
        if self.substrate != "GaN":
            raise ValueError("The standalone model currently supports a GaN substrate only.")
        positive = (
            "temperature_k", "grid_spacing_nm", "electron_sheet_density_cm2",
            "hole_sheet_density_cm2", "spectrum_energy_min_ev",
            "spectrum_energy_max_ev", "spectrum_energy_step_ev", "broadening_ev",
        )
        for name in positive + ("electric_field_kv_cm", "conduction_band_offset_ratio"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number.")
            if name in positive and value <= 0:
                raise ValueError(f"{name} must be positive.")
        if not 0 < self.conduction_band_offset_ratio < 1:
            raise ValueError("conduction_band_offset_ratio must lie strictly between 0 and 1.")
        span = self.spectrum_energy_max_ev - self.spectrum_energy_min_ev
        if span <= 0:
            raise ValueError("spectrum_energy_max_ev must exceed spectrum_energy_min_ev.")
        if self.spectrum_energy_step_ev > span:
            raise ValueError("spectrum_energy_step_ev cannot exceed the energy range.")
        for name in ("electron_states", "hole_states", "k_integration_points"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.k_integration_points < 32:
            raise ValueError("k_integration_points must be at least 32.")
        for name in ("include_strain", "include_polarization"):
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be true or false.")
            if value:
                raise ValueError(
                    f"{name}=true is unsupported by the standalone model. "
                    "Set it to false; electric_field_kv_cm supplies a uniform field explicitly."
                )
        for name in ("x_hkl", "y_hkl"):
            indices = getattr(self, name)
            if len(indices) != 3 or any(isinstance(i, bool) or not isinstance(i, int) for i in indices):
                raise ValueError(f"{name} must contain exactly three integers.")
        if self.x_hkl not in ((0, 0, 1), (0, 0, -1)):
            raise ValueError("The standalone model supports c-plane growth only: x_hkl must be [0, 0, 1] or [0, 0, -1].")
        if self.y_hkl[2] != 0 or not any(self.y_hkl[:2]):
            raise ValueError("y_hkl must be a nonzero basal-plane direction with its third index equal to zero.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SimulationSettings":
        """Build from the native 'settings' object, rejecting legacy inputs."""
        if not isinstance(data, Mapping):
            raise ValueError("Simulation settings must be a JSON object.")
        unknown = set(data) - {field.name for field in fields(cls)}
        if unknown:
            message = f"Unknown simulation setting keys: {sorted(str(key) for key in unknown)}."
            if unknown & {"electron_fermi_ev", "hole_fermi_ev"}:
                message += (
                    " Prescribed quasi-Fermi inputs belong to the removed external-solver model; "
                    "use electron_sheet_density_cm2 and hole_sheet_density_cm2 instead."
                )
            raise ValueError(message)
        return cls(**dict(data))

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe controls for a reproducible native run snapshot."""
        data = asdict(self)
        data["x_hkl"] = list(self.x_hkl)
        data["y_hkl"] = list(self.y_hkl)
        return data
