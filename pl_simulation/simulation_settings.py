"""Physical controls for the bundled 1D wurtzite optical calculation."""

from dataclasses import asdict, dataclass, fields
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class SimulationSettings:
    """Use nm/eV/K and explicitly prescribe electron/hole quasi-Fermi energies.

    Quasi-Fermi energies use the same energy reference as nextnano's band
    output; they are occupation inputs, not a conversion from laser power.
    x_hkl and y_hkl are the reduced three-index lattice-plane normals of
    crystal_wz. The solver uses the substrate lattice metric to construct
    the coordinate system; this class only rejects zero/parallel normals.
    """

    electron_fermi_ev: float
    hole_fermi_ev: float
    substrate: str = "GaN"
    temperature_k: float = 300.0
    x_hkl: tuple[int, int, int] = (0, 0, 1)
    y_hkl: tuple[int, int, int] = (1, 0, 0)
    grid_spacing_nm: float = 0.1
    electron_states: int = 4
    hole_states: int = 12
    spectrum_energy_min_ev: float = 2.0
    spectrum_energy_max_ev: float = 4.0
    spectrum_energy_step_ev: float = 0.002
    broadening_ev: float = 0.01
    include_strain: bool = True
    include_polarization: bool = True

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "x_hkl", tuple(self.x_hkl))
            object.__setattr__(self, "y_hkl", tuple(self.y_hkl))
        except TypeError as exc:
            raise ValueError("x_hkl and y_hkl must contain three integer indices.") from exc
        self.validate()

    def validate(self) -> None:
        """Validate supported controls without duplicating the solver physics."""
        if self.substrate != "GaN":
            raise ValueError("The bundled template currently supports a GaN substrate only.")
        positive = (
            "temperature_k", "grid_spacing_nm", "spectrum_energy_min_ev",
            "spectrum_energy_max_ev", "spectrum_energy_step_ev", "broadening_ev",
        )
        for name in positive + ("electron_fermi_ev", "hole_fermi_ev"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number.")
            if name in positive and value <= 0:
                raise ValueError(f"{name} must be positive.")
        span = self.spectrum_energy_max_ev - self.spectrum_energy_min_ev
        if span <= 0:
            raise ValueError("spectrum_energy_max_ev must exceed spectrum_energy_min_ev.")
        if self.spectrum_energy_step_ev > span:
            raise ValueError("spectrum_energy_step_ev cannot exceed the energy range.")
        if self.electron_fermi_ev <= self.hole_fermi_ev:
            raise ValueError("For the emission calculation electron_fermi_ev must exceed hole_fermi_ev.")
        for name in ("electron_states", "hole_states"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("include_strain", "include_polarization"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be true or false.")
        for name in ("x_hkl", "y_hkl"):
            indices = getattr(self, name)
            if len(indices) != 3 or any(isinstance(i, bool) or not isinstance(i, int) for i in indices):
                raise ValueError(f"{name} must contain exactly three integers.")
            if not any(indices):
                raise ValueError(f"{name} cannot be the zero normal.")
        x, y = self.x_hkl, self.y_hkl
        cross = (x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0])
        if not any(cross):
            raise ValueError("x_hkl and y_hkl cannot be parallel normals.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SimulationSettings":
        """Build from the 'settings' object in configs/simulation.json."""
        if not isinstance(data, Mapping):
            raise ValueError("Simulation settings must be a JSON object.")
        unknown = set(data) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown simulation setting keys: {sorted(unknown)}")
        required = {"electron_fermi_ev", "hole_fermi_ev"}
        missing = required - set(data)
        if missing:
            raise ValueError(f"Explicit quasi-Fermi occupation inputs are required: {sorted(missing)}")
        return cls(**dict(data))

    def to_input_variables(self) -> dict[str, int | float]:
        """Return names declared by templates/single_qw_pl.nnp."""
        return {
            "TEMPERATURE": self.temperature_k,
            "X_H": self.x_hkl[0], "X_K": self.x_hkl[1], "X_L": self.x_hkl[2],
            "Y_H": self.y_hkl[0], "Y_K": self.y_hkl[1], "Y_L": self.y_hkl[2],
            "GRID_SPACING": self.grid_spacing_nm,
            "ELECTRON_STATES": self.electron_states,
            "HOLE_STATES": self.hole_states,
            "ELECTRON_FERMI": self.electron_fermi_ev,
            "HOLE_FERMI": self.hole_fermi_ev,
            "ENERGY_MIN": self.spectrum_energy_min_ev,
            "ENERGY_MAX": self.spectrum_energy_max_ev,
            "ENERGY_STEP": self.spectrum_energy_step_ev,
            "BROADENING": self.broadening_ev,
            "STRAIN_ENABLED": int(self.include_strain),
            "POLARIZATION_ENABLED": int(self.include_polarization),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe controls for a reproducible run snapshot."""
        data = asdict(self)
        data["x_hkl"] = list(self.x_hkl)
        data["y_hkl"] = list(self.y_hkl)
        return data
