"""A single layer of a wurtzite GaN/InGaN heterostructure."""

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class Layer:
    """Store layer thickness in nm and indium mole fraction between 0 and 1."""

    name: str
    material: str
    thickness_nm: float
    role: str
    indium_fraction: float = 0.0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Reject invalid material, role, composition, or thickness."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Layer name must be a nonempty string.")
        if not isinstance(self.material, str) or self.material not in {"GaN", "InGaN"}:
            raise ValueError("Supported layer materials are GaN and InGaN.")
        if not isinstance(self.role, str) or self.role not in {"well", "barrier"}:
            raise ValueError("Layer role must be 'well' or 'barrier'.")
        for label, value in (
            ("thickness_nm", self.thickness_nm),
            ("indium_fraction", self.indium_fraction),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{label} must be a finite number.")
            if not math.isfinite(value):
                raise ValueError(f"{label} must be a finite number.")
        if self.thickness_nm <= 0:
            raise ValueError("Layer thickness_nm must be positive.")
        if not 0 <= self.indium_fraction <= 1:
            raise ValueError("indium_fraction must lie between 0 and 1.")
        if self.material == "GaN" and self.indium_fraction != 0:
            raise ValueError("GaN must have indium_fraction = 0.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Layer":
        """Build a layer from one entry of configs/single_qw.json."""
        if not isinstance(data, Mapping):
            raise ValueError("Each layer must be a JSON object.")
        allowed = {"name", "material", "thickness_nm", "role", "indium_fraction"}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"Unknown layer keys: {sorted(unknown)}")
        required = {"name", "material", "thickness_nm", "role"}
        missing = required - set(data)
        if missing:
            raise ValueError(f"Missing layer keys: {sorted(missing)}")
        return cls(**dict(data))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable layer snapshot."""
        return asdict(self)
