"""Ordered barrier/well/barrier structure and spatial layer boundaries."""

from dataclasses import dataclass
from typing import Any, Mapping

from .layer import Layer


@dataclass(frozen=True)
class SingleQWStructure:
    """Describe one InGaN well without storing material database values."""

    layers: tuple[Layer, ...]

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "layers", tuple(self.layers))
        except TypeError as exc:
            raise ValueError("layers must be an ordered sequence of Layer objects.") from exc
        self.validate()

    def validate(self) -> None:
        """Require exactly one well between two lower-indium barriers."""
        if len(self.layers) != 3 or not all(isinstance(layer, Layer) for layer in self.layers):
            raise ValueError("Single QW requires exactly three Layer objects.")
        for layer in self.layers:
            layer.validate()
        if tuple(layer.role for layer in self.layers) != ("barrier", "well", "barrier"):
            raise ValueError("Layer order must be barrier, well, barrier.")
        if len({layer.name for layer in self.layers}) != 3:
            raise ValueError("Layer names must be unique.")
        well = self.layers[1]
        if well.material != "InGaN" or well.indium_fraction <= 0:
            raise ValueError("The well must be InGaN with positive indium_fraction.")
        if any(layer.indium_fraction >= well.indium_fraction for layer in (self.layers[0], self.layers[2])):
            raise ValueError("Each barrier must have a lower In fraction than the well.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SingleQWStructure":
        """Read an object containing an ordered 'layers' list."""
        if not isinstance(data, Mapping) or set(data) != {"layers"}:
            raise ValueError("Structure configuration must contain only a 'layers' list.")
        if not isinstance(data["layers"], (list, tuple)):
            raise ValueError("Structure 'layers' must be a list.")
        return cls(tuple(Layer.from_dict(entry) for entry in data["layers"]))

    def get_layer_ranges(self) -> list[dict[str, Any]]:
        """Return layer boundaries in nm, beginning at x = 0."""
        ranges = []
        start_nm = 0.0
        for layer in self.layers:
            end_nm = start_nm + layer.thickness_nm
            ranges.append({**layer.to_dict(), "start_nm": start_nm, "end_nm": end_nm})
            start_nm = end_nm
        return ranges

    def get_well_regions(self) -> list[dict[str, Any]]:
        """Return a list even for SQW, so plotting can later support MQW."""
        return [region for region in self.get_layer_ranges() if region["role"] == "well"]

    def to_dict(self) -> dict[str, Any]:
        """Return the complete structure for an input snapshot."""
        return {"layers": [layer.to_dict() for layer in self.layers]}
