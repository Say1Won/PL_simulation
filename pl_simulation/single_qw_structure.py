"""Ordered barrier/well/barrier structure and spatial layer boundaries."""

from dataclasses import dataclass
from typing import Any, Mapping

from .layer import Layer


@dataclass(frozen=True)
class SingleQWStructure:
    """Describe a central InGaN layer, including non-confining compositions."""

    layers: tuple[Layer, ...]

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "layers", tuple(self.layers))
        except TypeError as exc:
            raise ValueError("layers must be an ordered sequence of Layer objects.") from exc
        self.validate()

    def validate(self) -> None:
        """Validate the layer layout without requiring bound states to exist."""
        if len(self.layers) != 3 or not all(isinstance(layer, Layer) for layer in self.layers):
            raise ValueError("Single QW requires exactly three Layer objects.")
        for layer in self.layers:
            layer.validate()
        if tuple(layer.role for layer in self.layers) != ("barrier", "well", "barrier"):
            raise ValueError("Layer order must be barrier, well, barrier.")
        if len({layer.name for layer in self.layers}) != 3:
            raise ValueError("Layer names must be unique.")
        well = self.layers[1]
        if well.material != "InGaN":
            raise ValueError("The central well layer must use the InGaN alloy model.")

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
