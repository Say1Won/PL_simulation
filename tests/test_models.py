"""Validate physical inputs and reject unsupported native-model controls."""

import math
import unittest

from pl_simulation.layer import Layer
from pl_simulation.single_qw_structure import SingleQWStructure
from pl_simulation.simulation_settings import SimulationSettings


def structure_data():
    return {
        "layers": [
            {"name": "left", "material": "GaN", "thickness_nm": 12.0, "role": "barrier"},
            {"name": "well", "material": "InGaN", "thickness_nm": 3.0,
             "role": "well", "indium_fraction": 0.2},
            {"name": "right", "material": "InGaN", "thickness_nm": 8.0,
             "role": "barrier", "indium_fraction": 0.05},
        ]
    }


class LayerBoundaryTests(unittest.TestCase):
    def test_nonphysical_thickness_and_compositions_are_rejected(self):
        for thickness in (0, -1, math.nan, math.inf, True):
            with self.subTest(thickness=thickness), self.assertRaises(ValueError):
                Layer("well", "InGaN", thickness, "well", 0.2)
        for fraction in (-0.01, 1.01, math.nan, math.inf, True):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                Layer("well", "InGaN", 3, "well", fraction)

    def test_pure_gan_cannot_silently_contain_indium(self):
        with self.assertRaises(ValueError):
            Layer("barrier", "GaN", 10, "barrier", 0.1)

    def test_unsupported_material_and_misspelled_config_are_rejected(self):
        with self.assertRaises(ValueError):
            Layer("barrier", "Si", 10, "barrier")
        with self.assertRaises(ValueError):
            Layer.from_dict({"name": "well", "material": "InGaN",
                             "thickness_nm": 3, "role": "well", "indium": 0.2})


class SingleQWBoundaryTests(unittest.TestCase):
    def test_asymmetric_structure_keeps_boundaries_and_well_region(self):
        structure = SingleQWStructure.from_dict(structure_data())
        ranges = structure.get_layer_ranges()
        self.assertEqual([(r["start_nm"], r["end_nm"]) for r in ranges],
                         [(0, 12), (12, 15), (15, 23)])
        self.assertEqual(structure.get_well_regions(), [ranges[1]])
        self.assertEqual(ranges[2]["indium_fraction"], 0.05)
        self.assertEqual(ranges[1]["thickness_nm"], 3)

    def test_equal_or_higher_indium_barrier_is_not_a_supported_well(self):
        for fraction in (0.2, 0.3):
            config = structure_data()
            config["layers"][2]["indium_fraction"] = fraction
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                SingleQWStructure.from_dict(config)

    def test_multiwell_or_wrong_order_is_rejected_in_singlewell_class(self):
        config = structure_data()
        config["layers"][0]["role"] = "well"
        with self.assertRaises(ValueError):
            SingleQWStructure.from_dict(config)
        config = structure_data()
        config["layers"].append(config["layers"][1])
        with self.assertRaises(ValueError):
            SingleQWStructure.from_dict(config)


class SettingsBoundaryTests(unittest.TestCase):
    def test_positive_density_defaults_and_legacy_occupation_migration(self):
        settings = SimulationSettings.from_dict({})
        self.assertGreater(settings.electron_sheet_density_cm2, 0)
        self.assertGreater(settings.hole_sheet_density_cm2, 0)
        for name in ("electron_sheet_density_cm2", "hole_sheet_density_cm2"):
            for density in (0, -1, math.nan, math.inf, True):
                with self.subTest(name=name, density=density), self.assertRaises(ValueError):
                    SimulationSettings(**{name: density})
        with self.assertRaisesRegex(ValueError, "sheet_density"):
            SimulationSettings.from_dict({"electron_fermi_ev": 3, "hole_fermi_ev": 0})

    def test_zero_parallel_and_invalid_orientation_normals(self):
        for x, y in (((0, 0, 0), (1, 0, 0)),
                     ((0, 0, 1), (0, 0, -2)),
                     ((1, 2, 3), (2, 4, 6)),
                     ((0.0, 0, 1), (1, 0, 0)),
                     ((0, 1), (1, 0, 0))):
            with self.subTest(x=x, y=y), self.assertRaises(ValueError):
                SimulationSettings(x_hkl=x, y_hkl=y)

    def test_valid_orientation_and_json_roundtrip_preserve_controls(self):
        settings = SimulationSettings.from_dict({
            "electron_sheet_density_cm2": 2e12, "hole_sheet_density_cm2": 8e11,
            "x_hkl": [0, 0, -1], "y_hkl": [0, 1, 0],
            "electric_field_kv_cm": -150,
        })
        self.assertEqual(SimulationSettings.from_dict(settings.to_dict()), settings)
        self.assertEqual(settings.x_hkl, (0, 0, -1))
        self.assertEqual(settings.electric_field_kv_cm, -150)

    def test_absent_strain_polarization_and_orientation_physics_are_rejected(self):
        for override in ({"include_strain": True}, {"include_polarization": True},
                         {"x_hkl": (1, 0, 0)}, {"x_hkl": (1, 0, 1)},
                         {"y_hkl": (1, 0, 1)}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                SimulationSettings(**override)

    def test_invalid_spectral_grid_and_nonfinite_controls_are_rejected(self):
        invalid = (
            {"spectrum_energy_min_ev": 4, "spectrum_energy_max_ev": 2},
            {"spectrum_energy_step_ev": 3},
            {"broadening_ev": 0}, {"grid_spacing_nm": math.nan},
            {"electron_states": 1.5}, {"include_strain": 1},
            {"electric_field_kv_cm": math.inf}, {"conduction_band_offset_ratio": 0},
            {"conduction_band_offset_ratio": 1}, {"k_integration_points": 1},
        )
        for override in invalid:
            with self.subTest(override=override), self.assertRaises(ValueError):
                SimulationSettings(**override)


if __name__ == "__main__":
    unittest.main()
