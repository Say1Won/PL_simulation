"""Validate native numerical results and their pickle-free storage contract."""

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch

import numpy as np

from pl_simulation.qw_results import QWResults


def calculation_fixture():
    """Arithmetic data for storage/unit tests; not a physical PL prediction."""
    position = np.linspace(0, 3, 61)
    envelope = np.sin(np.pi * position / 3)
    second_envelope = np.sin(2 * np.pi * position / 3)
    energy = np.linspace(2, 4, 1001)
    return {
        "position_nm": position,
        "conduction_ev": np.full(position.size, 3.4),
        "valence_ev": np.zeros(position.size),
        "electron_energies_ev": np.array([3.5, 3.7]),
        "valence_energies_ev": np.array([-0.1, -0.3]),
        "electron_wavefunctions": np.vstack((envelope, second_envelope)),
        "hole_wavefunctions": np.vstack((envelope, second_envelope)),
        "energy_ev": energy,
        "energy_intensity": np.ones_like(energy),
        "bound_state_mask": np.array([True, False]),
        "transition_matrix": np.array([[1 + 2j, 0], [0, 3 - 4j]]),
        "metadata": {
            "energy_reference": "GaN_unstrained_VBM",
            "energy_intensity_unit": "relative/eV",
            "model": "arithmetic_test_fixture",
            "nested": {"temperature_k": 300, "orientation": [0, 0, 1]},
        },
    }


class NativeResultsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def test_archive_roundtrip_preserves_numerics_metadata_and_shared_reference(self):
        calculation = calculation_fixture()
        result = QWResults.from_calculation(calculation)
        archive = result.save_calculation(self.directory / "result.npz")
        restored = QWResults.from_archive(archive)

        self.assertEqual(restored.metadata, calculation["metadata"])
        np.testing.assert_array_equal(restored.load_band_edges()["position_nm"],
                                      calculation["position_nm"])
        np.testing.assert_array_equal(restored.load_band_edges()["conduction_ev"],
                                      calculation["conduction_ev"])
        np.testing.assert_array_equal(restored.load_band_edges()["valence_ev"],
                                      calculation["valence_ev"])
        states = restored.load_states()
        self.assertEqual(states["energy_reference"],
                         restored.load_band_edges()["energy_reference"])
        for kind, energy_key, wave_key in (
            ("electron", "electron_energies_ev", "electron_wavefunctions"),
            ("valence", "valence_energies_ev", "hole_wavefunctions"),
        ):
            np.testing.assert_array_equal(states[f"{kind}_ev"], calculation[energy_key])
            np.testing.assert_array_equal(restored.load_wavefunctions()[kind]["values"],
                                          calculation[wave_key])
        with np.load(archive, allow_pickle=False) as raw:
            self.assertEqual(raw["metadata_json"].shape, ())
            self.assertEqual(raw["metadata_json"].dtype.kind, "U")
            self.assertEqual(json.loads(str(raw["metadata_json"])), calculation["metadata"])
            for key in ("energy_ev", "energy_intensity", "bound_state_mask", "transition_matrix"):
                np.testing.assert_array_equal(raw[key], calculation[key])
        self.assertEqual(restored.load_spectrum()["source"], str(archive.resolve()))

    def test_factory_and_metadata_do_not_alias_caller_values(self):
        calculation = calculation_fixture()
        expected_position = calculation["position_nm"].copy()
        result = QWResults.from_calculation(calculation)
        calculation["position_nm"][0] = -100
        calculation["metadata"]["nested"]["temperature_k"] = 999
        exported_metadata = result.metadata
        exported_metadata["nested"]["temperature_k"] = 1
        np.testing.assert_array_equal(result.load_band_edges()["position_nm"], expected_position)
        self.assertEqual(result.metadata["nested"]["temperature_k"], 300)

    def test_native_wavelength_density_conserves_energy_spectrum_signal(self):
        calculation = calculation_fixture()
        result = QWResults.from_calculation(calculation)
        spectrum = result.load_spectrum()
        wavelength = spectrum["wavelength_nm"]
        intensity = spectrum["intensity"]
        self.assertTrue(np.all(np.diff(wavelength) > 0))
        self.assertEqual(spectrum["intensity_unit"], "relative/nm")
        self.assertEqual(spectrum["source_axis"], "energy")
        # Constant density per eV must be transformed, not merely relabelled.
        np.testing.assert_allclose(intensity, QWResults.HC_EV_NM / wavelength**2)
        energy_integral = float(np.trapezoid(calculation["energy_intensity"],
                                            calculation["energy_ev"]))
        wavelength_integral = float(np.trapezoid(intensity, wavelength))
        self.assertAlmostEqual(wavelength_integral, energy_integral, places=5)
        self.assertGreater(intensity[0], intensity[-1])

    def test_native_factory_archive_and_loaders_work_without_nextnanopy(self):
        with patch.dict("sys.modules", {"nextnanopy": None}):
            result = QWResults.from_calculation(calculation_fixture())
            archive = result.save_calculation(self.directory / "result.npz")
            restored = QWResults.from_archive(archive)
            for loader in (restored.load_band_edges, restored.load_states,
                           restored.load_wavefunctions, restored.load_spectrum):
                self.assertTrue(loader())

    def test_object_metadata_or_array_is_rejected_without_pickle(self):
        calculation = calculation_fixture()
        payload = {key: value for key, value in calculation.items() if key != "metadata"}
        payload.update(metadata_json=np.asarray(json.dumps(calculation["metadata"])),
                       format_version=np.asarray(1, dtype=np.int64))
        for key in ("metadata_json", "electron_wavefunctions"):
            with self.subTest(key=key):
                malformed = dict(payload)
                malformed[key] = np.asarray({"object": "must not deserialize"}, dtype=object)
                archive = self.directory / f"object_{key}.npz"
                np.savez(archive, **malformed)
                with self.assertRaisesRegex(ValueError, "Invalid calculation archive"):
                    QWResults.from_archive(archive)

    def test_archive_rejects_corrupt_data_and_unknown_version(self):
        corrupt = self.directory / "corrupt.npz"
        corrupt.write_bytes(b"not a NumPy archive")
        with self.assertRaisesRegex(ValueError, "Invalid calculation archive"):
            QWResults.from_archive(corrupt)
        calculation = calculation_fixture()
        payload = {key: value for key, value in calculation.items() if key != "metadata"}
        payload.update(metadata_json=np.asarray(json.dumps(calculation["metadata"])),
                       format_version=np.asarray(99, dtype=np.int64))
        unknown = self.directory / "unknown_version.npz"
        np.savez(unknown, **payload)
        with self.assertRaisesRegex(ValueError, "format_version"):
            QWResults.from_archive(unknown)

    def test_factory_rejects_mismatched_states_or_wavefunction_positions(self):
        for key, replacement in (
            ("electron_energies_ev", np.array([3.5])),
            ("hole_wavefunctions", np.ones((2, 60))),
            ("electron_wavefunctions", np.zeros((2, 61))),
        ):
            with self.subTest(key=key):
                calculation = calculation_fixture()
                calculation[key] = replacement
                with self.assertRaises(ValueError):
                    QWResults.from_calculation(calculation)

    def test_factory_rejects_nonfinite_values_bad_axes_and_negative_spectrum(self):
        invalid_cases = (
            ("conduction_ev", np.full(61, np.nan)),
            ("position_nm", np.linspace(3, 0, 61)),
            ("energy_ev", np.linspace(-1, 1, 1001)),
            ("energy_intensity", -np.ones(1001)),
        )
        for key, replacement in invalid_cases:
            with self.subTest(key=key):
                calculation = calculation_fixture()
                calculation[key] = replacement
                with self.assertRaises(ValueError):
                    QWResults.from_calculation(calculation)
        calculation = calculation_fixture()
        calculation["metadata"]["nested"]["temperature_k"] = float("nan")
        with self.assertRaisesRegex(ValueError, "JSON-compatible"):
            QWResults.from_calculation(calculation)

    def test_saving_does_not_overwrite_an_existing_calculation(self):
        result = QWResults.from_calculation(calculation_fixture())
        archive = result.save_calculation(self.directory / "result.npz")
        saved_bytes = archive.read_bytes()
        with self.assertRaises(FileExistsError):
            result.save_calculation(archive)
        self.assertEqual(archive.read_bytes(), saved_bytes)
        restored = QWResults.from_archive(archive)
        self.assertEqual(restored.metadata["model"], "arithmetic_test_fixture")


if __name__ == "__main__":
    unittest.main()
