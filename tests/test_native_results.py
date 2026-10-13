"""Validate native numerical results and their pickle-free storage contract."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
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
            "energy_intensity_unit": "a.u.",
            "energy_density_axis_unit": "eV",
            "model": "arithmetic_test_fixture",
            "nested": {"temperature_k": 300, "orientation": [0, 0, 1]},
        },
    }


def no_signal_fixture(*, empty_kinds=("electron", "valence"), status="no_bound_transitions"):
    """Native no-signal result retaining only the states that actually exist."""
    calculation = calculation_fixture()
    for kind, energy_key, wave_key in (
        ("electron", "electron_energies_ev", "electron_wavefunctions"),
        ("valence", "valence_energies_ev", "hole_wavefunctions"),
    ):
        if kind in empty_kinds:
            calculation[energy_key] = np.empty(0)
            calculation[wave_key] = np.empty((0, len(calculation["position_nm"])))
    calculation["energy_intensity"] = np.zeros_like(calculation["energy_ev"])
    calculation["metadata"].update(pl_status=status, pl_message="No bound-state PL signal in this calculation.")
    return calculation


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

    def test_normalized_native_spectrum_retains_scale_and_conserves_raw_signal(self):
        calculation = calculation_fixture()
        result = QWResults.from_calculation(calculation)
        spectrum = result.load_spectrum()
        wavelength = spectrum["wavelength_nm"]
        intensity = spectrum["intensity"]
        self.assertTrue(np.all(np.diff(wavelength) > 0))
        self.assertEqual(spectrum["intensity_unit"], "a.u.")
        self.assertEqual(spectrum["source_axis"], "energy")
        self.assertEqual(float(np.max(intensity)), 1)
        reference = spectrum["normalization_reference_intensity"]
        raw_intensity = intensity * reference
        # Constant density per eV must be transformed, not merely relabelled.
        np.testing.assert_allclose(raw_intensity, QWResults.HC_EV_NM / wavelength**2)
        energy_integral = float(np.trapezoid(calculation["energy_intensity"],
                                            calculation["energy_ev"]))
        wavelength_integral = float(np.trapezoid(raw_intensity, wavelength))
        self.assertAlmostEqual(wavelength_integral, energy_integral, places=5)
        self.assertGreater(intensity[0], intensity[-1])
        again = result.load_spectrum()
        self.assertEqual(again["normalization_reference_intensity"], reference)
        np.testing.assert_array_equal(again["intensity"], intensity)

    def test_native_factory_archive_and_loaders_work_without_nextnanopy(self):
        with patch.dict("sys.modules", {"nextnanopy": None}):
            result = QWResults.from_calculation(calculation_fixture())
            archive = result.save_calculation(self.directory / "result.npz")
            restored = QWResults.from_archive(archive)
            for loader in (restored.load_band_edges, restored.load_states,
                           restored.load_wavefunctions, restored.load_spectrum):
                self.assertTrue(loader())

    def test_old_relative_archive_units_are_upgraded_without_changing_signal(self):
        current = QWResults.from_calculation(calculation_fixture()).load_spectrum()
        for old_unit in ("relative/eV", "a.u./eV"):
            with self.subTest(unit=old_unit):
                calculation = calculation_fixture()
                calculation["metadata"]["energy_intensity_unit"] = old_unit
                del calculation["metadata"]["energy_density_axis_unit"]
                result = QWResults.from_calculation(calculation)
                archive = result.save_calculation(self.directory / f"upgraded_{old_unit.split('/')[0]}.npz")
                restored = QWResults.from_archive(archive)
                self.assertEqual(restored.metadata["energy_intensity_unit"], "a.u.")
                self.assertEqual(restored.metadata["energy_density_axis_unit"], "eV")
                np.testing.assert_array_equal(restored.load_spectrum()["intensity"], current["intensity"])

    def test_arbitrary_unit_energy_spectrum_rejects_an_incorrect_density_axis(self):
        calculation = calculation_fixture()
        calculation["metadata"]["energy_density_axis_unit"] = "meV"
        with self.assertRaisesRegex(ValueError, "energy_density_axis_unit"):
            QWResults.from_calculation(calculation)

    def test_cached_relative_spectrum_is_normalized_and_physical_units_preserved(self):
        for old_unit, expected_unit in (
            ("relative/nm", "a.u."), ("a.u./nm", "a.u."), ("W/nm", "W/nm"),
        ):
            with self.subTest(unit=old_unit):
                result = QWResults.from_calculation(calculation_fixture())
                spectrum = result.load_spectrum()
                spectrum["intensity_unit"] = old_unit
                original_intensity = spectrum["intensity"].copy()
                loaded = result.load_spectrum()
                self.assertEqual(loaded["intensity_unit"], expected_unit)
                expected = original_intensity / max(original_intensity) if expected_unit == "a.u." else original_intensity
                np.testing.assert_array_equal(loaded["intensity"], expected)

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

    def test_no_signal_archive_preserves_empty_states_wavefunctions_and_status(self):
        for index, empty_kinds in enumerate((("electron", "valence"), ("electron",), ("valence",), ())):
            with self.subTest(empty_kinds=empty_kinds):
                status = "no_bound_transitions" if empty_kinds else "no_signal_in_window"
                calculation = no_signal_fixture(empty_kinds=empty_kinds, status=status)
                result = QWResults.from_calculation(calculation)
                archive = result.save_calculation(self.directory / f"no_signal_{index}.npz")
                restored = QWResults.from_archive(archive)
                self.assertEqual(restored.metadata, calculation["metadata"])
                for kind, energy_key, wave_key in (
                    ("electron", "electron_energies_ev", "electron_wavefunctions"),
                    ("valence", "valence_energies_ev", "hole_wavefunctions"),
                ):
                    np.testing.assert_array_equal(restored.load_states()[f"{kind}_ev"], calculation[energy_key])
                    np.testing.assert_array_equal(restored.load_wavefunctions()[kind]["values"], calculation[wave_key])
                for current in (result, restored):
                    spectrum = current.load_spectrum()
                    self.assertEqual(spectrum["signal_status"], status)
                    self.assertEqual(spectrum["signal_message"], calculation["metadata"]["pl_message"])
                    self.assertEqual(spectrum["intensity_unit"], "a.u.")
                    np.testing.assert_array_equal(spectrum["intensity"], np.zeros_like(spectrum["wavelength_nm"]))
                    self.assertNotIn("normalization_reference_intensity", spectrum)
                    self.assertNotIn("intensity_normalization", spectrum)
                with np.load(archive, allow_pickle=False) as raw:
                    np.testing.assert_array_equal(raw["energy_intensity"], calculation["energy_intensity"])
                    for wave_key in ("electron_wavefunctions", "hole_wavefunctions"):
                        np.testing.assert_array_equal(raw[wave_key], calculation[wave_key])

    def test_band_plot_overlays_only_available_states_and_wavefunctions(self):
        from matplotlib.figure import Figure

        structure = SimpleNamespace(get_well_regions=lambda: [{"start_nm": 1, "end_nm": 2}])
        for empty_kinds in (("electron", "valence"), ("electron",), ("valence",)):
            for include_wavefunctions in (False, True):
                with self.subTest(empty_kinds=empty_kinds, include_wavefunctions=include_wavefunctions):
                    calculation = no_signal_fixture(empty_kinds=empty_kinds)
                    result = QWResults.from_calculation(calculation)
                    with patch.object(Figure, "savefig", autospec=True) as save:
                        result.plot_band_diagram(self.directory / "bands.png", structure=structure,
                                                 include_wavefunctions=include_wavefunctions)
                    ax = save.call_args.args[0].axes[0]
                    count = len(calculation["electron_energies_ev"]) + len(calculation["valence_energies_ev"])
                    self.assertEqual(len(ax.collections), count)
                    self.assertEqual(len(ax.lines), 2 + (count if include_wavefunctions else 0))
                    self.assertFalse(any("No bound" in text.get_text() for text in ax.texts))
                    if count == 0:
                        self.assertEqual(len(ax.texts), 0)

    def test_empty_state_groups_still_require_matching_wavefunction_shapes(self):
        for replacement in (np.empty((0, 60)), np.empty(0), np.ones((1, 61))):
            with self.subTest(shape=replacement.shape):
                calculation = no_signal_fixture()
                calculation["electron_wavefunctions"] = replacement
                with self.assertRaisesRegex(ValueError, "shape"):
                    QWResults.from_calculation(calculation)
        calculation = no_signal_fixture(empty_kinds=("electron",))
        calculation["hole_wavefunctions"][:] = 0
        with self.assertRaisesRegex(ValueError, "zero envelope"):
            QWResults.from_calculation(calculation)

    def test_no_signal_metadata_cannot_hide_positive_emission(self):
        for status in ("no_bound_transitions", "no_signal_in_window"):
            with self.subTest(status=status):
                calculation = calculation_fixture()
                calculation["metadata"]["pl_status"] = status
                with self.assertRaisesRegex(ValueError, "all-zero"):
                    QWResults.from_calculation(calculation)
        calculation = no_signal_fixture()
        del calculation["metadata"]["pl_status"]
        calculation["energy_intensity"][:] = 1
        with self.assertRaisesRegex(ValueError, "without electron or valence states"):
            QWResults.from_calculation(calculation)

    def test_factory_validates_native_pl_metadata(self):
        for status in (None, False, "unknown", [], {}):
            with self.subTest(status=status):
                calculation = calculation_fixture()
                calculation["metadata"]["pl_status"] = status
                with self.assertRaisesRegex(ValueError, "pl_status"):
                    QWResults.from_calculation(calculation)
        calculation = no_signal_fixture()
        calculation["metadata"]["pl_message"] = None
        with self.assertRaisesRegex(ValueError, "pl_message"):
            QWResults.from_calculation(calculation)

    def test_unmarked_zero_spectrum_remains_rejected_by_loader(self):
        calculation = calculation_fixture()
        calculation["energy_intensity"][:] = 0
        result = QWResults.from_calculation(calculation)
        with self.assertRaisesRegex(ValueError, "signal|positive"):
            result.load_spectrum()

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
