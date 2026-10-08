"""Numerical spectrum tests use temporary arithmetic fixtures, not PL predictions."""

from pathlib import Path
import os
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from pl_simulation.peak_analyzer import PeakAnalyzer
from pl_simulation.qw_results import QWResults


def spectrum(wavelength, intensity):
    return {"wavelength_nm": wavelength, "intensity": intensity,
            "intensity_unit": "test_rate/nm", "axis": "wavelength", "axis_unit": "nm"}


class SpectrumFileTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        for name in ("matplotlib", "cache"):
            (self.directory / name).mkdir()
        environment = patch.dict(os.environ, {
            "MPLCONFIGDIR": str(self.directory / "matplotlib"),
            "XDG_CACHE_HOME": str(self.directory / "cache"),
        })
        environment.start()
        self.addCleanup(environment.stop)

    def write_data(self, name, header, rows):
        path = self.directory / name
        path.write_text(header + "\n" + "\n".join(
            " ".join(str(value) for value in row) for row in rows) + "\n", encoding="utf-8")
        return path

    def results(self, **overrides):
        spec = {"path": "spectrum.dat", "coordinate": "energy", "intensity": "intensity",
                "axis_kind": "energy", "spectral_density": True,
                "density_axis_unit": "eV", "wavelength_intensity_unit": "test_rate/nm"}
        spec.update(overrides)
        return QWResults(self.directory, {"spectrum": spec})

    def test_energy_to_wavelength_density_conserves_integrated_signal(self):
        energy = np.linspace(2, 4, 1001)
        self.write_data("spectrum.dat", "energy[eV] intensity[test_rate/eV]",
                        zip(energy, np.ones_like(energy)))
        loaded = self.results().load_spectrum()
        wavelength, intensity = loaded["wavelength_nm"], loaded["intensity"]
        self.assertTrue(np.all(np.diff(wavelength) > 0))
        integral = float(np.sum((intensity[:-1] + intensity[1:]) * np.diff(wavelength) / 2))
        self.assertAlmostEqual(integral, 2.0, places=5)
        np.testing.assert_allclose(intensity, QWResults.HC_EV_NM / wavelength**2)
        self.assertEqual(loaded["intensity_unit"], "test_rate/nm")
        # A constant energy density is not a constant wavelength density.
        self.assertGreater(intensity[0], intensity[-1])

    def test_density_per_mev_is_scaled_before_jacobian(self):
        self.write_data("spectrum.dat", "energy[meV] intensity[test_rate/meV]",
                        [(4000, 0.001), (2000, 0.001), (3000, 0.001)])
        loaded = self.results(density_axis_unit="meV").load_spectrum()
        wavelength = loaded["wavelength_nm"]
        np.testing.assert_allclose(wavelength, QWResults.HC_EV_NM / np.array([4, 3, 2]))
        np.testing.assert_allclose(loaded["intensity"], np.array([16, 9, 4]) / QWResults.HC_EV_NM)

    def test_wavelength_density_per_micrometer_becomes_per_nm(self):
        self.write_data("spectrum.dat", "wavelength[um] intensity[test_rate/um]",
                        [(0.5, 2000), (0.4, 1000)])
        loaded = self.results(coordinate="wavelength", axis_kind="wavelength",
                              density_axis_unit="um").load_spectrum()
        np.testing.assert_allclose(loaded["wavelength_nm"], [400, 500])
        np.testing.assert_allclose(loaded["intensity"], [1, 2])

    def test_energy_density_metadata_is_required_to_avoid_axis_only_conversion(self):
        self.write_data("spectrum.dat", "energy[eV] intensity[test_rate/eV]", [(2, 1), (3, 1)])
        for override in ({"spectral_density": False}, {"density_axis_unit": None},
                         {"wavelength_intensity_unit": None}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                self.results(**override).load_spectrum()

    def test_negative_signal_and_duplicate_axis_are_rejected(self):
        for rows in ([(2, 1), (3, -1)], [(2, 1), (2, 2)]):
            self.write_data("spectrum.dat", "energy[eV] intensity[test_rate/eV]", rows)
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.results().load_spectrum()

    def test_ambiguous_output_mapping_and_path_escape_are_rejected(self):
        self.write_data("a.dat", "energy[eV] intensity[test_rate/eV]", [(2, 1), (3, 1)])
        self.write_data("b.dat", "energy[eV] intensity[test_rate/eV]", [(2, 1), (3, 1)])
        with self.assertRaises(FileNotFoundError):
            self.results(path=None, glob="*.dat").load_spectrum()
        with self.assertRaises(ValueError):
            self.results(path="../external.dat").load_spectrum()

    def test_band_length_and_energy_units_are_converted_without_guessed_columns(self):
        self.write_data("bands.dat", "x[um] conduction[meV] valence[meV]",
                        [(0.02, 3200, 100), (0.00, 3500, 0), (0.01, 3300, 50)])
        results = QWResults(self.directory, {"band_edges": {
            "path": "bands.dat", "coordinate": "x", "conduction": "conduction",
            "valence": "valence", "energy_reference": "test_reference",
        }})
        bands = results.load_band_edges()
        np.testing.assert_allclose(bands["position_nm"], [0, 10, 20])
        np.testing.assert_allclose(bands["conduction_ev"], [3.5, 3.3, 3.2])
        np.testing.assert_allclose(bands["valence_ev"], [0, 0.05, 0.1])
        with self.assertRaises(ValueError):
            QWResults(self.directory).load_band_edges()

    def test_valence_edge_tracks_uppermost_configured_branch(self):
        self.write_data("bands.dat", "x[nm] conduction[eV] HH[eV] LH[eV] SO[eV]",
                        [(0, 3, 0.05, 0.10, -0.2), (10, 3, 0.20, 0.10, -0.2)])
        results = QWResults(self.directory, {"band_edges": {
            "path": "bands.dat", "coordinate": "x", "conduction": "conduction",
            "valence": ["HH", "LH", "SO"],
        }})
        bands = results.load_band_edges()
        np.testing.assert_allclose(bands["valence_ev"], [0.1, 0.2])
        np.testing.assert_allclose(bands["valence_components_ev"]["HH"], [0.05, 0.2])

    def test_state_overlay_cannot_mix_energy_references(self):
        self.write_data("bands.dat", "x[nm] conduction[eV] valence[eV]",
                        [(0, 3, 0), (10, 3, 0)])
        self.write_data("electron.dat", "index[1] energy[eV]", [(1, 3.2)])
        self.write_data("valence.dat", "index[1] energy[eV]", [(1, 0.2)])
        results = QWResults(self.directory, {
            "band_edges": {"path": "bands.dat", "coordinate": "x",
                           "conduction": "conduction", "valence": "valence",
                           "energy_reference": "band_reference"},
            "states": {"energy_reference": "different_state_reference",
                       "electron": {"path": "electron.dat", "variable": "energy"},
                       "valence": {"path": "valence.dat", "variable": "energy"}},
        })
        structure = SimpleNamespace(get_well_regions=lambda: [{"start_nm": 3, "end_nm": 7}])
        with self.assertRaisesRegex(ValueError, "matching declared energy references"):
            results.plot_band_diagram(self.directory / "invalid.png", structure=structure)
        self.assertFalse((self.directory / "invalid.png").exists())


class PeakNumericalTests(unittest.TestCase):
    def test_fwhm_interpolates_nonuniform_grid_crossings(self):
        analyzer = PeakAnalyzer(spectrum([400, 401, 403, 407, 409], [0, 3, 10, 2, 0]))
        peak = analyzer.find_spectrum_peak()
        self.assertEqual(peak["wavelength_nm"], 403)
        self.assertAlmostEqual(peak["fwhm_nm"], 405.5 - (401 + 4 / 7))
        self.assertAlmostEqual(peak["energy_ev"], PeakAnalyzer.HC_EV_NM / 403)
        self.assertEqual(peak["intensity_unit"], "test_rate/nm")

    def test_fwhm_uses_selected_lobe_in_multiple_peak_spectrum(self):
        analyzer = PeakAnalyzer(spectrum([400, 401, 402, 403, 404, 405, 406],
                                         [0, 8, 0, 0, 10, 8, 0]))
        self.assertEqual(analyzer.find_spectrum_peak()["wavelength_nm"], 404)
        self.assertAlmostEqual(analyzer.calculate_fwhm(), 1.875)
        self.assertAlmostEqual(analyzer.calculate_fwhm(peak_index=1), 1.0)

    def test_truncated_peak_has_no_resolved_fwhm(self):
        for intensity in ([10, 8, 4], [4, 8, 10], [6, 10, 6]):
            with self.subTest(intensity=intensity):
                analyzer = PeakAnalyzer(spectrum([400, 401, 402], intensity))
                self.assertIsNone(analyzer.calculate_fwhm())

    def test_no_signal_or_bad_axis_cannot_report_a_peak(self):
        invalid = (
            spectrum([400, 401], [0, 0]), spectrum([401, 400], [1, 2]),
            spectrum([400, 400], [1, 2]), spectrum([400, 401], [1, -1]),
            spectrum([400, np.nan], [1, 2]), spectrum([400, 401], [1]),
        )
        for data in invalid:
            with self.subTest(data=data), self.assertRaises(ValueError):
                PeakAnalyzer(data).find_spectrum_peak()

    def test_transition_estimate_requires_shared_reference_and_positive_gap(self):
        base = PeakAnalyzer.estimate_transition(3.2, 0.2, energy_reference="common")
        shifted = PeakAnalyzer.estimate_transition(7.2, 4.2, energy_reference="shifted_common")
        self.assertAlmostEqual(base["energy_ev"], shifted["energy_ev"])
        self.assertAlmostEqual(base["wavelength_nm"], shifted["wavelength_nm"])
        self.assertEqual(base["kind"], "level_spacing_estimate")
        with self.assertRaises(ValueError):
            PeakAnalyzer.estimate_transition(3, 0, energy_reference="")
        with self.assertRaises(ValueError):
            PeakAnalyzer.estimate_transition(0, 3, energy_reference="common")

    def test_overlap_is_normalized_and_invariant_under_complex_global_phase(self):
        x = np.linspace(0, 1, 101)
        electron = np.ones_like(x)
        hole = electron * (3 + 4j)
        self.assertAlmostEqual(PeakAnalyzer.calculate_overlap(x, electron, hole), 1)
        self.assertAlmostEqual(PeakAnalyzer.calculate_overlap(x, electron, 2 * x - 1), 0)
        with self.assertRaises(ValueError):
            PeakAnalyzer.calculate_overlap(x, electron, np.zeros_like(x))


if __name__ == "__main__":
    unittest.main()
