"""Standalone CLI integration with computed states and native result archives."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class CommandLineIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.output_root = self.directory / "outputs"
        self.environment = dict(os.environ)
        self.environment.update({
            "MPLCONFIGDIR": str(self.directory / "matplotlib"),
            "XDG_CACHE_HOME": str(self.directory / "cache"),
        })

    def invoke(self, *arguments, forbid_nextnanopy=False):
        arguments = [str(ROOT / "main.py"), "--output-root", str(self.output_root), *arguments]
        if forbid_nextnanopy:
            # Make even an installed nextnanopy unusable. Native execution must
            # neither import it nor need an external solver/configuration.
            code = (
                "import builtins, runpy, sys\n"
                "original_import = builtins.__import__\n"
                "def checked_import(name, *args, **kwargs):\n"
                "    if name.split('.')[0] == 'nextnanopy':\n"
                "        raise ImportError('nextnanopy intentionally unavailable in native test')\n"
                "    return original_import(name, *args, **kwargs)\n"
                "builtins.__import__ = checked_import\n"
                "sys.argv = sys.argv[1:]\n"
                "sys.path.insert(0, str(__import__('pathlib').Path(sys.argv[0]).parent))\n"
                "runpy.run_path(sys.argv[0], run_name='__main__')\n"
            )
            command = [sys.executable, "-c", code, *arguments]
        else:
            command = [sys.executable, *arguments]
        return subprocess.run(command, cwd=self.directory, env=self.environment,
                              capture_output=True, text=True, timeout=45)

    def load_summary(self, run_number=1):
        return json.loads((self.output_root / f"run_{run_number:03d}" /
                           "analysis/pl_summary.json").read_text(encoding="utf-8"))

    def test_prepare_only_saves_conditions_without_calculating(self):
        self.assertFalse(self.output_root.exists())
        completed = self.invoke("--prepare-only", forbid_nextnanopy=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        run = self.output_root / "run_001"
        manifest = json.loads((run / "inputs/run.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "prepared")
        for name in ("single_qw.json", "simulation.json", "materials.json"):
            self.assertIsInstance(json.loads((run / "inputs" / name).read_text(encoding="utf-8")), dict)
        self.assertFalse(list((run / "inputs").glob("*.nnp")))
        self.assertFalse((run / "figures").exists())
        self.assertFalse((run / "calculation").exists())

    def test_default_calculation_runs_without_nextnanopy_and_produces_outputs(self):
        completed = self.invoke(forbid_nextnanopy=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        run = self.output_root / "run_001"
        for filename in ("quantum_well.png", "pl_spectrum.png"):
            self.assertEqual((run / "figures" / filename).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        with (run / "analysis/pl_spectrum.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
        self.assertEqual(rows[0], ["wavelength_nm", "intensity [a.u.]"])
        values = np.array([[float(row[0]), float(row[1])] for row in rows[1:]])
        self.assertGreater(len(values), 100)
        self.assertTrue(np.all(np.diff(values[:, 0]) > 0))
        self.assertTrue(np.all(values[:, 1] >= 0))
        self.assertEqual(max(values[:, 1]), 1)
        summary = self.load_summary()
        self.assertEqual(summary["intensity_unit"], "a.u.")
        self.assertEqual(summary["intensity"], 1)
        self.assertEqual(summary["intensity_normalization"], "divide_by_wavelength_maximum")
        self.assertGreater(summary["normalization_reference_intensity"], 0)
        self.assertEqual(summary["calculation_metadata"]["energy_intensity_unit"], "a.u.")
        self.assertGreater(summary["energy_ev"], 2)
        self.assertLess(summary["energy_ev"], 4)
        self.assertGreater(summary["fwhm_nm"], 0)
        manifest = json.loads((run / "inputs/run.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "completed")
        self.assertFalse((run / "nextnano").exists())
        with np.load(run / "calculation/result.npz", allow_pickle=False) as archive:
            self.assertGreater(len(archive["electron_energies_ev"]), 0)
            self.assertGreater(len(archive["valence_energies_ev"]), 0)
            self.assertTrue(np.max(archive["energy_intensity"]) > 0)
            metadata = json.loads(str(archive["metadata_json"]))
            self.assertEqual(metadata["energy_intensity_unit"], "a.u.")
            self.assertEqual(metadata["energy_density_axis_unit"], "eV")

    def test_native_archive_analysis_preserves_spectrum_without_overwriting_first_run(self):
        first = self.invoke()
        self.assertEqual(first.returncode, 0, first.stderr)
        original_summary = self.load_summary()
        first_csv = self.output_root / "run_001/analysis/pl_spectrum.csv"
        original_csv = first_csv.read_bytes()
        archive_dir = self.output_root / "run_001/calculation"
        second = self.invoke("--results-dir", str(archive_dir), forbid_nextnanopy=True)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(first_csv.read_bytes(), original_csv)
        self.assertEqual((self.output_root / "run_002/analysis/pl_spectrum.csv").read_bytes(), original_csv)
        reloaded_summary = self.load_summary(2)
        for quantity in ("wavelength_nm", "energy_ev", "fwhm_nm", "intensity",
                         "intensity_normalization", "normalization_reference_intensity"):
            self.assertEqual(reloaded_summary[quantity], original_summary[quantity])

    def test_indium_fraction_override_changes_pl_and_records_the_applied_structure(self):
        source_path = ROOT / "configs/single_qw.json"
        original_bytes = source_path.read_bytes()
        original = json.loads(original_bytes)
        summaries = []
        for run_number, fraction in enumerate((0.20, 0.25), start=1):
            with self.subTest(fraction=fraction):
                completed = self.invoke("--indium-fraction", str(fraction))
                self.assertEqual(completed.returncode, 0, completed.stderr)
                run = self.output_root / f"run_{run_number:03d}"
                expected = json.loads(original_bytes)
                expected["layers"][1]["indium_fraction"] = fraction
                snapshot = json.loads((run / "inputs/single_qw.json").read_text())
                self.assertEqual(snapshot, expected)
                manifest = json.loads((run / "inputs/run.json").read_text())
                self.assertEqual(manifest["structure"], expected)
                summary = self.load_summary(run_number)
                self.assertEqual(summary["calculation_metadata"]["structure"], expected)
                self.assertEqual(summary["intensity"], 1)
                self.assertEqual(summary["intensity_unit"], "a.u.")
                self.assertFalse(summary["peak_at_boundary"])
                with np.load(run / "calculation/result.npz", allow_pickle=False) as archive:
                    metadata = json.loads(str(archive["metadata_json"]))
                self.assertEqual(metadata["structure"], expected)
                self.assertEqual(metadata["material_profiles"]["layers"][1]["indium_fraction"], fraction)
                self.assertEqual(source_path.read_bytes(), original_bytes)
                self.assertEqual(snapshot["layers"][0], original["layers"][0])
                self.assertEqual(snapshot["layers"][2], original["layers"][2])
                summaries.append(summary)
        self.assertGreater(summaries[1]["wavelength_nm"], summaries[0]["wavelength_nm"])
        self.assertLess(summaries[1]["energy_ev"], summaries[0]["energy_ev"])

    def test_indium_fraction_prepare_only_and_omitted_option_use_the_custom_structure(self):
        structure = json.loads((ROOT / "configs/single_qw.json").read_text())
        structure["layers"][1]["indium_fraction"] = 0.31
        structure_path = self.directory / "custom_structure.json"
        original_bytes = json.dumps(structure).encode()
        structure_path.write_bytes(original_bytes)
        for run_number, arguments, expected_fraction in (
            (1, (), 0.31), (2, ("--indium-fraction", "0.25"), 0.25),
        ):
            completed = self.invoke("--structure", str(structure_path), "--prepare-only", *arguments)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            run = self.output_root / f"run_{run_number:03d}"
            saved = json.loads((run / "inputs/single_qw.json").read_text())
            self.assertEqual(saved["layers"][1]["indium_fraction"], expected_fraction)
            self.assertFalse((run / "calculation").exists())
            self.assertEqual(structure_path.read_bytes(), original_bytes)

    def test_invalid_indium_fractions_fail_before_allocating_outputs(self):
        for fraction in ("-0.1", "1.1", "25", "nan", "inf", "-inf"):
            with self.subTest(fraction=fraction):
                completed = self.invoke(f"--indium-fraction={fraction}", "--prepare-only")
                self.assertEqual(completed.returncode, 1, completed.stderr)
                self.assertIn("indium_fraction", completed.stderr)
                self.assertFalse(self.output_root.exists())

    def test_indium_fraction_override_accepts_nonconfining_compositions(self):
        structure = json.loads((ROOT / "configs/single_qw.json").read_text())
        structure["layers"][0].update(material="InGaN", indium_fraction=0.10)
        structure_path = self.directory / "alloy_barrier.json"
        structure_path.write_text(json.dumps(structure))
        for fraction in (0, 0.10, 0.05):
            with self.subTest(fraction=fraction):
                completed = self.invoke("--structure", str(structure_path),
                                        "--indium-fraction", str(fraction), "--prepare-only")
                self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_zero_indium_runs_and_reloads_with_zero_pl_and_no_peak(self):
        completed = self.invoke("--indium-fraction", "0", "--include-wavefunctions")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("PL intensity: 0 a.u.; peak: none", completed.stdout)
        summary = self.load_summary()
        self.assertFalse(summary["has_signal"])
        self.assertEqual(summary["signal_status"], "no_bound_transitions")
        self.assertEqual(summary["intensity"], 0)
        self.assertEqual(summary["intensity_unit"], "a.u.")
        for name in ("wavelength_nm", "energy_ev", "peak_index", "fwhm_nm"):
            self.assertIsNone(summary[name])
        self.assertNotIn("intensity_normalization", summary)
        run = self.output_root / "run_001"
        manifest = json.loads((run / "inputs/run.json").read_text())
        self.assertEqual(manifest["status"], "completed")
        with np.load(run / "calculation/result.npz", allow_pickle=False) as archive:
            self.assertEqual(archive["electron_energies_ev"].size, 0)
            self.assertEqual(archive["valence_energies_ev"].size, 0)
            self.assertFalse(np.any(archive["energy_intensity"]))
        with (run / "analysis/pl_spectrum.csv").open(newline="") as stream:
            rows = list(csv.reader(stream))
        self.assertEqual(rows[0][1], "intensity [a.u.]")
        self.assertTrue(all(float(row[1]) == 0 for row in rows[1:]))
        for name in ("quantum_well.png", "pl_spectrum.png"):
            self.assertTrue((run / "figures" / name).is_file())
        reload = self.invoke("--results-dir", str(run / "calculation"))
        self.assertEqual(reload.returncode, 0, reload.stderr)
        self.assertEqual(self.load_summary(2)["intensity"], 0)
        self.assertIsNone(self.load_summary(2)["wavelength_nm"])
        self.assertEqual((run / "analysis/pl_spectrum.csv").read_bytes(),
                         (self.output_root / "run_002/analysis/pl_spectrum.csv").read_bytes())

    def test_user_thin_dilute_case_keeps_hole_states_and_returns_zero_pl(self):
        completed = self.invoke("--indium-fraction", "0.005", "--well-thickness-nm", "2")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        summary = self.load_summary()
        metadata = summary["calculation_metadata"]
        self.assertEqual(metadata["bound_electron_states"], 0)
        self.assertEqual(metadata["bound_hole_states"], 1)
        self.assertEqual(summary["intensity"], 0)
        self.assertIsNone(summary["wavelength_nm"])

    def test_pure_inn_endpoint_runs_and_saves_fraction_one(self):
        completed = self.invoke("--indium-fraction", "1")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        summary = self.load_summary()
        self.assertEqual(summary["calculation_metadata"]["structure"]["layers"][1]["indium_fraction"], 1)
        self.assertEqual(summary["intensity_unit"], "a.u.")

    def test_indium_fraction_cannot_override_archived_results(self):
        completed = self.invoke("--indium-fraction", "0.25", "--results-dir", str(self.directory))
        self.assertEqual(completed.returncode, 1, completed.stderr)
        self.assertIn("cannot be used with --results-dir", completed.stderr)
        self.assertFalse(self.output_root.exists())

    def test_well_thickness_changes_geometry_eigenvalues_and_pl(self):
        source_path = ROOT / "configs/single_qw.json"
        original_bytes = source_path.read_bytes()
        summaries, electron_ground_states = [], []
        for run_number, thickness in enumerate((3.0, 4.0), start=1):
            with self.subTest(thickness=thickness):
                completed = self.invoke("--indium-fraction", "0.20",
                                        "--well-thickness-nm", str(thickness))
                self.assertEqual(completed.returncode, 0, completed.stderr)
                run = self.output_root / f"run_{run_number:03d}"
                expected = json.loads(original_bytes)
                expected["layers"][1].update(indium_fraction=0.20, thickness_nm=thickness)
                self.assertEqual(json.loads((run / "inputs/single_qw.json").read_text()), expected)
                self.assertEqual(json.loads((run / "inputs/run.json").read_text())["structure"], expected)
                summary = self.load_summary(run_number)
                self.assertEqual(summary["calculation_metadata"]["structure"], expected)
                self.assertEqual(summary["intensity"], 1)
                self.assertEqual(summary["intensity_unit"], "a.u.")
                self.assertFalse(summary["peak_at_boundary"])
                self.assertTrue((run / "figures/quantum_well.png").is_file())
                with np.load(run / "calculation/result.npz", allow_pickle=False) as archive:
                    metadata = json.loads(str(archive["metadata_json"]))
                    self.assertEqual(metadata["structure"], expected)
                    layer = metadata["material_profiles"]["layers"][1]
                    self.assertEqual(layer["thickness_nm"], thickness)
                    self.assertAlmostEqual(layer["end_nm"] - layer["start_nm"], thickness)
                    self.assertAlmostEqual(float(archive["position_nm"][-1]),
                                           sum(item["thickness_nm"] for item in expected["layers"]))
                    electron_ground_states.append(float(archive["electron_energies_ev"][0]))
                self.assertEqual(source_path.read_bytes(), original_bytes)
                summaries.append(summary)
        self.assertLess(electron_ground_states[1], electron_ground_states[0])
        self.assertGreater(summaries[1]["wavelength_nm"], summaries[0]["wavelength_nm"])

    def test_well_overrides_can_be_independent_and_prepare_only_preserves_custom_defaults(self):
        structure = json.loads((ROOT / "configs/single_qw.json").read_text())
        structure["layers"][1].update(indium_fraction=0.31, thickness_nm=5.5)
        structure_path = self.directory / "custom_geometry.json"
        original_bytes = json.dumps(structure).encode()
        structure_path.write_bytes(original_bytes)
        cases = (
            ((), 0.31, 5.5),
            (("--well-thickness-nm", "4.2"), 0.31, 4.2),
            (("--indium-fraction", "0.25"), 0.25, 5.5),
            (("--indium-fraction", "0.25", "--well-thickness-nm", "4.2"), 0.25, 4.2),
        )
        for run_number, (arguments, fraction, thickness) in enumerate(cases, start=1):
            with self.subTest(arguments=arguments):
                completed = self.invoke("--structure", str(structure_path), "--prepare-only", *arguments)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                run = self.output_root / f"run_{run_number:03d}"
                expected = json.loads(original_bytes)
                expected["layers"][1].update(indium_fraction=fraction, thickness_nm=thickness)
                self.assertEqual(json.loads((run / "inputs/single_qw.json").read_text()), expected)
                self.assertFalse((run / "calculation").exists())
                self.assertEqual(structure_path.read_bytes(), original_bytes)

    def test_invalid_well_thickness_fails_before_allocating_outputs(self):
        for thickness in ("0", "-1", "nan", "inf", "-inf"):
            with self.subTest(thickness=thickness):
                completed = self.invoke(f"--well-thickness-nm={thickness}", "--prepare-only")
                self.assertEqual(completed.returncode, 1, completed.stderr)
                self.assertIn("thickness_nm", completed.stderr)
                self.assertFalse(self.output_root.exists())

    def test_well_thickness_cannot_override_archived_results(self):
        for arguments in (("--well-thickness-nm", "4"),
                          ("--well-thickness-nm", "4", "--indium-fraction", "0.25")):
            with self.subTest(arguments=arguments):
                completed = self.invoke(*arguments, "--results-dir", str(self.directory))
                self.assertEqual(completed.returncode, 1, completed.stderr)
                self.assertIn("cannot be used with --results-dir", completed.stderr)
                self.assertFalse(self.output_root.exists())

    def test_native_archive_without_signal_cannot_report_a_peak(self):
        original = self.invoke()
        self.assertEqual(original.returncode, 0, original.stderr)
        source = self.output_root / "run_001/calculation/result.npz"
        with np.load(source, allow_pickle=False) as archive:
            data = {name: archive[name] for name in archive.files}
        data["energy_intensity"] = np.zeros_like(data["energy_intensity"])
        invalid_archive = self.directory / "zero_signal.npz"
        np.savez_compressed(invalid_archive, **data)
        failed = self.invoke("--results-dir", str(invalid_archive))
        self.assertEqual(failed.returncode, 1)
        run = self.output_root / "run_002"
        manifest = json.loads((run / "inputs/run.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "failed")
        self.assertFalse((run / "analysis/pl_summary.json").exists())
        self.assertFalse((run / "figures").exists())

    def test_unsupported_physics_fails_before_allocating_output(self):
        configuration = json.loads((ROOT / "configs/simulation.json").read_text(encoding="utf-8"))
        configuration["settings"]["include_polarization"] = True
        settings_path = self.directory / "unsupported.json"
        settings_path.write_text(json.dumps(configuration), encoding="utf-8")
        failed = self.invoke("--settings", str(settings_path))
        self.assertEqual(failed.returncode, 1)
        self.assertIn("unsupported", failed.stderr.lower())
        self.assertFalse(self.output_root.exists())


if __name__ == "__main__":
    unittest.main()
