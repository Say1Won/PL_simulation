"""CLI integration with clearly synthetic temporary nextnano-format fixtures."""

import configparser
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CommandLineIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.output_root = self.directory / "outputs"
        self.source = self.directory / "synthetic_solver_format"
        self.source.mkdir()
        self.environment = dict(os.environ)
        self.environment.update({
            "MPLCONFIGDIR": str(self.directory / "matplotlib"),
            "XDG_CACHE_HOME": str(self.directory / "cache"),
        })
        self.configuration = json.loads((ROOT / "configs/simulation.json").read_text(encoding="utf-8"))

    def invoke(self, *arguments):
        return subprocess.run(
            [sys.executable, str(ROOT / "main.py"), "--output-root", str(self.output_root), *arguments],
            cwd=self.directory, env=self.environment, capture_output=True, text=True, timeout=30,
        )

    def write_synthetic_outputs(self, intensities=(0, 1, 5, 1, 0)):
        # These are arithmetic pipeline fixtures, not predictions for GaN/InGaN.
        bands = self.source / self.configuration["output_files"]["band_edges"]["path"]
        bands.parent.mkdir(parents=True, exist_ok=True)
        bands.write_text(
            "x[nm] Gamma[eV] HH[eV] LH[eV] SO[eV]\n"
            "0 3.5 0 0 -0.2\n20 3.1 0.1 0.05 -0.2\n"
            "23 3.1 0.1 0.05 -0.2\n43 3.5 0 0 -0.2\n", encoding="utf-8")
        optical = self.source / self.configuration["output_files"]["spectrum"]["path"]
        optical.parent.mkdir(parents=True, exist_ok=True)
        optical.write_text("Energy[eV] PhotonDensity[photons/cm^2/s/eV]\n" + "\n".join(
            f"{energy} {intensity}" for energy, intensity in zip((2, 2.5, 3, 3.5, 4), intensities)
        ) + "\n", encoding="utf-8")

    def test_prepare_only_creates_inputs_without_a_solver_or_existing_outputs(self):
        config = configparser.ConfigParser()
        config["nextnano++"] = {"exe": "", "database": "", "license": "",
                                "outputdirectory": "", "threads": "0"}
        configpath = self.directory / "prepare.ini"
        with configpath.open("w", encoding="utf-8") as stream:
            config.write(stream)
        self.assertFalse(self.output_root.exists())
        completed = self.invoke("--prepare-only", "--nextnano-config", str(configpath))
        self.assertEqual(completed.returncode, 0, completed.stderr)
        run = self.output_root / "run_001"
        manifest = json.loads((run / "inputs/run.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "prepared")
        self.assertTrue((run / "inputs/single_qw_pl.nnp").is_file())
        self.assertFalse((run / "figures").exists())
        self.assertFalse((run / "nextnano").exists())

    def test_existing_results_produce_pl_and_band_outputs_and_preserve_provenance(self):
        self.write_synthetic_outputs()
        self.assertFalse(self.output_root.exists())
        completed = self.invoke("--results-dir", str(self.source))
        self.assertEqual(completed.returncode, 0, completed.stderr)
        run = self.output_root / "run_001"
        for filename in ("quantum_well.png", "pl_spectrum.png"):
            image = run / "figures" / filename
            self.assertEqual(image.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        with (run / "analysis/pl_spectrum.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
        self.assertEqual(len(rows), 6)
        wavelengths = [float(row[0]) for row in rows[1:]]
        self.assertEqual(wavelengths, sorted(wavelengths))
        summary = json.loads((run / "analysis/pl_summary.json").read_text(encoding="utf-8"))
        self.assertAlmostEqual(summary["energy_ev"], 3)
        self.assertEqual(summary["intensity_unit"], "photons/cm^2/s/nm")
        self.assertEqual(summary["excitation_model"], "external_results_conditions_unverified")
        self.assertIs(summary["settings_verified_against_generated_input"], False)
        self.assertEqual(summary["configured_electron_fermi_ev"],
                         self.configuration["settings"]["electron_fermi_ev"])
        manifest = json.loads((run / "inputs/run.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["source_directory"], str(self.source.resolve()))

    def test_repeated_runs_allocate_a_new_directory_without_overwriting_first(self):
        self.write_synthetic_outputs()
        first = self.invoke("--results-dir", str(self.source))
        self.assertEqual(first.returncode, 0, first.stderr)
        summary = self.output_root / "run_001/analysis/pl_summary.json"
        original = summary.read_bytes()
        second = self.invoke("--results-dir", str(self.source))
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(summary.read_bytes(), original)
        self.assertTrue((self.output_root / "run_002/analysis/pl_summary.json").is_file())

    def test_no_emission_signal_fails_without_completed_analysis(self):
        self.write_synthetic_outputs(intensities=(0, 0, 0, 0, 0))
        completed = self.invoke("--results-dir", str(self.source))
        self.assertEqual(completed.returncode, 1)
        run = self.output_root / "run_001"
        manifest = json.loads((run / "inputs/run.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "failed")
        self.assertIn("at least one positive value", manifest["error"])
        self.assertFalse((run / "analysis/pl_summary.json").exists())
        self.assertFalse((run / "figures").exists())


if __name__ == "__main__":
    unittest.main()
