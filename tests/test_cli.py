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
        values = np.array([[float(row[0]), float(row[1])] for row in rows[1:]])
        self.assertGreater(len(values), 100)
        self.assertTrue(np.all(np.diff(values[:, 0]) > 0))
        self.assertTrue(np.all(values[:, 1] >= 0))
        self.assertGreater(max(values[:, 1]), 0)
        summary = self.load_summary()
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
        for quantity in ("wavelength_nm", "energy_ev", "fwhm_nm"):
            self.assertEqual(reloaded_summary[quantity], original_summary[quantity])

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
