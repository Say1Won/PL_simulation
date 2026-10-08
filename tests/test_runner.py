"""Real nextnanopy input preparation with mocked external solver processes."""

import configparser
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pl_simulation.nextnano_simulation import NextnanoSimulation
from pl_simulation.simulation_settings import SimulationSettings
from pl_simulation.single_qw_structure import SingleQWStructure


ROOT = Path(__file__).resolve().parents[1]


class RunnerLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        config = configparser.ConfigParser()
        paths = {}
        for key in ("exe", "database", "license"):
            source = self.directory / key
            source.write_text("test placeholder; external execution is mocked\n", encoding="utf-8")
            paths[key] = str(source)
        config["nextnano++"] = {**paths, "outputdirectory": str(self.directory), "threads": "0"}
        self.configpath = self.directory / "nextnanopy.ini"
        with self.configpath.open("w", encoding="utf-8") as stream:
            config.write(stream)
        self.structure = SingleQWStructure.from_dict(json.loads(
            (ROOT / "configs/single_qw.json").read_text(encoding="utf-8")))
        self.settings = SimulationSettings.from_dict(json.loads(
            (ROOT / "configs/simulation.json").read_text(encoding="utf-8"))["settings"])
        self.runner = NextnanoSimulation(ROOT / "templates/single_qw_pl.nnp", self.configpath)

    def prepare(self):
        self.runner.load_template()
        self.runner.apply_parameters(self.structure, self.settings)
        return self.runner.save_input(self.directory / "inputs" / "prepared.nnp")

    def test_real_input_preparation_preserves_template_and_applies_parameters(self):
        template = self.runner.template_path.read_bytes()
        snapshot = self.prepare()
        self.assertEqual(self.runner.template_path.read_bytes(), template)
        self.assertTrue(snapshot.is_file())
        import nextnanopy as nn
        loaded = nn.InputFile(snapshot, configpath=self.configpath)
        expected = self.structure.to_input_variables() | self.settings.to_input_variables()
        for name, value in expected.items():
            with self.subTest(variable=name):
                self.assertEqual(float(loaded.variables[name].value), float(value))
        with self.assertRaises(FileExistsError):
            self.runner.save_input(snapshot)

    def test_template_missing_a_physical_variable_is_rejected(self):
        incomplete = self.directory / "incomplete.nnp"
        incomplete.write_text(self.runner.template_path.read_text(encoding="utf-8").replace(
            "$WELL_THICKNESS", "$UNCONFIGURED_THICKNESS"), encoding="utf-8")
        runner = NextnanoSimulation(incomplete, self.configpath)
        with self.assertRaisesRegex(ValueError, "WELL_THICKNESS"):
            runner.apply_parameters(self.structure, self.settings)

    def test_unprepared_and_unsaved_inputs_cannot_run(self):
        with self.assertRaises(RuntimeError):
            self.runner.run(self.directory / "results")
        self.runner.load_template()
        with self.assertRaises(RuntimeError):
            self.runner.run(self.directory / "results")

    def test_existing_outputs_cannot_be_reused_as_a_successful_run(self):
        self.prepare()
        destination = self.directory / "results"
        destination.mkdir()
        (destination / "old.dat").write_text("old fixture\n", encoding="utf-8")
        with patch.object(self.runner.input_file, "execute") as execute:
            with self.assertRaises(FileExistsError):
                self.runner.run(destination)
            execute.assert_not_called()

    def test_failure_exit_is_rejected_even_when_a_dat_file_exists(self):
        self.prepare()
        destination = self.directory / "results"

        def failed_execution(**kwargs):
            output = Path(kwargs["outputdirectory"])
            (output / "partial.dat").write_text("partial test fixture\n", encoding="utf-8")
            return {"process": SimpleNamespace(returncode=7), "outputdirectory": str(output)}

        with patch.object(self.runner.input_file, "execute", side_effect=failed_execution):
            with self.assertRaisesRegex(RuntimeError, "code 7"):
                self.runner.run(destination, show_log=False)

    def test_zero_exit_without_data_is_not_a_success(self):
        self.prepare()
        destination = self.directory / "results"
        info = {"process": SimpleNamespace(returncode=0), "outputdirectory": str(destination)}
        with patch.object(self.runner.input_file, "execute", return_value=info):
            with self.assertRaisesRegex(RuntimeError, r"no .*\.dat"):
                self.runner.run(destination, show_log=False)

    def test_zero_exit_with_only_empty_data_is_not_a_success(self):
        self.prepare()
        destination = self.directory / "results"

        def empty_execution(**kwargs):
            output = Path(kwargs["outputdirectory"])
            (output / "empty.dat").touch()
            return {"process": SimpleNamespace(returncode=0), "outputdirectory": str(output)}

        with patch.object(self.runner.input_file, "execute", side_effect=empty_execution):
            with self.assertRaises(RuntimeError):
                self.runner.run(destination, show_log=False)

    def test_success_requests_convergence_check_and_returns_actual_output_directory(self):
        self.prepare()
        destination = self.directory / "results"

        def completed_execution(**kwargs):
            output = Path(kwargs["outputdirectory"])
            (output / "output.dat").write_text("successful test fixture\n", encoding="utf-8")
            return {"process": SimpleNamespace(returncode=0), "outputdirectory": str(output)}

        with patch.object(self.runner.input_file, "execute", side_effect=completed_execution) as execute:
            self.assertEqual(self.runner.run(destination, show_log=False), destination.resolve())
            self.assertIs(execute.call_args.kwargs["convergenceCheck"], True)
            self.assertEqual(execute.call_args.kwargs["convergence_check_mode"], "terminate")
            self.assertIs(execute.call_args.kwargs["create_subdirectory"], False)


if __name__ == "__main__":
    unittest.main()
