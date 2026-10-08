"""Prepare and execute a real nextnano++ calculation through nextnanopy."""

from __future__ import annotations

import configparser
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .simulation_settings import SimulationSettings
from .single_qw_structure import SingleQWStructure


class NextnanoSimulation:
    """Own the input/solver lifecycle; constructing this object runs nothing.

    A temporary local configuration permits input preparation without a solver
    installation. Execution requires an existing nextnanopy configuration with
    valid executable, database and license paths.
    """

    def __init__(self, template_path: str | Path, configpath: str | Path | None = None):
        self.template_path = Path(template_path).expanduser().resolve()
        self.configpath = Path(configpath).expanduser().resolve() if configpath else None
        self.input_file: Any = None
        self.execution_info: dict[str, Any] | None = None
        self._temporary_config: TemporaryDirectory | None = None

    def load_template(self) -> Any:
        """Read the input without creating or modifying a home configuration."""
        if not self.template_path.is_file():
            raise FileNotFoundError(f"Input template not found: {self.template_path}")
        import nextnanopy as nn

        configpath = self.configpath
        if configpath is not None and not configpath.is_file():
            raise FileNotFoundError(f"nextnanopy configuration not found: {configpath}")
        if configpath is None:
            default = Path.home() / ".nextnanopy-config"
            if default.is_file():
                configpath = default
            else:
                self._temporary_config = TemporaryDirectory(prefix="pl-nextnanopy-")
                configpath = Path(self._temporary_config.name) / "nextnanopy.ini"
                config = configparser.ConfigParser()
                config["nextnano++"] = {
                    "exe": "", "database": "", "license": "",
                    "outputdirectory": "", "threads": "0",
                }
                with configpath.open("w", encoding="utf-8") as stream:
                    config.write(stream)
        self.input_file = nn.InputFile(self.template_path, configpath=configpath)
        if self.input_file.product != "nextnano++":
            raise ValueError("The input template must be a nextnano++ .nnp input.")
        return self.input_file

    def apply_parameters(self, structure: SingleQWStructure, settings: SimulationSettings) -> None:
        """Apply validated structure and physical settings to declared variables."""
        structure.validate()
        settings.validate()
        if self.input_file is None:
            self.load_template()
        variables = structure.to_input_variables() | settings.to_input_variables()
        missing = sorted(set(variables) - set(self.input_file.variables.keys()))
        if missing:
            raise ValueError("Template is missing variables: " + ", ".join(missing))
        for name, value in variables.items():
            self.input_file.set_variable(name, value=value)

    def save_input(self, path: str | Path) -> Path:
        """Save an input snapshot, refusing to overwrite an existing file."""
        if self.input_file is None:
            raise RuntimeError("Load a template and apply parameters before saving.")
        destination = Path(path).expanduser().resolve()
        if destination.exists():
            raise FileExistsError(f"Input snapshot already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        return Path(self.input_file.save(destination, overwrite=False)).resolve()

    def run(self, output_directory: str | Path, *, show_log: bool = True) -> Path:
        """Run synchronously and reject failed/nonconverged solver calculations.

        The destination must be empty. Each main.py invocation allocates a fresh
        run directory, so previous data cannot be mistaken for a successful run.
        """
        if self.input_file is None:
            raise RuntimeError("Prepare and save an input file before execution.")
        if Path(self.input_file.fullpath).resolve() == self.template_path:
            raise RuntimeError("Save an input snapshot before executing the solver.")
        configuration = self.input_file.config
        for key in ("exe", "database", "license"):
            value = str(configuration.get("nextnano++", key)).strip()
            if not value or not Path(value).expanduser().is_file():
                raise FileNotFoundError(
                    f"Set a valid nextnano++ {key} path in your nextnanopy "
                    "configuration. See README.md; --prepare-only needs no solver."
                )
        destination = Path(output_directory).expanduser().resolve()
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f"Solver output directory must be empty: {destination}")
        destination.mkdir(parents=True, exist_ok=True)
        info = self.input_file.execute(
            show_log=show_log,
            convergenceCheck=True,
            convergence_check_mode="terminate",
            create_subdirectory=False,
            outputdirectory=str(destination),
        )
        self.execution_info = info
        process = info.get("process")
        if process is None or process.returncode is None:
            raise RuntimeError("The solver did not return a completed process.")
        if process.returncode != 0:
            raise RuntimeError(
                f"nextnano++ exited with code {process.returncode}. "
                f"See {info.get('logfile', destination)}"
            )
        if not any(path.is_file() and path.stat().st_size > 0 for path in destination.rglob("*.dat")):
            raise RuntimeError("The solver produced no nonempty .dat output. Inspect its log.")
        return Path(info["outputdirectory"]).resolve()
