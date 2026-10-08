"""Command-line entry point for the InGaN single-QW workflow."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

from pl_simulation import (
    NextnanoSimulation, PeakAnalyzer, QWResults, SimulationSettings,
    SingleQWStructure,
)

PROJECT_ROOT = Path(__file__).resolve().parent


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON configuration must contain an object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def allocate_run(output_root: Path) -> Path:
    """Atomically reserve a new numbered run without overwriting old results."""
    output_root.mkdir(parents=True, exist_ok=True)
    for index in range(1, 1_000_000):
        candidate = output_root / f"run_{index:03d}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError("No free run directory available.")


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute a single InGaN QW band diagram and PL spectrum using nextnano++."
    )
    parser.add_argument("--structure", type=Path, default=PROJECT_ROOT / "configs/single_qw.json")
    parser.add_argument("--settings", type=Path, default=PROJECT_ROOT / "configs/simulation.json")
    parser.add_argument("--template", type=Path, default=PROJECT_ROOT / "templates/single_qw_pl.nnp")
    parser.add_argument("--nextnano-config", type=Path, help="Existing nextnanopy .ini configuration.")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true", help="Prepare inputs without starting a solver.")
    mode.add_argument("--results-dir", type=Path, help="Analyze existing nextnano++ outputs without running a solver.")
    parser.add_argument("--include-states", action="store_true", help="Overlay eigenstates from configured output mappings.")
    parser.add_argument("--include-wavefunctions", action="store_true", help="Overlay wavefunctions from configured output mappings.")
    parser.add_argument("--quiet", action="store_true", help="Do not stream the solver log.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    run_directory: Path | None = None
    manifest: dict[str, Any] = {}
    try:
        structure = SingleQWStructure.from_dict(read_json(arguments.structure))
        simulation_config = read_json(arguments.settings)
        unknown = set(simulation_config) - {"settings", "output_files"}
        if unknown:
            raise ValueError("Unknown simulation configuration keys: " + ", ".join(sorted(unknown)))
        settings = SimulationSettings.from_dict(simulation_config["settings"])
        mappings = simulation_config.get("output_files", {})
        if not isinstance(mappings, dict):
            raise ValueError("output_files must be a JSON object.")
        if arguments.results_dir is not None and not arguments.results_dir.is_dir():
            raise FileNotFoundError(f"Existing results directory not found: {arguments.results_dir}")

        run_directory = allocate_run(arguments.output_root.expanduser().resolve())
        snapshots = run_directory / "inputs"
        write_json(snapshots / "single_qw.json", structure.to_dict())
        write_json(snapshots / "simulation.json", simulation_config)
        manifest = {
            "status": "preparing",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "excitation_model": "prescribed_quasi_fermi_levels",
            "structure": structure.to_dict(),
            "settings": settings.to_dict(),
            "output_files": mappings,
            "solver_validation": "Required: run with a licensed nextnano++ 3.0+ installation.",
        }
        write_json(snapshots / "run.json", manifest)

        if arguments.results_dir is None:
            simulation = NextnanoSimulation(arguments.template, arguments.nextnano_config)
            simulation.load_template()
            simulation.apply_parameters(structure, settings)
            prepared_input = simulation.save_input(snapshots / "single_qw_pl.nnp")
            manifest["input_file"] = str(prepared_input)
            if arguments.prepare_only:
                manifest["status"] = "prepared"
                write_json(snapshots / "run.json", manifest)
                print(f"Input prepared: {prepared_input}")
                print("No physical simulation was run. Remove --prepare-only to execute nextnano++.")
                return 0
            manifest["status"] = "running"
            write_json(snapshots / "run.json", manifest)
            source = simulation.run(run_directory / "nextnano", show_log=not arguments.quiet)
        else:
            source = arguments.results_dir.expanduser().resolve()
            manifest["analysis_of_existing_results"] = True
            manifest["excitation_model"] = "external_results_conditions_unverified"

        manifest["source_directory"] = str(source)
        results = QWResults(source, files=mappings)
        # Load and validate both required scientific outputs before making plots.
        results.load_band_edges()
        spectrum = results.load_spectrum()
        analyzer = PeakAnalyzer(spectrum)
        summary = analyzer.find_spectrum_peak()
        for key in ("source", "source_axis", "source_axis_unit", "spectral_density", "conversion"):
            if key in spectrum:
                summary[key] = spectrum[key]
        figures = run_directory / "figures"
        analysis = run_directory / "analysis"
        results.plot_band_diagram(
            figures / "quantum_well.png", structure=structure,
            include_states=arguments.include_states,
            include_wavefunctions=arguments.include_wavefunctions,
        )
        analyzer.plot_pl_spectrum(figures / "pl_spectrum.png")
        analyzer.export_csv(analysis / "pl_spectrum.csv")
        # Retain the excitation/reference conditions alongside the peak result.
        summary["excitation_model"] = manifest["excitation_model"]
        summary["configured_electron_fermi_ev"] = settings.electron_fermi_ev
        summary["configured_hole_fermi_ev"] = settings.hole_fermi_ev
        summary["settings_verified_against_generated_input"] = arguments.results_dir is None
        summary["source_directory"] = str(source)
        summary["orientation"] = {"x_hkl": list(settings.x_hkl), "y_hkl": list(settings.y_hkl)}
        write_json(analysis / "pl_summary.json", summary)
        manifest["status"] = "completed"
        write_json(snapshots / "run.json", manifest)
        print(f"Results saved: {run_directory}")
        print(f"PL peak: {summary['wavelength_nm']:.3f} nm ({summary['energy_ev']:.4f} eV)")
        return 0
    except (ValueError, KeyError, OSError, RuntimeError, ImportError) as error:
        if run_directory is not None:
            manifest.update(status="failed", error=str(error))
            write_json(run_directory / "inputs/run.json", manifest)
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
