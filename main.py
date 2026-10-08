"""Run a standalone effective-mass single-QW and relative PL calculation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

from pl_simulation import (
    PeakAnalyzer, QuantumWellSimulation, QWResults, SimulationSettings,
    SingleQWStructure,
)
from pl_simulation.material_parameters import load_materials

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
    """Reserve a numbered directory without overwriting earlier runs."""
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
        description="Calculate a single QW and relative PL directly in Python; no external solver required."
    )
    parser.add_argument("--structure", type=Path, default=PROJECT_ROOT / "configs/single_qw.json")
    parser.add_argument("--settings", type=Path, default=PROJECT_ROOT / "configs/simulation.json")
    parser.add_argument("--materials", type=Path, default=PROJECT_ROOT / "configs/materials.json")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true", help="Validate and snapshot inputs without calculation.")
    mode.add_argument("--results-dir", type=Path, help="Reanalyze a native result.npz file/folder, or explicitly mapped legacy .dat outputs.")
    parser.add_argument("--include-states", action="store_true", help="Overlay configured legacy states; native states are included by default.")
    parser.add_argument("--include-wavefunctions", action="store_true", help="Overlay wavefunctions with arbitrary display scaling.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    run_directory: Path | None = None
    manifest: dict[str, Any] = {}
    try:
        structure = SingleQWStructure.from_dict(read_json(arguments.structure))
        configuration = read_json(arguments.settings)
        unknown = set(configuration) - {"settings", "output_files"}
        if unknown:
            raise ValueError("Unknown simulation configuration keys: " + ", ".join(sorted(unknown)))
        settings = SimulationSettings.from_dict(configuration["settings"])
        materials = load_materials(arguments.materials)
        mappings = configuration.get("output_files", {})
        if not isinstance(mappings, dict):
            raise ValueError("output_files must be a JSON object.")
        existing = None
        if arguments.results_dir is not None:
            existing = arguments.results_dir.expanduser().resolve()
            if not existing.exists():
                raise FileNotFoundError(f"Existing results not found: {existing}")

        run_directory = allocate_run(arguments.output_root.expanduser().resolve())
        snapshots = run_directory / "inputs"
        write_json(snapshots / "single_qw.json", structure.to_dict())
        write_json(snapshots / "simulation.json", configuration)
        write_json(snapshots / "materials.json", materials)
        manifest = {
            "status": "prepared",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "calculation_engine": "python_effective_mass",
            "excitation_model": "prescribed_sheet_densities",
            "structure": structure.to_dict(),
            "settings": settings.to_dict(),
        }
        write_json(snapshots / "run.json", manifest)
        if arguments.prepare_only:
            print(f"Inputs prepared: {snapshots}")
            print("No calculation was run. Remove --prepare-only to compute in Python.")
            return 0

        native = True
        if existing is None:
            manifest["status"] = "running"
            write_json(snapshots / "run.json", manifest)
            calculation = QuantumWellSimulation(structure, settings, materials).run()
            calculation["metadata"].update({
                "structure": structure.to_dict(), "settings": settings.to_dict(),
                "materials": materials,
            })
            results = QWResults.from_calculation(calculation)
            source = results.save_calculation(run_directory / "calculation/result.npz")
        else:
            source = existing / "result.npz" if existing.is_dir() else existing
            if source.is_file() and source.suffix.lower() == ".npz":
                results = QWResults.from_archive(source)
                if "structure" in results.metadata:
                    structure = SingleQWStructure.from_dict(results.metadata["structure"])
                    manifest["structure"] = structure.to_dict()
                    write_json(snapshots / "single_qw.json", structure.to_dict())
                if "settings" in results.metadata:
                    manifest["settings"] = results.metadata["settings"]
                    write_json(snapshots / "simulation.json", {"settings": results.metadata["settings"]})
                if "materials" in results.metadata:
                    write_json(snapshots / "materials.json", results.metadata["materials"])
            elif existing.is_dir() and mappings:
                # Optional DataFile import only; no solver execution or benchmark.
                native = False
                source = existing
                results = QWResults(existing, files=mappings)
            else:
                raise ValueError("Specify a result.npz archive/directory, or configure output_files for legacy .dat import.")
            manifest["analysis_of_existing_results"] = True
            manifest["calculation_engine"] = "python_effective_mass" if native else "legacy_file_import"
            manifest["excitation_model"] = "archived_calculation" if native else "external_conditions_unverified"

        manifest["source"] = str(source)
        results.load_band_edges()
        spectrum = results.load_spectrum()
        analyzer = PeakAnalyzer(spectrum)
        summary = analyzer.find_spectrum_peak()
        for key in ("source", "source_axis", "source_axis_unit", "spectral_density", "conversion"):
            if key in spectrum:
                summary[key] = spectrum[key]
        results.plot_band_diagram(
            run_directory / "figures/quantum_well.png", structure=structure,
            include_states=native or arguments.include_states,
            include_wavefunctions=arguments.include_wavefunctions,
        )
        analyzer.plot_pl_spectrum(run_directory / "figures/pl_spectrum.png")
        analyzer.export_csv(run_directory / "analysis/pl_spectrum.csv")
        summary.update({
            "calculation_engine": manifest["calculation_engine"],
            "source": str(source),
            "calculation_metadata": results.metadata if native else {},
            "intensity_interpretation": "relative_spectral_density" if native else "configured_legacy_units",
        })
        write_json(run_directory / "analysis/pl_summary.json", summary)
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
