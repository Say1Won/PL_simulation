"""Standalone scalar effective-mass quantum well and relative PL calculation.

No external solver is used.  The optical calculation assumes one parabolic
electron band, one decoupled hole band, vertical transitions, and a constant
interband momentum matrix element.  It predicts a *relative* spectrum; a
laser-power-to-carrier-density or absolute radiative-efficiency model is not
included.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq
from scipy.special import expit, logsumexp, roots_legendre

from .material_parameters import build_band_profiles


class QuantumWellSimulation:
    """Solve a single QW using prescribed electron and hole sheet densities.

    Lengths are in nm, energies in eV, and masses are relative to the free
    electron mass.  Wavefunctions are real envelopes normalized such that
    ``sum(abs(psi)**2) * dx_nm == 1``.  Dirichlet boundaries describe isolated
    wells only when the outer barriers are sufficiently thick.
    """

    HBAR2_OVER_2M0_EV_NM2 = 0.0380998211615486
    BOLTZMANN_EV_K = 8.617333262145e-5
    MAX_GRID_POINTS = 20_000
    MAX_SPECTRUM_POINTS = 20_000

    def __init__(self, structure: Any, settings: Any, materials: dict[str, Any]):
        self.structure = structure
        self.settings = settings
        self.materials = materials

    @staticmethod
    def solve_states(
        position_nm: np.ndarray,
        potential_ev: np.ndarray,
        mass_z: np.ndarray,
        requested_states: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the lowest Dirichlet states of the BenDaniel-Duke operator.

        The flux coefficient ``1/m`` is harmonically averaged at each cell
        face, i.e. ``2 / (m_left + m_right)``.  This treats a mass discontinuity
        as two half-cell resistances and produces a symmetric Hamiltonian.
        Only the requested low eigenpairs are computed.
        """
        x = np.asarray(position_nm, dtype=float)
        potential = np.asarray(potential_ev, dtype=float)
        mass = np.asarray(mass_z, dtype=float)
        if x.ndim != 1 or len(x) < 4 or len(x) > QuantumWellSimulation.MAX_GRID_POINTS:
            raise ValueError("The spatial grid must contain between 4 and 20000 points.")
        if potential.shape != x.shape or mass.shape != x.shape:
            raise ValueError("Potential, mass and position must have matching one-dimensional shapes.")
        if not all(np.all(np.isfinite(values)) for values in (x, potential, mass)):
            raise ValueError("The spatial grid, potential and effective mass must be finite.")
        spacing = np.diff(x)
        if np.any(spacing <= 0) or not np.allclose(spacing, spacing[0], rtol=1e-8, atol=1e-12):
            raise ValueError("The eigenproblem requires a strictly increasing uniform spatial grid.")
        if np.any(mass <= 0):
            raise ValueError("Effective masses must be positive.")
        if isinstance(requested_states, bool) or not isinstance(requested_states, (int, np.integer)):
            raise ValueError("requested_states must be a positive integer.")
        if not 1 <= requested_states <= len(x) - 2:
            raise ValueError("requested_states cannot exceed the number of interior grid points.")
        dx = float(spacing[0])
        coefficient = QuantumWellSimulation.HBAR2_OVER_2M0_EV_NM2 / dx**2
        inverse_face_mass = 2.0 / (mass[:-1] + mass[1:])
        diagonal = coefficient * (inverse_face_mass[:-1] + inverse_face_mass[1:]) + potential[1:-1]
        off_diagonal = -coefficient * inverse_face_mass[1:-1]
        eigenvalues, eigenvectors = eigh_tridiagonal(
            diagonal, off_diagonal, select="i", select_range=(0, int(requested_states) - 1),
            check_finite=False,
        )
        wavefunctions = np.zeros((int(requested_states), len(x)), dtype=float)
        wavefunctions[:, 1:-1] = eigenvectors.T / math.sqrt(dx)
        return eigenvalues, wavefunctions

    @staticmethod
    def _log_sheet_density(
        mu: float, energies: np.ndarray, masses: np.ndarray, temperature_k: float,
    ) -> float:
        """Log density in nm^-2, including spin degeneracy two.

        Each 2D parabolic subband contributes
        ``m*kT/(2*pi*C) * log(1 + exp((mu-E)/kT))`` with C=hbar^2/(2*m0).
        The log form also remains usable in the nondegenerate low-density limit.
        """
        kt = QuantumWellSimulation.BOLTZMANN_EV_K * temperature_k
        argument = (mu - energies) / kt
        log_fermi_integral = np.empty_like(argument)
        dilute = argument < -30.0
        log_fermi_integral[dilute] = argument[dilute]
        log_fermi_integral[~dilute] = np.log(np.logaddexp(0.0, argument[~dilute]))
        log_dos = np.log(masses * kt / (2.0 * math.pi * QuantumWellSimulation.HBAR2_OVER_2M0_EV_NM2))
        return float(logsumexp(log_dos + log_fermi_integral))

    @staticmethod
    def chemical_potential(
        energies: np.ndarray, masses: np.ndarray, density_cm2: float, temperature_k: float,
    ) -> float:
        """Invert the occupied 2D subband density without fitting a PL peak."""
        energies = np.asarray(energies, dtype=float)
        masses = np.asarray(masses, dtype=float)
        if energies.ndim != 1 or not len(energies) or masses.shape != energies.shape:
            raise ValueError("Subband energies and masses must be nonempty matching vectors.")
        if not np.all(np.isfinite(energies)) or not np.all(np.isfinite(masses)) or np.any(masses <= 0):
            raise ValueError("Subband energies must be finite and masses positive and finite.")
        for value, name in ((density_cm2, "sheet density"), (temperature_k, "temperature")):
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"The {name} must be a positive finite number.")
        log_target = math.log(float(density_cm2)) - 14.0 * math.log(10.0)

        def residual(mu: float) -> float:
            return QuantumWellSimulation._log_sheet_density(mu, energies, masses, float(temperature_k)) - log_target

        width = max(1.0, 40.0 * QuantumWellSimulation.BOLTZMANN_EV_K * temperature_k)
        lower = float(np.min(energies)) - width
        upper = float(np.max(energies)) + width
        for _ in range(64):
            if residual(lower) <= 0 and residual(upper) >= 0:
                return float(brentq(residual, lower, upper, xtol=1e-13, rtol=1e-13))
            width *= 2.0
            lower = float(np.min(energies)) - width
            upper = float(np.max(energies)) + width
        raise ValueError("Could not bracket the requested carrier density.")

    @staticmethod
    def _subband_masses(waves: np.ndarray, local_mass: np.ndarray, dx_nm: float) -> np.ndarray:
        return 1.0 / (np.sum(waves**2 / local_mass[None, :], axis=1) * dx_nm)

    def _spectrum(
        self, energy_ev: np.ndarray, electron_energies: np.ndarray, hole_energies: np.ndarray,
        electron_masses: np.ndarray, hole_masses: np.ndarray, electron_mu: float, hole_mu: float,
        valence_top_ev: float, overlaps: np.ndarray,
    ) -> tuple[np.ndarray, list[dict[str, float]], int]:
        """Integrate broadened occupied vertical transitions over k_parallel.

        With t=k^2, the spin-degenerate 2D state measure is dt/(2*pi).
        The photon-energy factor follows a constant momentum matrix element;
        common dielectric/matrix-element prefactors are omitted.  Gaussian
        broadening has a unit-area kernel and sigma ``broadening_ev``.
        """
        settings = self.settings
        sigma = float(settings.broadening_ev)
        kt = self.BOLTZMANN_EV_K * float(settings.temperature_k)
        requested_points = int(settings.k_integration_points)
        # Composite quadrature prevents a narrow Gaussian from falling between
        # widely spaced nodes, while bounding temporary memory to small batches.
        panel_order = min(64, requested_points)
        abscissae, weights = roots_legendre(panel_order)
        result = np.zeros_like(energy_ev)
        transitions: list[dict[str, float]] = []
        largest_node_count = 0
        for e_index, electron in enumerate(electron_energies):
            alpha_e = self.HBAR2_OVER_2M0_EV_NM2 / electron_masses[e_index]
            for h_index, hole in enumerate(hole_energies):
                overlap = float(overlaps[e_index, h_index])
                if overlap <= 1e-14:
                    continue
                alpha_h = self.HBAR2_OVER_2M0_EV_NM2 / hole_masses[h_index]
                alpha_joint = alpha_e + alpha_h
                edge_ev = float(electron + hole - valence_top_ev)
                transitions.append({
                    "electron_state": int(e_index), "hole_state": int(h_index),
                    "edge_energy_ev": edge_ev, "envelope_overlap_squared": overlap,
                })
                spectral_limit = max(0.0, (float(energy_ev[-1]) + 8.0 * sigma - edge_ev) / alpha_joint)
                electron_tail = (max(0.0, electron_mu - electron) + 40.0 * kt) / alpha_e
                hole_tail = (max(0.0, hole_mu - hole) + 40.0 * kt) / alpha_h
                maximum_t = min(spectral_limit, electron_tail, hole_tail)
                if maximum_t <= 0:
                    continue
                # At least the requested total points, and resolve sigma even
                # when an unusually wide spectrum or sharp line is requested.
                panels = max(
                    1, math.ceil(requested_points / panel_order),
                    math.ceil(alpha_joint * maximum_t / (sigma * panel_order / 2.0)),
                )
                if panels * panel_order > 200_000:
                    raise ValueError("The PL quadrature is too large; increase broadening or narrow the energy window.")
                largest_node_count = max(largest_node_count, panels * panel_order)
                edges = np.linspace(0.0, maximum_t, panels + 1)
                half_width = maximum_t / (2.0 * panels)
                nodes = (edges[:-1, None] + half_width + half_width * abscissae[None, :]).ravel()
                node_weights = np.tile(weights * half_width, panels)
                photon_ev = edge_ev + alpha_joint * nodes
                f_e = expit((electron_mu - electron - alpha_e * nodes) / kt)
                f_h = expit((hole_mu - hole - alpha_h * nodes) / kt)
                rate = overlap * f_e * f_h * np.maximum(photon_ev, 0.0) * node_weights / (2.0 * math.pi)
                # Only energy/node matrices of at most ~two MiB are allocated.
                for start in range(0, len(nodes), 128):
                    stop = start + 128
                    photon_chunk = photon_ev[start:stop]
                    # Gaussian probability beyond eight sigma is <1.3e-15.
                    # Restrict each batch to its local energy range instead of
                    # computing kernels that immediately underflow to zero.
                    energy_lower = int(np.searchsorted(energy_ev, photon_chunk[0] - 8.0 * sigma))
                    energy_upper = int(np.searchsorted(energy_ev, photon_chunk[-1] + 8.0 * sigma, side="right"))
                    for energy_start in range(energy_lower, energy_upper, 2048):
                        energy_stop = min(energy_start + 2048, energy_upper)
                        delta = (energy_ev[energy_start:energy_stop, None] - photon_ev[None, start:stop]) / sigma
                        kernel = np.exp(-0.5 * delta**2) / (math.sqrt(2.0 * math.pi) * sigma)
                        result[energy_start:energy_stop] += kernel @ rate[start:stop]
        return result, transitions, largest_node_count

    def run(self) -> dict[str, Any]:
        """Return band profiles, confined states and a relative PL spectrum.

        A box-discretized continuum is never used as a confined QW state.
        Sheet densities are assigned only to the selected bound subbands.
        """
        self.structure.validate()
        self.settings.validate()
        settings = self.settings
        if settings.k_integration_points > 200_000:
            raise ValueError("k_integration_points cannot exceed 200000.")
        length_nm = sum(layer.thickness_nm for layer in self.structure.layers)
        intervals = int(math.ceil(length_nm / settings.grid_spacing_nm))
        if intervals + 1 > self.MAX_GRID_POINTS or intervals < 3:
            raise ValueError("The requested spatial grid must contain between 4 and 20000 points.")
        x = np.linspace(0.0, length_nm, intervals + 1)
        dx = float(x[1] - x[0])
        profiles = build_band_profiles(self.structure, settings, self.materials, x)
        keys = (
            "conduction_ev", "valence_ev", "electron_mass_z", "hole_mass_z",
            "electron_mass_xy", "hole_mass_xy",
        )
        for name in keys:
            profiles[name] = np.asarray(profiles[name], dtype=float)
            if profiles[name].shape != x.shape or not np.all(np.isfinite(profiles[name])):
                raise ValueError(f"Material profile {name} must be a finite vector on the spatial grid.")
            if "mass" in name and np.any(profiles[name] <= 0):
                raise ValueError(f"Material profile {name} must be positive.")
        conduction = profiles["conduction_ev"]
        valence = profiles["valence_ev"]
        if np.any(conduction <= valence):
            raise ValueError("The local conduction band must lie above the local valence band.")
        electron_candidates, electron_waves = self.solve_states(
            x, conduction, profiles["electron_mass_z"], settings.electron_states,
        )
        valence_top = float(np.max(valence))
        hole_potential = valence_top - valence
        hole_candidates, hole_waves = self.solve_states(
            x, hole_potential, profiles["hole_mass_z"], settings.hole_states,
        )
        electron_threshold = float(min(conduction[0], conduction[-1]))
        hole_threshold = float(min(hole_potential[0], hole_potential[-1]))
        e_bound = electron_candidates < electron_threshold
        h_bound = hole_candidates < hole_threshold
        if not np.any(e_bound) or not np.any(h_bound):
            raise ValueError("No bound electron/hole states were found; revise the well, field, or spatial grid.")
        electron_energies = electron_candidates[e_bound]
        electron_waves = electron_waves[e_bound]
        hole_energies = hole_candidates[h_bound]
        hole_waves = hole_waves[h_bound]
        electron_masses = self._subband_masses(electron_waves, profiles["electron_mass_xy"], dx)
        hole_masses = self._subband_masses(hole_waves, profiles["hole_mass_xy"], dx)
        electron_mu = self.chemical_potential(
            electron_energies, electron_masses, settings.electron_sheet_density_cm2, settings.temperature_k,
        )
        hole_mu = self.chemical_potential(
            hole_energies, hole_masses, settings.hole_sheet_density_cm2, settings.temperature_k,
        )
        spectrum_intervals = math.ceil(
            (settings.spectrum_energy_max_ev - settings.spectrum_energy_min_ev) / settings.spectrum_energy_step_ev,
        )
        if spectrum_intervals + 1 > self.MAX_SPECTRUM_POINTS:
            raise ValueError("The requested spectrum exceeds 20000 energy samples.")
        energy = np.linspace(settings.spectrum_energy_min_ev, settings.spectrum_energy_max_ev, spectrum_intervals + 1)
        overlaps = np.abs(electron_waves @ hole_waves.T * dx)**2
        intensity, transitions, quadrature_nodes = self._spectrum(
            energy, electron_energies, hole_energies, electron_masses, hole_masses,
            electron_mu, hole_mu, valence_top, overlaps,
        )
        if not np.all(np.isfinite(intensity)) or not np.any(intensity > 0):
            raise ValueError("No finite PL signal in the selected energy window; revise the window or carrier densities.")
        density_residuals = {}
        for carrier, mu, energies, masses, density in (
            ("electron", electron_mu, electron_energies, electron_masses, settings.electron_sheet_density_cm2),
            ("hole", hole_mu, hole_energies, hole_masses, settings.hole_sheet_density_cm2),
        ):
            difference = self._log_sheet_density(mu, energies, masses, settings.temperature_k) - (math.log(density) - 14.0 * math.log(10.0))
            density_residuals[carrier] = float(math.expm1(difference))
        warnings = []
        if int(np.sum(e_bound)) == settings.electron_states:
            warnings.append("All requested electron states are bound; increase electron_states to check omitted subbands.")
        if int(np.sum(h_bound)) == settings.hole_states:
            warnings.append("All requested hole states are bound; increase hole_states to check omitted subbands.")
        kt = self.BOLTZMANN_EV_K * settings.temperature_k
        if electron_mu > electron_threshold - 5.0 * kt or hole_mu > hole_threshold - 5.0 * kt:
            warnings.append("A chemical potential lies within five kT of a barrier continuum; omitted unbound carriers may matter.")
        if float(energy[1] - energy[0]) > settings.broadening_ev / 2.0:
            warnings.append("The spectrum sampling is coarse relative to the Gaussian sigma; refine the energy step for peak/FWHM.")
        metadata = {
            "model": "1D_scalar_effective_mass_bound_subband_relative_PL",
            "energy_reference": profiles.get("metadata", {}).get("energy_reference", "unstrained_GaN_VBM"),
            "energy_intensity_unit": "relative/eV",
            "excitation_model": "prescribed_electron_and_hole_sheet_densities",
            "boundary_condition": "Dirichlet_zero_envelopes_at_outer_barrier_edges",
            "actual_grid_spacing_nm": dx,
            "actual_spectrum_step_ev": float(energy[1] - energy[0]),
            "bound_electron_states": int(len(electron_energies)),
            "bound_hole_states": int(len(hole_energies)),
            "electron_continuum_threshold_ev": electron_threshold,
            "hole_quasiparticle_continuum_threshold_ev": hole_threshold,
            "electron_chemical_potential_ev": electron_mu,
            "hole_quasiparticle_chemical_potential_ev": hole_mu,
            "hole_valence_quasi_fermi_ev": valence_top - hole_mu,
            "hole_quasiparticle_energy_reference_ev": valence_top,
            "electron_sheet_density_cm2": float(settings.electron_sheet_density_cm2),
            "hole_sheet_density_cm2": float(settings.hole_sheet_density_cm2),
            "density_relative_residuals": density_residuals,
            "broadening_model": "unit_area_Gaussian",
            "broadening_sigma_ev": float(settings.broadening_ev),
            "optical_prefactor": "photon_energy_times_constant_interband_momentum_matrix_element",
            "maximum_k_quadrature_nodes": quadrature_nodes,
            "transitions": transitions,
            "material_profiles": profiles.get("metadata", {}),
            "warnings": warnings,
            "limitations": [
                "Unstrained scalar electron and decoupled hole effective-mass bands; no multiband valence mixing.",
                "No self-consistent Poisson, doping, automatic polarization charge, carrier screening, or excitons.",
                "Only bound subbands are occupied; continuum-carrier capture and recombination are omitted.",
                "Prescribed sheet densities do not determine laser power, lifetime, nonradiative rates, or absolute PL intensity.",
                "Outer-barrier thickness, spatial grid, subband counts, and k quadrature require convergence checks.",
            ],
        }
        return {
            "position_nm": x, "conduction_ev": conduction, "valence_ev": valence,
            "electron_energies_ev": electron_energies,
            "valence_energies_ev": valence_top - hole_energies,
            "electron_wavefunctions": electron_waves, "hole_wavefunctions": hole_waves,
            "electron_mass_xy": electron_masses, "hole_mass_xy": hole_masses,
            "energy_ev": energy, "energy_intensity": intensity, "metadata": metadata,
        }
