"""Native solver checks against analytic quantum mechanics, not nextnano++."""

import json
from pathlib import Path
import unittest

import numpy as np
from scipy.constants import electron_volt, hbar, m_e, Boltzmann

from pl_simulation.material_parameters import load_materials
from pl_simulation.quantum_well_simulation import QuantumWellSimulation
from pl_simulation.simulation_settings import SimulationSettings
from pl_simulation.single_qw_structure import SingleQWStructure


ROOT = Path(__file__).resolve().parents[1]
KINETIC_EV_NM2 = hbar**2 / (2 * m_e * electron_volt) * 1e18
KB_EV = Boltzmann / electron_volt


class EnvelopeHamiltonianTests(unittest.TestCase):
    def test_infinite_box_eigenvalues_converge_to_analytic_solution(self):
        length, mass = 10.0, 0.2
        exact = KINETIC_EV_NM2 / mass * (np.arange(1, 4) * np.pi / length)**2
        errors = []
        for intervals in (80, 160):
            x = np.linspace(0, length, intervals + 1)
            energies, _ = QuantumWellSimulation.solve_states(
                x, np.zeros_like(x), np.full_like(x, mass), 3)
            np.testing.assert_allclose(energies, exact, rtol=0.0013)
            errors.append(np.max(np.abs(energies - exact)))
        self.assertLess(errors[1], errors[0] / 3.8)

    def test_wavefunctions_are_orthonormal_and_obey_dirichlet_boundaries(self):
        x = np.linspace(0, 8, 161)
        _, waves = QuantumWellSimulation.solve_states(
            x, np.zeros_like(x), np.where(x < 4, 0.2, 0.4), 5)
        self.assertEqual(waves.shape, (5, len(x)))
        np.testing.assert_allclose(waves[:, [0, -1]], 0, atol=1e-14)
        np.testing.assert_allclose(waves @ waves.T * (x[1] - x[0]), np.eye(5), atol=1e-10)

    def test_global_potential_shift_preserves_level_spacings_and_envelopes(self):
        x = np.linspace(0, 10, 201)
        mass = np.full_like(x, 0.25)
        potential = np.where((x > 3) & (x < 7), 0.0, 0.3)
        original, waves = QuantumWellSimulation.solve_states(x, potential, mass, 3)
        shifted, shifted_waves = QuantumWellSimulation.solve_states(x, potential + 7.0, mass, 3)
        np.testing.assert_allclose(shifted - original, 7, atol=1e-11)
        np.testing.assert_allclose(abs(shifted_waves), abs(waves), atol=1e-9)

    def test_uniform_mass_scaling_has_expected_inverse_energy_scaling(self):
        x = np.linspace(0, 10, 121)
        light, _ = QuantumWellSimulation.solve_states(x, np.zeros_like(x), np.full_like(x, 0.2), 3)
        heavy, _ = QuantumWellSimulation.solve_states(x, np.zeros_like(x), np.full_like(x, 0.4), 3)
        np.testing.assert_allclose(light, 2 * heavy, rtol=1e-10)

    def test_mass_interface_and_asymmetric_potential_are_invariant_under_reflection(self):
        x = np.linspace(0, 10, 201)
        mass = np.where(x < 4, 0.18, 0.32)
        potential = np.where((x > 2) & (x < 6), 0.0, 0.45)
        energies, waves = QuantumWellSimulation.solve_states(x, potential, mass, 3)
        mirrored_energies, mirrored_waves = QuantumWellSimulation.solve_states(
            x, potential[::-1], mass[::-1], 3)
        np.testing.assert_allclose(mirrored_energies, energies, atol=1e-11)
        np.testing.assert_allclose(abs(mirrored_waves[:, ::-1]), abs(waves), atol=1e-9)


class OccupationTests(unittest.TestCase):
    def test_chemical_potential_reproduces_total_sheet_density(self):
        energies = np.array([0.05, 0.15, 0.28])
        masses = np.array([0.18, 0.21, 0.25])
        for temperature in (20, 300, 800):
            for requested in (1e9, 1e12, 1e14):
                with self.subTest(temperature=temperature, requested=requested):
                    mu = QuantumWellSimulation.chemical_potential(energies, masses, requested, temperature)
                    thermal = KB_EV * temperature
                    density = np.sum(masses * thermal / (2 * np.pi * KINETIC_EV_NM2)
                                     * np.logaddexp(0, (mu - energies) / thermal)) * 1e14
                    self.assertAlmostEqual(density / requested, 1.0, places=8)

    def test_single_subband_matches_closed_form_fermi_energy(self):
        energy, mass, density, temperature = 0.25, 0.2, 1e12, 300
        thermal = KB_EV * temperature
        dos_times_thermal = mass * thermal / (2 * np.pi * KINETIC_EV_NM2)
        analytic = energy + thermal * np.log(np.expm1(density / 1e14 / dos_times_thermal))
        actual = QuantumWellSimulation.chemical_potential(
            np.array([energy]), np.array([mass]), density, temperature)
        self.assertAlmostEqual(actual, analytic, places=10)


class NativeCalculationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.materials = load_materials(ROOT / "configs/materials.json")
        cls.structure = SingleQWStructure.from_dict(json.loads(
            (ROOT / "configs/single_qw.json").read_text(encoding="utf-8")))

    def test_default_calculation_produces_bound_states_and_real_positive_pl(self):
        result = QuantumWellSimulation(self.structure, SimulationSettings(), self.materials).run()
        x = result["position_nm"]
        self.assertTrue(np.all(np.diff(x) > 0))
        self.assertGreater(len(result["electron_energies_ev"]), 0)
        self.assertGreater(len(result["valence_energies_ev"]), 0)
        for waves in (result["electron_wavefunctions"], result["hole_wavefunctions"]):
            np.testing.assert_allclose(np.sum(waves**2, axis=1) * (x[1] - x[0]), 1, atol=1e-9)
        self.assertLess(max(result["electron_energies_ev"]),
                        min(result["conduction_ev"][[0, -1]]))
        self.assertGreater(min(result["valence_energies_ev"]),
                           max(result["valence_ev"][[0, -1]]))
        emission = result["energy_intensity"]
        self.assertTrue(np.all(np.isfinite(emission)))
        self.assertTrue(np.all(emission >= 0))
        self.assertGreater(np.max(emission), 0)
        self.assertGreater(np.argmax(emission), 0)
        self.assertLess(np.argmax(emission), len(emission) - 1)

    def test_structure_without_confined_states_is_rejected(self):
        data = self.structure.to_dict()
        data["layers"][1]["indium_fraction"] = 1e-8
        data["layers"][1]["thickness_nm"] = 0.1
        structure = SingleQWStructure.from_dict(data)
        with self.assertRaisesRegex(ValueError, "bound|confined"):
            QuantumWellSimulation(structure, SimulationSettings(), self.materials).run()


if __name__ == "__main__":
    unittest.main()
