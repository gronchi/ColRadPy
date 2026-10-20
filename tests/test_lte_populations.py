"""Analytic checks for Boltzmann populations and Saha stage balance."""

import unittest

import numpy as np
from scipy import constants

from colradpy.saha_boltzmann_lte_pops import boltzmann_population, saha_population


EV_PER_CM = constants.h * constants.c * 100 / constants.e


class LTEPopulationTests(unittest.TestCase):
    def test_boltzmann_two_levels_and_total_normalization(self):
        temperatures = np.array([1.0, 2.0])
        w = np.array([0.0, 1.0])
        energy = np.array([100.0, 100.0 + 1.0 / EV_PER_CM])
        excited = 3 * np.exp(-1 / temperatures)
        relative = boltzmann_population(w, energy, temperatures)
        np.testing.assert_allclose(relative[0], 1.0)
        np.testing.assert_allclose(relative[1], excited)
        normalized = boltzmann_population(w, energy, temperatures, norm_tot=True)
        np.testing.assert_allclose(normalized.sum(axis=0), 1.0)
        np.testing.assert_allclose(normalized[1] / normalized[0], excited)
        self.assertEqual(boltzmann_population(w, energy, 1.0).shape, (2, 1))

    def test_saha_two_stage_partition_functions_and_density(self):
        temp = np.array([1.0, 2.0])
        density = np.array([1e16, 2e16])
        # Lower stage: J=0 ground and J=1 excited at 1 eV.
        # Upper stage: J=1/2 ground and J=1/2 excited at 2 eV.
        w_z = np.array([0.0, 1.0])
        w_z1 = np.array([0.5, 0.5])
        energy_z = np.array([0.0, 1.0 / EV_PER_CM])
        energy_z1 = np.array([0.0, 2.0 / EV_PER_CM])
        ci = 4.0 / EV_PER_CM
        partition_z = 1 + 3 * np.exp(-1 / temp)
        partition_z1 = 2 * (1 + np.exp(-2 / temp))
        thermal = 2 * (2 * np.pi * constants.m_e * constants.e * temp /
                       constants.h**2)**1.5 / 1e6
        expected = thermal / density * partition_z1 / partition_z * np.exp(-4 / temp)
        actual = saha_population(ci, w_z, w_z1, energy_z, energy_z1, temp, density)
        np.testing.assert_allclose(actual, expected, rtol=1e-14)
        np.testing.assert_allclose(
            saha_population(ci, w_z, w_z1, energy_z, energy_z1, temp, density,
                            norm_tot=True), expected / (1 + expected))
        np.testing.assert_allclose(
            saha_population(ci, w_z, w_z1, energy_z, energy_z1, temp, 1e16),
            expected * density / 1e16)

    def test_invalid_physical_inputs(self):
        with self.assertRaises(ValueError):
            boltzmann_population([0, 1], [0, 1], [0])
        with self.assertRaises(ValueError):
            saha_population(1, [0], [0], [0], [0], [1], 0)
        with self.assertRaises(ValueError):
            saha_population(1, [0], [0], [0], [0], [1, 2], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
