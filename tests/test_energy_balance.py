"""Run from the repository root with: python -m unittest discover -s tests -v.

Checks the plasma parameters the energy balance derives from its inputs, which
are given in cm^-3 while the collision formulas work in SI units.
"""

import unittest

import numpy as np
from scipy import constants

from colradpy.energy_balance import coulomb_logarithm, energy_balance
from colradpy.solve_matrix_exponential import solve_matrix_exponential


class StubIonizationBalance:
    """Neutral, singly and doubly charged states with fixed rates (s^-1)."""

    def __init__(self, electron_dens, times):
        self.rates = np.array([[-2.0, 1.0, 0.0], [2.0, -2.0, 1.0], [0.0, 1.0, -1.0]])
        self.data = {"user": {
            "init_abund": np.array([1.0, 0.0, 0.0]),
            "soln_times": times,
            "dens_grid": np.array([electron_dens]),
        }}

    def populate_ion_matrix(self):
        # ionization_balance stores rate coefficients; solve multiplies by n_e.
        density = self.data["user"]["dens_grid"][0]
        self.data["ion_matrix"] = (self.rates / density)[:, :, None, None]

    def solve_no_source(self, n0, td_t):
        pops, eigenvalues, eigenvectors = solve_matrix_exponential(
            self.rates[:, :, None, None], n0, td_t)
        self.data["processed"] = {
            "pops_td": pops, "eigen_val": eigenvalues, "eigen_vec": eigenvectors,
        }


def make_balance(electron_temp, electron_dens):
    times = np.array([1e-3, 1e-2, 1e-1])
    balance = energy_balance.__new__(energy_balance)
    balance.ion_balance = StubIonizationBalance(electron_dens, times)
    balance.data = {"user": {
        "mass": 12.0, "polarizability": 12.0,
        "electron_temp": electron_temp, "electron_dens": electron_dens,
        "ion_temp": electron_temp, "ion_dens": electron_dens, "ion_mass": 2.0,
        "ion_charge_number": 1, "init_temp": np.array([1.0, 1.0, 1.0]),
    }}
    return balance


class EnergyBalanceTests(unittest.TestCase):
    def test_debye_length_uses_density_in_si_units(self):
        electron_temp, electron_dens = 10.0, 1e13  # eV, cm^-3
        balance = make_balance(electron_temp, electron_dens)
        balance.solve()
        # lambda_D = sqrt(epsilon_0 T_e / (n_e e)) with T_e in eV and n_e in m^-3.
        expected = np.sqrt(constants.epsilon_0 * electron_temp
                           / (electron_dens * 1e6 * constants.e))
        np.testing.assert_allclose(balance.debye_length, expected, rtol=1e-12)
        np.testing.assert_allclose(balance.debye_length, 7.43394e-6, rtol=1e-5)
        self.assertTrue(np.isfinite(balance.data["processed"]["temperatures"]).all())

    def test_coulomb_logarithm_for_carbon_on_deuterium(self):
        # C2+ on D+ at 10 eV with n_e = 1e13 cm^-3; a Debye length computed from
        # the density in cm^-3 gave ln(Lambda) = 17.07 instead.
        balance = make_balance(10.0, 1e13)
        balance.solve()
        log = coulomb_logarithm(
            masses=np.array([12.0, 2.0]) * constants.value("atomic mass constant"),
            charges=np.array([2, 1]),
            temperatures=np.array([10.0, 10.0]),
            debye_length=balance.debye_length,
        )
        np.testing.assert_allclose(log, 10.15863, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
